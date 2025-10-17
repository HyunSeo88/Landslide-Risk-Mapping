"""
SBAS Interferogram Pair Generation for Landslide Study
Generates optimal interferogram pairs based on temporal/spatial baseline criteria
with event-focused star connections
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import json

# Configuration
MAX_TEMPORAL_BASELINE = 48  # days
MAX_PERPENDICULAR_BASELINE = None  # Set to None to disable B_perp filtering (or value in meters)
SENTINEL1_CYCLE_DAYS = 12  # or 6 depending on acquisition mode

# Critical landslide events (YYYY-MM-DD) - Based on actual occurrence data
CRITICAL_EVENTS = [
    "2020-06-12",  # First event cluster
    "2020-06-29",  # Second event
    "2020-07-12",  # Third event
    "2020-07-23",  # Fourth event
    "2020-07-28",  # MAJOR EVENT - very high occurrence
    "2020-09-01",  # MAJOR EVENT - very high occurrence
]

# Event anchor offsets (days before/after event)
EVENT_ANCHORS = [-36, -24, -12, 12, 24, 36]


def load_acquisition_dates(csv_path=None):
    """
    Load Sentinel-1 acquisition dates
    Expected CSV columns: date, orbit, B_perp (optional)
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    else:
        # Example: generate dates for 2020-03 to 2020-09 with 12-day cycle
        start = datetime(2020, 3, 11)
        end = datetime(2020, 9, 19)  # Last available data
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=SENTINEL1_CYCLE_DAYS)

        df = pd.DataFrame({
            'date': dates,
            'orbit': ['ASCENDING'] * len(dates),  # Adjust as needed
            'B_perp': [0] * len(dates)  # Placeholder, compute from actual orbit data
        })

    df = df.sort_values('date').reset_index(drop=True)
    return df


def generate_baseline_network(df, max_dt=MAX_TEMPORAL_BASELINE):
    """
    Generate basic temporal baseline network (Δt ≤ max_dt)
    Creates pairs at 12, 24, 36, 48 day intervals
    """
    pairs = []
    dates = df['date'].tolist()

    for i, date1 in enumerate(dates):
        for j in range(i + 1, len(dates)):
            date2 = dates[j]
            dt = (date2 - date1).days

            if dt > max_dt:
                break

            # Prefer multiples of 12 days (Sentinel-1 cycle)
            if dt in [12, 24, 36, 48]:
                b_perp = abs(df.loc[j, 'B_perp'] - df.loc[i, 'B_perp'])

                # Check B_perp threshold if enabled
                if MAX_PERPENDICULAR_BASELINE is None or b_perp < MAX_PERPENDICULAR_BASELINE:
                    pairs.append({
                        'primary': date1.strftime('%Y%m%d'),
                        'secondary': date2.strftime('%Y%m%d'),
                        'dt_days': dt,
                        'B_perp_m': b_perp,
                        'type': 'baseline'
                    })

    return pairs


def generate_event_star_connections(df, events=CRITICAL_EVENTS, anchors=EVENT_ANCHORS):
    """
    Generate event-focused star connections
    Connect each event date to anchors at specified offsets
    """
    pairs = []
    dates = df['date'].tolist()

    for event_str in events:
        event = datetime.strptime(event_str, '%Y-%m-%d')

        # Find nearest acquisition to event
        event_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - event).days))
        event_date = dates[event_idx]

        for offset in anchors:
            target = event_date + timedelta(days=offset)

            # Find nearest acquisition to target
            target_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - target).days))
            target_date = dates[target_idx]

            dt = abs((target_date - event_date).days)

            if dt > 0 and dt <= 48:  # Skip self-pairs and respect max baseline
                primary_idx = min(event_idx, target_idx)
                secondary_idx = max(event_idx, target_idx)

                b_perp = abs(df.loc[secondary_idx, 'B_perp'] - df.loc[primary_idx, 'B_perp'])

                # Check B_perp threshold if enabled
                if MAX_PERPENDICULAR_BASELINE is None or b_perp < MAX_PERPENDICULAR_BASELINE:
                    pair = {
                        'primary': dates[primary_idx].strftime('%Y%m%d'),
                        'secondary': dates[secondary_idx].strftime('%Y%m%d'),
                        'dt_days': dt,
                        'B_perp_m': b_perp,
                        'type': f'event_{event_str}'
                    }

                    # Check if already exists
                    if not any(p['primary'] == pair['primary'] and p['secondary'] == pair['secondary'] for p in pairs):
                        pairs.append(pair)

    return pairs


def merge_and_deduplicate_pairs(baseline_pairs, event_pairs):
    """Merge baseline and event pairs, remove duplicates"""
    all_pairs = baseline_pairs + event_pairs

    # Deduplicate based on primary-secondary combination
    unique_pairs = []
    seen = set()

    for pair in all_pairs:
        key = (pair['primary'], pair['secondary'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    return unique_pairs


def export_pair_list(pairs, output_csv='sbas_pairs.csv', output_json='sbas_pairs.json'):
    """Export pair list to CSV and JSON"""
    df = pd.DataFrame(pairs)
    df = df.sort_values(['primary', 'secondary']).reset_index(drop=True)

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} pairs to {output_csv}")

    # Convert numpy types to native Python types for JSON serialization
    pairs_json = []
    for pair in pairs:
        pairs_json.append({
            'primary': str(pair['primary']),
            'secondary': str(pair['secondary']),
            'dt_days': int(pair['dt_days']),
            'B_perp_m': float(pair['B_perp_m']),
            'type': str(pair['type'])
        })

    # Save JSON (for scripting)
    with open(output_json, 'w') as f:
        json.dump(pairs_json, f, indent=2)
    print(f"Exported {len(pairs_json)} pairs to {output_json}")

    # Print statistics
    print(f"\nPair Statistics:")
    print(f"  Total pairs: {len(df)}")
    print(f"  Baseline network: {len(df[df['type'] == 'baseline'])}")
    print(f"  Event-focused: {len(df[df['type'] != 'baseline'])}")
    print(f"  Temporal baseline: {df['dt_days'].min()}-{df['dt_days'].max()} days")
    print(f"  Mean dt: {df['dt_days'].mean():.1f} days")

    return df


def main():
    # Load acquisition dates
    print("Loading acquisition dates...")
    df = load_acquisition_dates()  # Or pass CSV path
    print(f"Found {len(df)} acquisition dates from {df['date'].min()} to {df['date'].max()}")

    # Generate baseline network
    print("\nGenerating baseline network (Δt ≤ 48 days)...")
    baseline_pairs = generate_baseline_network(df)
    print(f"  Created {len(baseline_pairs)} baseline pairs")

    # Generate event-focused connections
    print("\nGenerating event-focused star connections...")
    event_pairs = generate_event_star_connections(df)
    print(f"  Created {len(event_pairs)} event pairs")

    # Merge and deduplicate
    print("\nMerging and deduplicating...")
    all_pairs = merge_and_deduplicate_pairs(baseline_pairs, event_pairs)

    # Export
    output_dir = "C:/Users/user/AIRLab/research/Landslide/sentinel-1/sbas/"
    export_pair_list(all_pairs,
                     output_csv=f"{output_dir}sbas_pairs.csv",
                     output_json=f"{output_dir}sbas_pairs.json")


if __name__ == "__main__":
    main()
