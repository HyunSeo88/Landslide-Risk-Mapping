"""
Negative Sampling Strategies for Landslide Risk Model

This module provides various negative sampling strategies to balance
positive (landslide) and negative (stable) samples.

Strategies:
1. Random: Completely random sampling
2. Temporal-matched: Match temporal distribution of positive samples
3. Hard negative mining: Focus on difficult samples (after initial training)
4. Mixed: Combination of above strategies
"""

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def sample_negatives_random(
    positive_samples: List[Dict],
    all_slope_ids: List[int],
    date_range: tuple,
    ratio: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Random negative sampling

    Args:
        positive_samples: List of positive sample dicts
        all_slope_ids: List of all available slope IDs
        date_range: (start_date, end_date) as datetime objects
        ratio: Negative to positive ratio (1.0 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        negative_samples: List of negative sample dicts
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Extract positive slope IDs
    pos_slope_ids = set(s['cat'] for s in positive_samples)

    # Get non-landslide slopes
    neg_slope_ids = [sid for sid in all_slope_ids if sid not in pos_slope_ids]

    if len(neg_slope_ids) == 0:
        raise ValueError("No negative slope IDs available")

    # Calculate number of negative samples
    n_negative = int(len(positive_samples) * ratio)

    # Random sampling
    start_date, end_date = date_range
    total_days = (end_date - start_date).days + 1

    negative_samples = []
    for _ in range(n_negative):
        # Random slope (with replacement to ensure we get enough samples)
        slope_id = random.choice(neg_slope_ids)

        # Random date
        random_days = random.randint(0, total_days - 1)
        event_date = start_date + timedelta(days=random_days)

        negative_samples.append({
            'cat': slope_id,
            'event_date': event_date,
            'label': 0
        })

    return negative_samples


def sample_negatives_temporal_matched(
    positive_samples: List[Dict],
    all_slope_ids: List[int],
    ratio: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Temporal-matched negative sampling
    Matches the temporal distribution of positive samples

    Args:
        positive_samples: List of positive sample dicts
        all_slope_ids: List of all available slope IDs
        ratio: Negative to positive ratio (1.0 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        negative_samples: List of negative sample dicts
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Extract positive slope IDs and dates
    pos_slope_ids = set(s['cat'] for s in positive_samples)
    pos_dates = [s['event_date'] for s in positive_samples]

    # Get non-landslide slopes
    neg_slope_ids = [sid for sid in all_slope_ids if sid not in pos_slope_ids]

    if len(neg_slope_ids) == 0:
        raise ValueError("No negative slope IDs available")

    # Calculate temporal distribution
    date_counts = pd.Series(pos_dates).value_counts().sort_index()

    # Sample negatives matching temporal distribution
    negative_samples = []

    for date, count in date_counts.items():
        # Number of negatives for this date
        n_neg_for_date = int(count * ratio)

        # Sample slopes for this date
        if n_neg_for_date > 0:
            sampled_slopes = np.random.choice(
                neg_slope_ids,
                size=min(n_neg_for_date, len(neg_slope_ids)),
                replace=False
            )

            for slope_id in sampled_slopes:
                negative_samples.append({
                    'cat': int(slope_id),
                    'event_date': date,
                    'label': 0
                })

    # If not enough samples due to limited neg_slope_ids, fill randomly
    if len(negative_samples) < len(positive_samples) * ratio:
        n_missing = int(len(positive_samples) * ratio) - len(negative_samples)
        additional_samples = sample_negatives_random(
            positive_samples, all_slope_ids,
            (min(pos_dates), max(pos_dates)),
            ratio=n_missing / len(positive_samples),
            random_seed=random_seed
        )
        negative_samples.extend(additional_samples)

    return negative_samples


def sample_negatives_temporal_matched_pure(
    positive_samples: List[Dict],
    all_slope_ids: List[int],
    ratio: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Pure temporal-matched negative sampling (without random fill-up)
    Used in mixed strategy to avoid double random sampling

    Args:
        positive_samples: List of positive sample dicts
        all_slope_ids: List of all available slope IDs
        ratio: Negative to positive ratio (1.0 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        negative_samples: List of negative sample dicts
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Extract positive slope IDs and dates
    pos_slope_ids = set(s['cat'] for s in positive_samples)
    pos_dates = [s['event_date'] for s in positive_samples]

    # Get non-landslide slopes
    neg_slope_ids = [sid for sid in all_slope_ids if sid not in pos_slope_ids]

    if len(neg_slope_ids) == 0:
        raise ValueError("No negative slope IDs available")

    # Calculate temporal distribution
    date_counts = pd.Series(pos_dates).value_counts().sort_index()

    # Sample negatives matching temporal distribution
    negative_samples = []

    for date, count in date_counts.items():
        # Number of negatives for this date
        n_neg_for_date = int(count * ratio)

        # Sample slopes for this date
        if n_neg_for_date > 0:
            sampled_slopes = np.random.choice(
                neg_slope_ids,
                size=min(n_neg_for_date, len(neg_slope_ids)),
                replace=False
            )

            for slope_id in sampled_slopes:
                negative_samples.append({
                    'cat': int(slope_id),
                    'event_date': date,
                    'label': 0
                })

    # NOTE: No random fill-up here (unlike the original function)
    return negative_samples


def sample_negatives_hard(
    positive_samples: List[Dict],
    negative_pool: List[Dict],
    predictions: np.ndarray,
    hard_threshold: tuple = (0.3, 0.7),
    hard_ratio: float = 0.5,
    ratio: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Hard negative mining
    Prioritize samples that model finds difficult to classify

    Args:
        positive_samples: List of positive sample dicts
        negative_pool: Pool of negative sample dicts
        predictions: Model predictions for negative_pool (probabilities)
        hard_threshold: (low, high) threshold for hard negatives
        hard_ratio: Ratio of hard negatives in final sample
        ratio: Negative to positive ratio (1.0 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        negative_samples: List of negative sample dicts
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    n_total = int(len(positive_samples) * ratio)
    n_hard = int(n_total * hard_ratio)
    n_random = n_total - n_hard

    # Find hard negatives (predictions in [low, high] range)
    low, high = hard_threshold
    hard_mask = (predictions >= low) & (predictions <= high)
    hard_indices = np.where(hard_mask)[0]

    # Sample hard negatives
    if len(hard_indices) > 0:
        # Sort by proximity to 0.5 (most uncertain)
        uncertainty = np.abs(predictions[hard_indices] - 0.5)
        sorted_indices = hard_indices[np.argsort(uncertainty)]

        # Take top n_hard most uncertain
        sampled_hard_indices = sorted_indices[:min(n_hard, len(sorted_indices))]
        hard_negatives = [negative_pool[i] for i in sampled_hard_indices]
    else:
        hard_negatives = []

    # Adjust random sample count if not enough hard negatives
    n_random = n_total - len(hard_negatives)

    # Sample random negatives from pool
    available_indices = list(set(range(len(negative_pool))) - set(sampled_hard_indices if hard_negatives else []))
    if len(available_indices) >= n_random:
        sampled_random_indices = random.sample(available_indices, n_random)
    else:
        sampled_random_indices = available_indices

    random_negatives = [negative_pool[i] for i in sampled_random_indices]

    # Combine
    negative_samples = hard_negatives + random_negatives

    return negative_samples


def sample_negatives_mixed(
    positive_samples: List[Dict],
    all_slope_ids: List[int],
    date_range: tuple,
    temporal_ratio: float = 0.5,
    ratio: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Mixed strategy: Combine temporal-matched and random sampling

    Args:
        positive_samples: List of positive sample dicts
        all_slope_ids: List of all available slope IDs
        date_range: (start_date, end_date) as datetime objects
        temporal_ratio: Ratio of temporal-matched samples (0.5 = 50% temporal, 50% random)
        ratio: Negative to positive ratio (1.0 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        negative_samples: List of negative sample dicts
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    n_total = int(len(positive_samples) * ratio)
    n_temporal = int(n_total * temporal_ratio)
    n_random = n_total - n_temporal

    # Pure temporal-matched sampling (without random fill-up)
    temporal_samples = sample_negatives_temporal_matched_pure(
        positive_samples, all_slope_ids,
        ratio=n_temporal / len(positive_samples),
        random_seed=random_seed
    )

    # Random sampling for the remaining
    random_samples = sample_negatives_random(
        positive_samples, all_slope_ids, date_range,
        ratio=n_random / len(positive_samples),
        random_seed=random_seed + 1 if random_seed is not None else None
    )

    negative_samples = temporal_samples + random_samples

    return negative_samples


def create_balanced_samples(
    positive_samples_path: str,
    all_slope_ids: List[int],
    strategy: str = 'temporal',
    ratio: float = 1.0,
    start_date: str = '2020311',
    end_date: str = '20200920',
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create balanced dataset with positive and negative samples

    Args:
        positive_samples_path: Path to CSV with positive samples
        all_slope_ids: List of all available slope IDs
        strategy: Sampling strategy ('random', 'temporal', 'mixed')
        ratio: Negative to positive ratio (1.0 = 1:1)
        start_date: Start date for sampling (YYYYMMDD)
        end_date: End date for sampling (YYYYMMDD)
        random_seed: Random seed for reproducibility
        output_path: Path to save balanced dataset (optional)

    Returns:
        balanced_df: DataFrame with balanced samples
    """
    # Load positive samples
    pos_df = pd.read_csv(positive_samples_path, encoding='utf-8-sig')
    pos_df['event_date'] = pd.to_datetime(pos_df['event_date'], format='%Y-%m-%d')

    # Filter by date range
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')

    pos_df = pos_df[
        (pos_df['event_date'] >= start_dt) &
        (pos_df['event_date'] <= end_dt)
    ]

    positive_samples = pos_df.to_dict('records')

    print(f"Positive samples: {len(positive_samples)}")

    # Sample negative samples
    if strategy == 'random':
        negative_samples = sample_negatives_random(
            positive_samples, all_slope_ids, (start_dt, end_dt),
            ratio=ratio, random_seed=random_seed
        )
    elif strategy == 'temporal':
        negative_samples = sample_negatives_temporal_matched(
            positive_samples, all_slope_ids,
            ratio=ratio, random_seed=random_seed
        )
    elif strategy == 'mixed':
        negative_samples = sample_negatives_mixed(
            positive_samples, all_slope_ids, (start_dt, end_dt),
            temporal_ratio=0.5, ratio=ratio, random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Negative samples: {len(negative_samples)}")

    # Convert to DataFrame
    neg_df = pd.DataFrame(negative_samples)

    # Combine positive and negative
    balanced_df = pd.concat([pos_df[['cat', 'event_date', 'label']], neg_df], ignore_index=True)

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(f"  Positive: {(balanced_df['label'] == 1).sum()}")
    print(f"  Negative: {(balanced_df['label'] == 0).sum()}")

    # Save if output path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        balanced_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    return balanced_df


# Example usage
if __name__ == "__main__":
    # Load graph to get all slope IDs
    import torch

    GRAPH_PATH = r"D:\Landslide\data\model_ready\graph_data_v3.pt"
    POSITIVE_SAMPLES_PATH = r"D:\Landslide\data\model_ready\gyeongnam_positive_samples.csv"
    OUTPUT_PATH = r"D:\Landslide\data\model_ready\balanced_samples.csv"

    # Load graph
    graph_data = torch.load(GRAPH_PATH, weights_only=False)
    all_slope_ids = graph_data.cat.numpy().tolist()

    print(f"Total slope IDs: {len(all_slope_ids)}")

    # Create balanced dataset
    balanced_df = create_balanced_samples(
        positive_samples_path=POSITIVE_SAMPLES_PATH,
        all_slope_ids=all_slope_ids,
        strategy='temporal',  # 'random', 'temporal', 'mixed'   #####지금 random, mixed 오류 있음.
        ratio=1.0,  # 1:1 ratio
        start_date='20200101',
        end_date='20200930',
        random_seed=42,
        output_path=OUTPUT_PATH
    )

    print("\n" + "="*70)
    print("Sample distribution by date:")
    print("="*70)
    date_dist = balanced_df.groupby([balanced_df['event_date'].dt.date, 'label']).size().unstack(fill_value=0)
    print(date_dist.head(10))
