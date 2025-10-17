"""
Data Loader for GNN-RNN Hybrid Landslide Risk Model

This module provides dataset and data loader classes for loading:
1. Static features (GNN input) from graph structure
2. Dynamic time series (RNN input) from rainfall/InSAR/NDVI data
3. Labels (landslide occurrence)

Author: Landslide Risk Analysis Project
Date: 2025-01-15
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class LandslideDataset(Dataset):
    """
    Dataset for landslide risk prediction with configurable dynamic features

    Args:
        graph_path: Path to graph_data.pt (contains static features + graph structure)
        rainfall_path: Path to ldaps_slope_statistics.csv (wide format)
        samples_path: Path to landslide_samples.csv (cat, event_date, label)
        window_size: Number of days in temporal window (default: 5)
        use_insar: Whether to include InSAR displacement features (default: False)
        use_ndvi: Whether to include NDVI difference features (default: False)
        insar_path: Path to InSAR time series data (optional)
        ndvi_path: Path to NDVI difference data (optional)
        start_date: Start date for valid samples (format: 'YYYYMMDD')
        end_date: End date for valid samples (format: 'YYYYMMDD')
    """

    def __init__(
        self,
        graph_path: str,
        rainfall_path: str,
        samples_path: str,
        window_size: int = 5,
        use_insar: bool = False,
        use_ndvi: bool = False,
        insar_path: Optional[str] = None,
        ndvi_path: Optional[str] = None,
        start_date: str = '20200511',
        end_date: str = '20200920',
        include_target_day: bool = True
    ):
        super().__init__()

        self.window_size = window_size
        self.use_insar = use_insar
        self.use_ndvi = use_ndvi
        self.include_target_day = include_target_day

        # Date range validation
        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')

        # Load graph data
        print(f"Loading graph from {graph_path}...")
        self.graph_data = torch.load(graph_path, weights_only=False)
        self.static_features = self.graph_data.x  # (num_nodes, 15)
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr

        # Create cat to node_idx mapping
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.graph_data.cat.numpy())}
        print(f"  Graph loaded: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")

        # Load rainfall data (wide format)
        print(f"Loading rainfall data from {rainfall_path}...")
        self.rainfall_df = pd.read_csv(rainfall_path)
        print(f"  Rainfall data loaded: {len(self.rainfall_df)} slope units")

        # Extract available date range from rainfall columns
        self.rainfall_dates = self._extract_rainfall_dates()
        print(f"  Available rainfall dates: {self.rainfall_dates[0]} to {self.rainfall_dates[-1]}")

        # Load InSAR data (if enabled)
        if self.use_insar:
            if insar_path is None:
                raise ValueError("insar_path must be provided when use_insar=True")
            print(f"Loading InSAR data from {insar_path}...")
            self.insar_df = pd.read_csv(insar_path)
            print(f"  InSAR data loaded")
        else:
            self.insar_df = None

        # Load NDVI data (if enabled)
        if self.use_ndvi:
            if ndvi_path is None:
                raise ValueError("ndvi_path must be provided when use_ndvi=True")
            print(f"Loading NDVI data from {ndvi_path}...")
            self.ndvi_df = pd.read_csv(ndvi_path)
            print(f"  NDVI data loaded")
        else:
            self.ndvi_df = None

        # Load samples
        print(f"Loading samples from {samples_path}...")
        samples_df = pd.read_csv(samples_path, encoding='utf-8-sig')  # Handle BOM
        samples_df['event_date'] = pd.to_datetime(samples_df['event_date'], format='%Y-%m-%d')

        # Filter samples by date range
        samples_df = samples_df[
            (samples_df['event_date'] >= self.start_date) &
            (samples_df['event_date'] <= self.end_date)
        ]

        # Filter samples with valid cat and available rainfall data
        valid_samples = []
        for _, row in samples_df.iterrows():
            cat = int(row['cat'])
            event_date = row['event_date']

            # Check if cat exists in graph
            if cat not in self.cat_to_node:
                continue

            # Check if cat exists in rainfall data
            if cat not in self.rainfall_df['cat'].values:
                continue

            # Check if window dates are available
            window_start = event_date - timedelta(days=window_size - 1)
            if not self._check_date_range_available(window_start, event_date):
                continue

            valid_samples.append({
                'cat': cat,
                'event_date': event_date,
                'label': int(row['label'])
            })

        self.samples = valid_samples
        print(f"  Valid samples: {len(self.samples)} (after filtering)")
        print(f"  Positive samples: {sum(s['label'] == 1 for s in self.samples)}")
        print(f"  Negative samples: {sum(s['label'] == 0 for s in self.samples)}")

        # Calculate dynamic feature dimension
        self.dynamic_dim = self._calculate_dynamic_dim()
        print(f"  Dynamic feature dimension: {self.dynamic_dim}")

    def _extract_rainfall_dates(self) -> List[str]:
        """Extract sorted list of dates from rainfall column names"""
        date_cols = [col for col in self.rainfall_df.columns if col.startswith('acc3d_mean_')]
        dates = sorted([col.split('_')[-1] for col in date_cols])
        return dates

    def _check_date_range_available(self, start_date: datetime, end_date: datetime) -> bool:
        """Check if all dates in range are available in rainfall data"""
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y%m%d')
            if date_str not in self.rainfall_dates:
                return False
            current += timedelta(days=1)
        return True

    def _calculate_dynamic_dim(self) -> int:
        """Calculate total dynamic feature dimension"""
        dim = 6  # Rainfall features (always included)
        if self.use_insar:
            dim += 2  # cumulative_displacement, displacement_delay_days
        if self.use_ndvi:
            dim += 1  # ndvi_diff
        return dim

    def _get_rainfall_features(self, cat: int, date: datetime) -> np.ndarray:
        """
        Get rainfall features for a specific cat and date

        Returns:
            features: (6,) array [acc3d_mean, acc3d_max, acc7d_mean,
                                   acc7d_max, peak1h_mean, peak1h_max]
        """
        date_str = date.strftime('%Y%m%d')

        # Find row for this cat
        cat_rows = self.rainfall_df[self.rainfall_df['cat'] == cat]
        
        if len(cat_rows) == 0:
            raise ValueError(f"Cat {cat} not found in rainfall data")
        
        row = cat_rows.iloc[0]

        features = np.array([
            row[f'acc3d_mean_{date_str}'],
            row[f'acc3d_max_{date_str}'],
            row[f'acc7d_mean_{date_str}'],
            row[f'acc7d_max_{date_str}'],
            row[f'peak1h_mean_{date_str}'],
            row[f'peak1h_max_{date_str}']
        ], dtype=np.float32)

        return features

    def _get_insar_features(self, cat: int, date: datetime) -> np.ndarray:
        """
        Get InSAR displacement features for a specific cat and date

        Returns:
            features: (2,) array [cumulative_displacement, displacement_delay_days]
        """
        if not self.use_insar or self.insar_df is None:
            raise RuntimeError("InSAR not enabled")

        # TODO: Implement InSAR data loading
        # This is a placeholder implementation
        # Actual implementation depends on InSAR data format

        # Placeholder: return zeros
        displacement = 0.0
        delay = 0  # Days since last valid stack

        return np.array([displacement, delay], dtype=np.float32)

    def _get_ndvi_features(self, cat: int, date: datetime) -> np.ndarray:
        """
        Get NDVI difference features for a specific cat and date

        Returns:
            features: (1,) array [ndvi_diff]
        """
        if not self.use_ndvi or self.ndvi_df is None:
            raise RuntimeError("NDVI not enabled")

        # TODO: Implement NDVI data loading
        # This is a placeholder implementation

        # Placeholder: return zero
        ndvi_diff = 0.0

        return np.array([ndvi_diff], dtype=np.float32)

    def _create_temporal_window(self, cat: int, target_date: datetime, 
                               include_target_day: bool = True) -> np.ndarray:
        """
        Create temporal window of dynamic features

        Args:
            cat: Slope unit ID
            target_date: Target date (event date)
            include_target_day: If True, use [-4, -3, -2, -1, 0] (default)
                               If False, use [-5, -4, -3, -2, -1] (for real-time prediction)

        Returns:
            window_data: (window_size, dynamic_dim) array
        """
        window_data = []

        # Determine day offset range based on include_target_day
        if include_target_day:
            day_range = range(-self.window_size + 1, 1)  # [-4, -3, -2, -1, 0]
        else:
            day_range = range(-self.window_size, 0)      # [-5, -4, -3, -2, -1]

        for day_offset in day_range:
            current_date = target_date + timedelta(days=day_offset)

            # Rainfall features (always included)
            rainfall_features = self._get_rainfall_features(cat, current_date)

            # Combine features
            features = [rainfall_features]

            if self.use_insar:
                insar_features = self._get_insar_features(cat, current_date)
                features.append(insar_features)

            if self.use_ndvi:
                # NDVI diff is same for entire window (use target date)
                ndvi_features = self._get_ndvi_features(cat, target_date)
                features.append(ndvi_features)

            # Concatenate all features
            daily_features = np.concatenate(features)
            window_data.append(daily_features)

        window_data = np.array(window_data, dtype=np.float32)  # (window_size, dynamic_dim)

        return window_data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Returns:
            sample: Dictionary containing:
                - node_idx: Node index in graph (int)
                - cat: Slope unit ID (int)
                - event_date: Event date (str)
                - dynamic_features: (window_size, dynamic_dim) tensor
                - label: Binary label (float)
        """
        sample_info = self.samples[idx]
        cat = sample_info['cat']
        event_date = sample_info['event_date']
        label = sample_info['label']

        # Get node index
        node_idx = self.cat_to_node[cat]

        # Get dynamic features
        dynamic_features = self._create_temporal_window(cat, event_date, self.include_target_day)

        return {
            'node_idx': torch.tensor(node_idx, dtype=torch.long),
            'cat': torch.tensor(cat, dtype=torch.long),
            'event_date': event_date.strftime('%Y%m%d'),
            'dynamic_features': torch.tensor(dynamic_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class LandslideCollator:
    """
    Custom collator for batching landslide samples

    Args:
        graph_data: PyTorch Geometric Data object (static features + graph structure)
    """

    def __init__(self, graph_data: Data):
        self.graph_data = graph_data

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples

        Args:
            batch: List of sample dictionaries

        Returns:
            batch_dict: Dictionary containing:
                - static_x: (num_nodes, 15) - all node features
                - edge_index: (2, num_edges) - graph connectivity
                - edge_attr: (num_edges, 1) - edge weights
                - node_indices: (batch_size,) - indices of batch samples
                - dynamic_x: (batch_size, window_size, dynamic_dim) - time series
                - labels: (batch_size,) - binary labels
                - cats: (batch_size,) - slope unit IDs
                - event_dates: List[str] - event dates
        """
        node_indices = torch.stack([sample['node_idx'] for sample in batch])
        dynamic_x = torch.stack([sample['dynamic_features'] for sample in batch])
        labels = torch.stack([sample['label'] for sample in batch])
        cats = torch.stack([sample['cat'] for sample in batch])
        event_dates = [sample['event_date'] for sample in batch]

        return {
            'static_x': self.graph_data.x,
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr,
            'node_indices': node_indices,
            'dynamic_x': dynamic_x,
            'labels': labels,
            'cats': cats,
            'event_dates': event_dates
        }


def create_dataloaders(
    graph_path: str,
    rainfall_path: str,
    samples_path: str,
    batch_size: int = 256,
    window_size: int = 5,
    use_insar: bool = False,
    use_ndvi: bool = False,
    insar_path: Optional[str] = None,
    ndvi_path: Optional[str] = None,
    start_date: str = '20200511',
    end_date: str = '20200920',
    train_ratio: float = 0.8,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, LandslideDataset]:
    """
    Create train and validation data loaders

    Args:
        graph_path: Path to graph_data.pt
        rainfall_path: Path to rainfall statistics CSV
        samples_path: Path to landslide samples CSV
        batch_size: Batch size
        window_size: Temporal window size (days)
        use_insar: Whether to use InSAR features
        use_ndvi: Whether to use NDVI features
        insar_path: Path to InSAR data (if use_insar=True)
        ndvi_path: Path to NDVI data (if use_ndvi=True)
        start_date: Start date for valid samples
        end_date: End date for valid samples
        train_ratio: Ratio of training data
        random_seed: Random seed for reproducibility
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: Full dataset object (for reference)
    """
    # Create dataset
    dataset = LandslideDataset(
        graph_path=graph_path,
        rainfall_path=rainfall_path,
        samples_path=samples_path,
        window_size=window_size,
        use_insar=use_insar,
        use_ndvi=use_ndvi,
        insar_path=insar_path,
        ndvi_path=ndvi_path,
        start_date=start_date,
        end_date=end_date
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"\nDataset split:")
    print(f"  Training: {train_size} samples")
    print(f"  Validation: {val_size} samples")

    # Create collator
    collator = LandslideCollator(dataset.graph_data)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, dataset


# Example usage
if __name__ == "__main__":
    # Paths (adjust as needed)
    GRAPH_PATH = r"D:\Landslide\data\model_ready\graph_data.pt"
    RAINFALL_PATH = r"D:\Landslide\data\model_ready\ldaps_slope_statistics.csv"
    SAMPLES_PATH = r"D:\Landslide\data\processed\gyeongnam\landslide_samples.csv"

    # Create data loaders
    train_loader, val_loader, dataset = create_dataloaders(
        graph_path=GRAPH_PATH,
        rainfall_path=RAINFALL_PATH,
        samples_path=SAMPLES_PATH,
        batch_size=32,
        window_size=5,
        use_insar=False,  # Disable InSAR for now
        use_ndvi=False,   # Disable NDVI for now
        start_date='20200511',
        end_date='20200920',
        train_ratio=0.8,
        random_seed=42
    )

    # Test batch loading
    print("\n" + "="*70)
    print("Testing data loader...")
    print("="*70)

    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Static features: {batch['static_x'].shape}")
        print(f"  Edge index: {batch['edge_index'].shape}")
        print(f"  Node indices: {batch['node_indices'].shape}")
        print(f"  Dynamic features: {batch['dynamic_x'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        print(f"  Label distribution: {batch['labels'].sum():.0f} positive, "
              f"{(batch['labels'].shape[0] - batch['labels'].sum()):.0f} negative")

        if batch_idx == 0:  # Only show first batch
            break

    print("\n" + "="*70)
    print("Data loader test completed!")
    print("="*70)
