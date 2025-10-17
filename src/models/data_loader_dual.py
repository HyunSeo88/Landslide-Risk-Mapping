"""
Data Loader for Dual-Stream Hierarchical GNN-U-Net

This module provides dataset and data loader for the dual-stream architecture:
- State Stream: Static terrain rasters (spatial texture)
- Trigger Stream: Temporal dynamic sequences (window_size=5)

Key Features:
- Modular dynamic variable configuration
- Temporal window construction (5-day sequences)
- Separate State/Trigger loading
- Flexible normalization strategies

Author: Landslide Risk Analysis Project  
Date: 2025-01-17
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window

from src.utils.raster_utils import compute_slope_bboxes


# ============================================================
# Dual-Stream Dataset
# ============================================================

class DualStreamDataset(Dataset):
    """
    Dual-Stream dataset for hierarchical model
    
    Returns patches with:
    - GNN context (1 channel, from Stage 1)
    - State features (N_state channels, static terrain)
    - Trigger sequence (T=5, N_trigger channels per timestep, dynamic)
    
    Args:
        graph_path: Path to graph data
        samples_path: Path to samples CSV
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to slope bboxes cache
        gnn_raster_path: Path to GNN susceptibility raster (from Stage 1)
        static_raster_dir: Directory containing static rasters
        static_variables: List of static variable names
        dynamic_raster_base: Base path to dynamic rasters
        dynamic_variables: Dict of dynamic variable configurations
        patch_size: Patch size
        window_size: Temporal window size (default: 5)
        start_date: Start date
        end_date: End date
    """
    
    def __init__(
        self,
        graph_path: str,
        samples_path: str,
        slope_id_raster_path: str,
        slope_bboxes_cache_path: str,
        gnn_raster_path: str,
        static_raster_dir: str,
        static_variables: List[str],
        dynamic_raster_base: str,
        dynamic_variables: Dict[str, List[str]],
        patch_size: int = 128,
        window_size: int = 5,
        start_date: str = '20190101',
        end_date: str = '20200930'
    ):
        super().__init__()
        
        self.slope_id_raster_path = slope_id_raster_path
        self.gnn_raster_path = gnn_raster_path
        self.static_raster_dir = static_raster_dir
        self.static_variables = static_variables
        self.dynamic_raster_base = dynamic_raster_base
        self.dynamic_variables = dynamic_variables
        self.patch_size = patch_size
        self.window_size = window_size
        
        # Date range
        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')
        
        # Load slope ID raster
        print(f"Loading slope ID raster from {slope_id_raster_path}...")
        with rasterio.open(slope_id_raster_path) as src:
            self.slope_id_raster = src.read(1)
            self.raster_height, self.raster_width = src.shape
            self.raster_transform = src.transform
            self.raster_crs = src.crs
        print(f"  Shape: {self.raster_height} × {self.raster_width}")
        
        # Load GNN raster
        print(f"Loading GNN raster from {gnn_raster_path}...")
        with rasterio.open(gnn_raster_path) as src:
            self.gnn_raster = src.read(1)
        print(f"  GNN raster loaded")
        
        # Build static raster paths
        self.static_raster_paths = []
        for var in static_variables:
            path = os.path.join(static_raster_dir, f"{var}.tif")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Static raster not found: {path}")
            self.static_raster_paths.append(path)
        print(f"  Static rasters: {len(self.static_raster_paths)} files")
        
        # Compute slope bboxes
        print("Computing slope bounding boxes...")
        self.slope_bboxes = compute_slope_bboxes(
            slope_id_raster_path=slope_id_raster_path,
            cache_path=slope_bboxes_cache_path,
            patch_size=patch_size
        )
        
        # Load graph data
        print(f"Loading graph from {graph_path}...")
        self.graph_data = torch.load(graph_path, weights_only=False)
        self.cat_values = self.graph_data.cat
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.cat_values.numpy())}
        print(f"  Graph: {self.graph_data.num_nodes} nodes")
        
        # Load samples
        print(f"Loading samples from {samples_path}...")
        samples_df = pd.read_csv(samples_path, encoding='utf-8-sig')
        samples_df['event_date'] = pd.to_datetime(samples_df['event_date'], format='%Y-%m-%d')
        
        # Filter samples
        # Must have window_size days before event_date
        samples_df = samples_df[
            (samples_df['event_date'] >= self.start_date + timedelta(days=window_size-1)) &
            (samples_df['event_date'] <= self.end_date) &
            (samples_df['cat'].isin(self.cat_to_node.keys()))
        ]
        
        self.samples = []
        for _, row in samples_df.iterrows():
            self.samples.append({
                'cat': int(row['cat']),
                'event_date': row['event_date'],
                'label': int(row['label'])
            })
        
        print(f"  Valid samples: {len(self.samples)}")
        print(f"  Positive: {sum(s['label'] == 1 for s in self.samples)}")
        print(f"  Negative: {sum(s['label'] == 0 for s in self.samples)}")
        
        # Calculate channel counts
        self.num_state_channels = len(static_variables)
        self.num_trigger_channels = sum(
            len(channels) for channels in dynamic_variables.values()
        )
        print(f"  State channels: {self.num_state_channels}")
        print(f"  Trigger channels (per timestep): {self.num_trigger_channels}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _calculate_patch_window(self, cat: int) -> Tuple[Window, Tuple[int, int, int, int]]:
        """Calculate patch window for a given slope"""
        bbox = self.slope_bboxes[cat]
        
        center_row = bbox['centroid_row']
        center_col = bbox['centroid_col']
        
        row_start = max(0, center_row - self.patch_size // 2)
        col_start = max(0, center_col - self.patch_size // 2)
        
        row_end = min(self.raster_height, row_start + self.patch_size)
        col_end = min(self.raster_width, col_start + self.patch_size)
        
        if row_end - row_start < self.patch_size:
            row_start = max(0, row_end - self.patch_size)
        if col_end - col_start < self.patch_size:
            col_start = max(0, col_end - self.patch_size)
        
        window = Window(col_start, row_start, self.patch_size, self.patch_size)
        bounds = (row_start, row_end, col_start, col_end)
        
        return window, bounds
    
    def _load_temporal_dynamic_patch(
        self,
        event_date: datetime,
        window: Window,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int
    ) -> np.ndarray:
        """
        Load temporal sequence of dynamic rasters
        
        Args:
            event_date: Event date
            window: Rasterio window
            row_start, row_end, col_start, col_end: Patch bounds
        
        Returns:
            temporal_data: (T, C_trigger, H, W) array
        """
        temporal_patches = []
        
        # Create temporal window (5 days before event)
        for t in range(self.window_size):
            # Calculate date for this timestep
            current_date = event_date - timedelta(days=self.window_size - 1 - t)
            date_str = current_date.strftime('%Y%m%d')
            
            # Load all dynamic variables for this timestep
            timestep_patches = []
            
            for var_group, channels in self.dynamic_variables.items():
                for channel in channels:
                    # Construct raster path
                    raster_path = os.path.join(
                        self.dynamic_raster_base,
                        channel,
                        f"{date_str}_{channel}_mm_5179_30m.tif"
                    )
                    
                    if not os.path.exists(raster_path):
                        # Use zeros if file not found
                        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                    else:
                        with rasterio.open(raster_path) as src:
                            patch = src.read(1, window=window).astype(np.float32)
                        
                        # Pad if needed
                        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                            patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                            h, w = patch.shape
                            patch_padded[:h, :w] = patch
                            patch = patch_padded
                    
                    # Handle NaN
                    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    timestep_patches.append(patch)
            
            # Stack channels for this timestep
            timestep_data = np.stack(timestep_patches, axis=0)  # (C_trigger, H, W)
            temporal_patches.append(timestep_data)
        
        # Stack temporal dimension
        temporal_data = np.stack(temporal_patches, axis=0)  # (T, C_trigger, H, W)
        
        return temporal_data
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample
        
        Returns:
            sample: Dictionary containing:
                - 'cat': Slope ID
                - 'zone_patch': (H, W) slope ID raster patch
                - 'gnn_patch': (1, H, W) GNN susceptibility
                - 'state_patch': (C_state, H, W) static terrain features
                - 'trigger_sequence': (T, C_trigger, H, W) temporal dynamic features
                - 'label': Binary label
        """
        sample_info = self.samples[idx]
        cat = sample_info['cat']
        event_date = sample_info['event_date']
        label = sample_info['label']
        
        # Calculate patch window
        window, (row_start, row_end, col_start, col_end) = self._calculate_patch_window(cat)
        
        # 1. Extract zone patch
        zone_patch = self.slope_id_raster[row_start:row_end, col_start:col_end].copy()
        
        # 2. Extract GNN patch
        gnn_patch = self.gnn_raster[row_start:row_end, col_start:col_end].copy()
        gnn_patch = np.where(gnn_patch < 0, 0.0, gnn_patch)  # Handle NoData
        
        # Pad if needed
        if zone_patch.shape[0] < self.patch_size or zone_patch.shape[1] < self.patch_size:
            zone_patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=zone_patch.dtype)
            gnn_patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=gnn_patch.dtype)
            
            h, w = zone_patch.shape
            zone_patch_padded[:h, :w] = zone_patch
            gnn_patch_padded[:h, :w] = gnn_patch
            
            zone_patch = zone_patch_padded
            gnn_patch = gnn_patch_padded
        
        # 3. Extract static patches (State stream)
        state_patches = []
        for raster_path in self.static_raster_paths:
            with rasterio.open(raster_path) as src:
                patch = src.read(1, window=window).astype(np.float32)
            
            # Pad if needed
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                h, w = patch.shape
                patch_padded[:h, :w] = patch
                patch = patch_padded
            
            # Handle NaN
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize (per-channel z-score)
            valid_mask = ~np.isnan(patch) & (patch != 0)
            if valid_mask.sum() > 0:
                mean = patch[valid_mask].mean()
                std = patch[valid_mask].std()
                if std > 1e-6:
                    patch = (patch - mean) / (std + 1e-6)
                else:
                    patch = patch - mean
            
            state_patches.append(patch)
        
        state_patch = np.stack(state_patches, axis=0)  # (N_state, H, W)
        
        # 4. Extract temporal dynamic patches (Trigger stream)
        trigger_sequence = self._load_temporal_dynamic_patch(
            event_date, window, row_start, row_end, col_start, col_end
        )  # (T, N_trigger, H, W)
        
        # Clip extreme values
        state_patch = np.clip(state_patch, -10, 10)
        trigger_sequence = np.clip(trigger_sequence, -100, 100)  # Rainfall can be large
        
        # Convert to tensors
        zone_patch = torch.from_numpy(zone_patch).long()
        gnn_patch = torch.from_numpy(gnn_patch[np.newaxis, ...]).float()  # (1, H, W)
        state_patch = torch.from_numpy(state_patch).float()  # (N_state, H, W)
        trigger_sequence = torch.from_numpy(trigger_sequence).float()  # (T, N_trigger, H, W)
        
        return {
            'cat': torch.tensor(cat, dtype=torch.long),
            'zone_patch': zone_patch,
            'gnn_patch': gnn_patch,
            'state_patch': state_patch,
            'trigger_sequence': trigger_sequence,
            'label': torch.tensor(label, dtype=torch.float32)
        }


# ============================================================
# Collator
# ============================================================

class DualStreamCollator:
    """
    Collator for dual-stream batches
    
    Args:
        graph_data: Graph data (for compatibility)
    """
    
    def __init__(self, graph_data):
        self.graph_data = graph_data
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch of samples"""
        cats = torch.stack([s['cat'] for s in batch])
        zone_patches = torch.stack([s['zone_patch'] for s in batch])
        gnn_patches = torch.stack([s['gnn_patch'] for s in batch])
        state_patches = torch.stack([s['state_patch'] for s in batch])
        trigger_sequences = torch.stack([s['trigger_sequence'] for s in batch])
        labels = torch.stack([s['label'] for s in batch])
        
        return {
            'cats': cats,
            'zone_patches': zone_patches,
            'gnn_patches': gnn_patches,
            'state_patches': state_patches,
            'trigger_sequences': trigger_sequences,
            'labels': labels,
            # Keep graph data for compatibility
            'static_x': self.graph_data.x,
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr
        }


# ============================================================
# DataLoader Factory
# ============================================================

def create_dual_dataloaders(
    graph_path: str,
    samples_path: str,
    slope_id_raster_path: str,
    slope_bboxes_cache_path: str,
    gnn_raster_path: str,
    static_raster_dir: str,
    static_variables: List[str],
    dynamic_raster_base: str,
    dynamic_variables: Dict[str, List[str]],
    patch_size: int = 128,
    window_size: int = 5,
    batch_size: int = 16,
    start_date: str = '20190101',
    end_date: str = '20200930',
    train_ratio: float = 0.8,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DualStreamDataset]:
    """
    Create train and validation data loaders for dual-stream model
    
    Args:
        graph_path: Path to graph data
        samples_path: Path to samples CSV
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to slope bboxes cache
        gnn_raster_path: Path to GNN raster (from Stage 1)
        static_raster_dir: Directory containing static rasters
        static_variables: List of static variable names
        dynamic_raster_base: Base path to dynamic rasters
        dynamic_variables: Dict of dynamic variable configurations
        patch_size: Patch size
        window_size: Temporal window size
        batch_size: Batch size
        start_date: Start date
        end_date: End date
        train_ratio: Training ratio
        random_seed: Random seed
        num_workers: Number of workers
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: Full dataset
    """
    # Create dataset
    dataset = DualStreamDataset(
        graph_path=graph_path,
        samples_path=samples_path,
        slope_id_raster_path=slope_id_raster_path,
        slope_bboxes_cache_path=slope_bboxes_cache_path,
        gnn_raster_path=gnn_raster_path,
        static_raster_dir=static_raster_dir,
        static_variables=static_variables,
        dynamic_raster_base=dynamic_raster_base,
        dynamic_variables=dynamic_variables,
        patch_size=patch_size,
        window_size=window_size,
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
    collator = DualStreamCollator(dataset.graph_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, dataset


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing Dual-Stream Data Loader")
    print("="*70)
    
    print("\nThis module requires actual data files to test.")
    print("Use in training pipeline with real data paths.")
    
    print("\n" + "="*70)
    print("Dual-stream data loader module loaded successfully!")
    print("="*70)

