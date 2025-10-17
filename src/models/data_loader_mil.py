"""
MIL Data Loader for Hierarchical GNN-U-Net Model

This module provides dataset and data loader for Multiple Instance Learning (MIL)
with raster-based inputs.

Key Components:
- LandslideRasterDataset: Loads multi-channel raster stacks for each sample
- MILCollator: Batches samples with slope-pixel mapping for MIL aggregation

Author: Landslide Risk Analysis Project
Date: 2025-01-16
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd

from src.utils.raster_utils import RasterConverter, DynamicRasterLoader, compute_slope_bboxes
import rasterio
from rasterio.windows import Window


class LandslideRasterDataset(Dataset):
    """
    Patch-based dataset for MIL learning with GNN-U-Net
    
    Each sample consists of:
    - Slope ID (cat)
    - Event date
    - Label (landslide occurrence: 0/1)
    
    Returns a 128x128 patch containing the slope, along with:
    - GNN-derived susceptibility values mapped to patch pixels
    - Dynamic raster values for the patch
    - Slope mask (bag mask for MIL)
    
    Args:
        graph_path: Path to graph_data.pt (for static features)
        samples_path: Path to samples CSV (cat, event_date, label)
        slope_polygons_path: Path to slope polygons GeoPackage (unused in patch mode)
        raster_base_path: Base path to dynamic raster files
        reference_raster_path: Path to reference raster (for metadata)
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to cached slope bboxes
        patch_size: Patch size (default: 128)
        dynamic_variables: List of dynamic variables to load
        dynamic_statistics: List of statistics to load (empty for pixel-level)
        start_date: Start date for valid samples
        end_date: End date for valid samples
    """
    
    def __init__(
        self,
        graph_path: str,
        samples_path: str,
        slope_polygons_path: str,
        raster_base_path: str,
        reference_raster_path: str,
        slope_id_raster_path: str,
        slope_bboxes_cache_path: str,
        patch_size: int = 128,
        dynamic_variables: List[str] = ['acc3d', 'acc7d', 'peak1h'],
        dynamic_statistics: List[str] = [],
        start_date: str = '20200301',
        end_date: str = '20200930'
    ):
        super().__init__()
        
        self.graph_path = graph_path
        self.samples_path = samples_path
        self.slope_polygons_path = slope_polygons_path
        self.raster_base_path = raster_base_path
        self.reference_raster_path = reference_raster_path
        self.slope_id_raster_path = slope_id_raster_path
        self.patch_size = patch_size
        self.dynamic_variables = dynamic_variables
        self.dynamic_statistics = dynamic_statistics
        
        # Date range
        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')
        
        # Load slope ID raster (memory mapped for efficiency)
        print(f"Loading slope ID raster from {slope_id_raster_path}...")
        with rasterio.open(slope_id_raster_path) as src:
            self.slope_id_raster = src.read(1)
            self.raster_height, self.raster_width = src.shape
            self.raster_transform = src.transform
            self.raster_crs = src.crs
        print(f"  Shape: {self.raster_height} × {self.raster_width}")
        
        # Compute or load slope bounding boxes
        print("Computing slope bounding boxes...")
        self.slope_bboxes = compute_slope_bboxes(
            slope_id_raster_path=slope_id_raster_path,
            cache_path=slope_bboxes_cache_path,
            patch_size=patch_size
        )
        
        # Initialize dynamic raster loader
        print("Initializing dynamic raster loader...")
        self.dynamic_loader = DynamicRasterLoader(
            raster_base_path=raster_base_path,
            variables=dynamic_variables,
            statistics=dynamic_statistics
        )
        
        # Load graph data
        print(f"Loading graph from {graph_path}...")
        self.graph_data = torch.load(graph_path, weights_only=False)
        self.static_features = self.graph_data.x
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr
        self.cat_values = self.graph_data.cat
        
        # Create cat to node mapping
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.cat_values.numpy())}
        
        print(f"  Graph: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        
        # Load samples
        print(f"Loading samples from {samples_path}...")
        samples_df = pd.read_csv(samples_path, encoding='utf-8-sig')
        samples_df['event_date'] = pd.to_datetime(samples_df['event_date'], format='%Y-%m-%d')
        
        # Filter by date range
        samples_df = samples_df[
            (samples_df['event_date'] >= self.start_date) &
            (samples_df['event_date'] <= self.end_date)
        ]
        
        # Filter by valid cat
        samples_df = samples_df[samples_df['cat'].isin(self.cat_to_node.keys())]
        
        # Convert to list of dicts
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
        
        # Calculate number of channels
        # For pixel-level prediction: one channel per variable (no statistics)
        if dynamic_statistics:
            self.num_dynamic_channels = len(dynamic_variables) * len(dynamic_statistics)
        else:
            self.num_dynamic_channels = len(dynamic_variables)
        print(f"  Dynamic channels: {self.num_dynamic_channels}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _calculate_patch_window(self, cat: int) -> Tuple[Window, Tuple[int, int, int, int]]:
        """
        Calculate patch window for a given slope
        
        Args:
            cat: Slope ID
        
        Returns:
            window: Rasterio Window object
            bounds: (row_start, row_end, col_start, col_end) for array slicing
        """
        bbox = self.slope_bboxes[cat]
        
        # Calculate patch center (slope centroid)
        center_row = bbox['centroid_row']
        center_col = bbox['centroid_col']
        
        # Calculate patch bounds (centered on slope)
        row_start = max(0, center_row - self.patch_size // 2)
        col_start = max(0, center_col - self.patch_size // 2)
        
        # Ensure patch doesn't exceed raster bounds
        row_end = min(self.raster_height, row_start + self.patch_size)
        col_end = min(self.raster_width, col_start + self.patch_size)
        
        # Adjust start if we hit the boundary
        if row_end - row_start < self.patch_size:
            row_start = max(0, row_end - self.patch_size)
        if col_end - col_start < self.patch_size:
            col_start = max(0, col_end - self.patch_size)
        
        # Create rasterio Window
        window = Window(col_start, row_start, self.patch_size, self.patch_size)
        bounds = (row_start, row_end, col_start, col_end)
        
        return window, bounds
    
    def _load_dynamic_patch(self, event_date_str: str, window: Window) -> np.ndarray:
        """
        Load dynamic raster patch for a given date and window
        
        Args:
            event_date_str: Event date (YYYYMMDD)
            window: Rasterio window
        
        Returns:
            dynamic_patch: (C, H, W) array of dynamic features
        """
        patches = []
        
        for var in self.dynamic_variables:
            # Construct file path
            raster_path = os.path.join(
                self.raster_base_path,
                var,
                f"{event_date_str}_{var}_mm_5179_30m.tif"
            )
            
            if not os.path.exists(raster_path):
                raise FileNotFoundError(f"Dynamic raster not found: {raster_path}")
            
            # Load patch using window
            with rasterio.open(raster_path) as src:
                patch = src.read(1, window=window)
            
            patches.append(patch)
        
        # Stack along channel dimension: (C, H, W)
        dynamic_patch = np.stack(patches, axis=0)
        
        return dynamic_patch
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample (patch-based)
        
        Returns:
            sample: Dictionary containing:
                - 'cat': Slope ID
                - 'node_idx': Node index in graph
                - 'event_date_str': Event date (YYYYMMDD string)
                - 'zone_patch': (H, W) slope ID raster patch
                - 'dynamic_patch': (C, H, W) tensor of dynamic features
                - 'label': Binary label
        """
        sample_info = self.samples[idx]
        cat = sample_info['cat']
        event_date = sample_info['event_date']
        label = sample_info['label']
        
        # Get node index
        node_idx = self.cat_to_node[cat]
        
        # Calculate patch window
        window, (row_start, row_end, col_start, col_end) = self._calculate_patch_window(cat)
        
        # Extract zone patch from slope ID raster
        zone_patch = self.slope_id_raster[row_start:row_end, col_start:col_end].copy()
        
        # Load dynamic raster patch
        date_str = event_date.strftime('%Y%m%d')
        dynamic_patch = self._load_dynamic_patch(date_str, window)
        
        # Convert to tensors
        zone_patch = torch.from_numpy(zone_patch).long()
        dynamic_patch = torch.from_numpy(dynamic_patch).float()
        
        return {
            'cat': torch.tensor(cat, dtype=torch.long),
            'node_idx': torch.tensor(node_idx, dtype=torch.long),
            'event_date_str': date_str,
            'zone_patch': zone_patch,  # (H, W)
            'dynamic_patch': dynamic_patch,  # (C, H, W)
            'label': torch.tensor(label, dtype=torch.float32)
        }


class MILCollator:
    """
    Custom collator for patch-based MIL batching
    
    Combines samples into batches with zone patches for GNN mapping.
    
    Args:
        graph_data: PyTorch Geometric Data object (for static features)
    """
    
    def __init__(self, graph_data: torch.Tensor):
        self.graph_data = graph_data
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples (patch-based)
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            batch_dict: Dictionary containing:
                - 'static_x': (num_nodes, static_dim) - all node features
                - 'edge_index': (2, num_edges) - graph connectivity
                - 'edge_attr': (num_edges, 1) - edge weights
                - 'node_indices': (B,) - node indices for this batch
                - 'cats': (B,) - slope IDs
                - 'zone_patches': (B, H, W) - slope ID raster patches
                - 'dynamic_patches': (B, C, H, W) - dynamic feature patches
                - 'labels': (B,) - binary labels
                - 'event_date_strs': List of date strings
        """
        # Extract and stack components
        node_indices = torch.stack([s['node_idx'] for s in batch])
        cats = torch.stack([s['cat'] for s in batch])
        zone_patches = torch.stack([s['zone_patch'] for s in batch])
        dynamic_patches = torch.stack([s['dynamic_patch'] for s in batch])
        labels = torch.stack([s['label'] for s in batch])
        event_date_strs = [s['event_date_str'] for s in batch]
        
        return {
            'static_x': self.graph_data.x,
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr,
            'node_indices': node_indices,
            'cats': cats,
            'zone_patches': zone_patches,  # (B, H, W)
            'dynamic_patches': dynamic_patches,  # (B, C, H, W)
            'labels': labels,
            'event_date_strs': event_date_strs
        }


def create_mil_dataloaders(
    graph_path: str,
    samples_path: str,
    slope_polygons_path: str,
    raster_base_path: str,
    reference_raster_path: str,
    slope_id_raster_path: str,
    slope_bboxes_cache_path: str,
    patch_size: int = 128,
    batch_size: int = 4,
    dynamic_variables: List[str] = ['acc3d', 'acc7d', 'peak1h'],
    dynamic_statistics: List[str] = [],
    start_date: str = '20200301',
    end_date: str = '20200930',
    train_ratio: float = 0.8,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, LandslideRasterDataset]:
    """
    Create train and validation data loaders for patch-based MIL
    
    Args:
        graph_path: Path to graph data
        samples_path: Path to samples CSV
        slope_polygons_path: Path to slope polygons (unused in patch mode)
        raster_base_path: Base path to raster files
        reference_raster_path: Path to reference raster
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to cached slope bboxes
        patch_size: Patch size (default: 128)
        batch_size: Batch size
        dynamic_variables: List of dynamic variables
        dynamic_statistics: List of statistics (empty for pixel-level)
        start_date: Start date
        end_date: End date
        train_ratio: Training data ratio
        random_seed: Random seed
        num_workers: Number of workers
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: Full dataset object
    """
    # Create dataset
    dataset = LandslideRasterDataset(
        graph_path=graph_path,
        samples_path=samples_path,
        slope_polygons_path=slope_polygons_path,
        raster_base_path=raster_base_path,
        reference_raster_path=reference_raster_path,
        slope_id_raster_path=slope_id_raster_path,
        slope_bboxes_cache_path=slope_bboxes_cache_path,
        patch_size=patch_size,
        dynamic_variables=dynamic_variables,
        dynamic_statistics=dynamic_statistics,
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
    collator = MILCollator(dataset.graph_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, dataset


# ============================================================
# Stage 2: Dataset with High-Resolution Static Rasters
# ============================================================

class LandslideRasterDatasetStage2(Dataset):
    """
    Stage 2 dataset for U-Net training with high-resolution static rasters
    
    Input channels (12 total):
    - Channel 0: GNN susceptibility (pre-generated)
    - Channels 1-8: High-resolution static rasters (slope, aspect, curvature, etc.)
    - Channels 9-11: Dynamic rainfall rasters (acc3d, acc7d, peak1h)
    
    Args:
        graph_path: Path to graph data (for compatibility)
        samples_path: Path to samples CSV
        slope_polygons_path: Path to slope polygons (unused)
        raster_base_path: Base path to dynamic rasters
        reference_raster_path: Path to reference raster
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to cached bboxes
        gnn_raster_path: Path to pre-generated GNN raster
        static_raster_dir: Directory containing static rasters
        static_raster_files: List of static raster filenames
        patch_size: Patch size
        dynamic_variables: List of dynamic variables
        start_date: Start date
        end_date: End date
    """
    
    def __init__(
        self,
        graph_path: str,
        samples_path: str,
        slope_polygons_path: str,
        raster_base_path: str,
        reference_raster_path: str,
        slope_id_raster_path: str,
        slope_bboxes_cache_path: str,
        gnn_raster_path: str,
        static_raster_dir: str,
        static_raster_files: List[str],
        patch_size: int = 128,
        dynamic_variables: List[str] = ['acc3d', 'acc7d', 'peak1h'],
        start_date: str = '20190101',
        end_date: str = '20200930'
    ):
        super().__init__()
        
        self.graph_path = graph_path
        self.samples_path = samples_path
        self.raster_base_path = raster_base_path
        self.slope_id_raster_path = slope_id_raster_path
        self.gnn_raster_path = gnn_raster_path
        self.static_raster_dir = static_raster_dir
        self.static_raster_files = static_raster_files
        self.patch_size = patch_size
        self.dynamic_variables = dynamic_variables
        
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
        self.static_raster_paths = [
            os.path.join(static_raster_dir, fname) for fname in static_raster_files
        ]
        print(f"  Static rasters: {len(self.static_raster_paths)} files")
        for path in self.static_raster_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Static raster not found: {path}")
        
        # Compute or load slope bounding boxes
        print("Computing slope bounding boxes...")
        self.slope_bboxes = compute_slope_bboxes(
            slope_id_raster_path=slope_id_raster_path,
            cache_path=slope_bboxes_cache_path,
            patch_size=patch_size
        )
        
        # Load graph data (for cat_to_node mapping)
        print(f"Loading graph from {graph_path}...")
        self.graph_data = torch.load(graph_path, weights_only=False)
        self.cat_values = self.graph_data.cat
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.cat_values.numpy())}
        print(f"  Graph: {self.graph_data.num_nodes} nodes")
        
        # Load samples
        print(f"Loading samples from {samples_path}...")
        samples_df = pd.read_csv(samples_path, encoding='utf-8-sig')
        samples_df['event_date'] = pd.to_datetime(samples_df['event_date'], format='%Y-%m-%d')
        
        # Filter by date range and valid cat
        samples_df = samples_df[
            (samples_df['event_date'] >= self.start_date) &
            (samples_df['event_date'] <= self.end_date) &
            (samples_df['cat'].isin(self.cat_to_node.keys()))
        ]
        
        # Convert to list
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
        
        # Total channels: 1 (GNN) + N (static) + M (dynamic)
        self.num_total_channels = 1 + len(static_raster_files) + len(dynamic_variables)
        print(f"  Total channels: {self.num_total_channels} "
              f"(1 GNN + {len(static_raster_files)} static + {len(dynamic_variables)} dynamic)")
    
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
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample
        
        Returns:
            sample: Dictionary containing:
                - 'cat': Slope ID
                - 'zone_patch': (H, W) slope ID raster patch
                - 'combined_patch': (C, H, W) combined input (GNN + static + dynamic)
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
        
        # Handle NoData in GNN patch (-9999 -> 0)
        gnn_patch = np.where(gnn_patch < 0, 0.0, gnn_patch)
        
        # Pad zone_patch and gnn_patch if needed
        if zone_patch.shape[0] < self.patch_size or zone_patch.shape[1] < self.patch_size:
            zone_patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=zone_patch.dtype)
            gnn_patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=gnn_patch.dtype)
            
            h, w = zone_patch.shape
            zone_patch_padded[:h, :w] = zone_patch
            gnn_patch_padded[:h, :w] = gnn_patch
            
            zone_patch = zone_patch_padded
            gnn_patch = gnn_patch_padded
        
        # 3. Extract static patches
        static_patches = []
        for raster_path in self.static_raster_paths:
            with rasterio.open(raster_path) as src:
                patch = src.read(1, window=window)
            # Handle NaN and extreme values
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pad if needed
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                h, w = patch.shape
                patch_padded[:h, :w] = patch
                patch = patch_padded
            
            static_patches.append(patch)
        static_patch = np.stack(static_patches, axis=0)  # (N_static, H, W)
        
        # 4. Extract dynamic patches
        dynamic_patches = []
        date_str = event_date.strftime('%Y%m%d')
        
        for var in self.dynamic_variables:
            raster_path = os.path.join(
                self.raster_base_path,
                var,
                f"{date_str}_{var}_mm_5179_30m.tif"
            )
            
            if not os.path.exists(raster_path):
                # Use zeros if file not found
                patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            else:
                with rasterio.open(raster_path) as src:
                    patch = src.read(1, window=window)
                # Handle NaN and extreme values
                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Pad if needed
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    patch_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                    h, w = patch.shape
                    patch_padded[:h, :w] = patch
                    patch = patch_padded
            
            dynamic_patches.append(patch)
        
        dynamic_patch = np.stack(dynamic_patches, axis=0)  # (N_dynamic, H, W)
        
        # 5. Normalize static patches (z-score normalization per channel)
        # GNN patch is already in [0, 1], dynamic patches are in reasonable range
        # But static patches (especially TWI, flow accumulation) can have extreme values
        for i in range(static_patch.shape[0]):
            channel = static_patch[i]
            valid_mask = ~np.isnan(channel) & (channel != 0)
            if valid_mask.sum() > 0:
                mean = channel[valid_mask].mean()
                std = channel[valid_mask].std()
                if std > 1e-6:  # Avoid division by zero
                    static_patch[i] = (channel - mean) / (std + 1e-6)
                else:
                    static_patch[i] = channel - mean
        
        # 6. Combine all channels
        combined_patch = np.concatenate([
            gnn_patch[np.newaxis, ...],  # (1, H, W)
            static_patch,                 # (N_static, H, W)
            dynamic_patch                 # (N_dynamic, H, W)
        ], axis=0)  # (C_total, H, W)
        
        # Clip extreme values to prevent overflow
        combined_patch = np.clip(combined_patch, -10, 10)
        
        # Convert to tensors
        zone_patch = torch.from_numpy(zone_patch).long()
        combined_patch = torch.from_numpy(combined_patch).float()
        
        return {
            'cat': torch.tensor(cat, dtype=torch.long),
            'zone_patch': zone_patch,
            'combined_patch': combined_patch,
            'label': torch.tensor(label, dtype=torch.float32)
        }


class MILCollatorStage2:
    """
    Collator for Stage 2 MIL batches
    
    Note: GNN forward pass is not needed since GNN output is pre-computed
    and included in combined_patch.
    """
    
    def __init__(self, graph_data):
        """
        Args:
            graph_data: Graph data (kept for compatibility, but not used in forward)
        """
        self.graph_data = graph_data
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch of samples"""
        cats = torch.stack([s['cat'] for s in batch])
        zone_patches = torch.stack([s['zone_patch'] for s in batch])
        combined_patches = torch.stack([s['combined_patch'] for s in batch])
        labels = torch.stack([s['label'] for s in batch])
        
        return {
            'cats': cats,
            'zone_patches': zone_patches,
            'combined_patches': combined_patches,
            'labels': labels,
            # Keep graph data for compatibility (not used in training)
            'static_x': self.graph_data.x,
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr
        }


def create_mil_dataloaders_stage2(
    graph_path: str,
    samples_path: str,
    slope_polygons_path: str,
    raster_base_path: str,
    reference_raster_path: str,
    slope_id_raster_path: str,
    slope_bboxes_cache_path: str,
    gnn_raster_path: str,
    static_raster_dir: str,
    static_raster_files: List[str],
    patch_size: int = 128,
    batch_size: int = 16,
    dynamic_variables: List[str] = ['acc3d', 'acc7d', 'peak1h'],
    start_date: str = '20190101',
    end_date: str = '20200930',
    train_ratio: float = 0.8,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, LandslideRasterDatasetStage2]:
    """
    Create train and validation data loaders for Stage 2
    
    Args:
        graph_path: Path to graph data
        samples_path: Path to samples CSV
        slope_polygons_path: Path to slope polygons (unused)
        raster_base_path: Base path to dynamic rasters
        reference_raster_path: Path to reference raster (unused)
        slope_id_raster_path: Path to slope ID raster
        slope_bboxes_cache_path: Path to slope bboxes cache
        gnn_raster_path: Path to pre-generated GNN raster
        static_raster_dir: Directory containing static rasters
        static_raster_files: List of static raster filenames
        patch_size: Patch size
        batch_size: Batch size
        dynamic_variables: List of dynamic variables
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
    dataset = LandslideRasterDatasetStage2(
        graph_path=graph_path,
        samples_path=samples_path,
        slope_polygons_path=slope_polygons_path,
        raster_base_path=raster_base_path,
        reference_raster_path=reference_raster_path,
        slope_id_raster_path=slope_id_raster_path,
        slope_bboxes_cache_path=slope_bboxes_cache_path,
        gnn_raster_path=gnn_raster_path,
        static_raster_dir=static_raster_dir,
        static_raster_files=static_raster_files,
        patch_size=patch_size,
        dynamic_variables=dynamic_variables,
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
    collator = MILCollatorStage2(dataset.graph_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, dataset


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing MIL Data Loader")
    print("="*70)
    
    # Note: Requires actual data files
    print("\nThis module requires actual data files to test.")
    print("Use in training pipeline with real data paths.")
    
    print("\n" + "="*70)
    print("MIL data loader module loaded successfully!")
    print("="*70)

