"""
Raster Conversion Utilities for GNN-U-Net Model

This module provides utilities to convert GNN outputs (slope-level predictions)
to raster format for U-Net input, and to load dynamic raster stacks.

Key Functions:
- load_slope_geometries(): Load slope polygons from GeoPackage
- create_slope_to_pixel_mapping(): Map slope IDs to pixel indices
- gnn_output_to_raster(): Convert GNN node predictions to raster
- load_dynamic_raster_stack(): Load multi-channel raster for specific date

Author: Landslide Risk Analysis Project
Date: 2025-01-16
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import torch
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import Window
import geopandas as gpd
from affine import Affine


class RasterConverter:
    """
    Handles conversion between GNN outputs and raster format
    
    Args:
        slope_polygons_path: Path to slope polygons GeoPackage
        reference_raster_path: Path to reference raster (for CRS, bounds, resolution)
        cache_mapping: Whether to cache slope-to-pixel mapping in memory
    """
    
    def __init__(
        self,
        slope_polygons_path: str,
        reference_raster_path: Optional[str] = None,
        cache_mapping: bool = True
    ):
        self.slope_polygons_path = slope_polygons_path
        self.reference_raster_path = reference_raster_path
        self.cache_mapping = cache_mapping
        
        # Cache
        self._slope_gdf = None
        self._slope_to_pixels = None
        self._raster_meta = None
        self._transform = None
        self._shape = None
        
        # Load reference raster metadata if provided
        if reference_raster_path is not None:
            self._load_reference_metadata()
    
    def _load_reference_metadata(self):
        """Load metadata from reference raster"""
        with rasterio.open(self.reference_raster_path) as src:
            self._raster_meta = src.meta.copy()
            self._transform = src.transform
            self._shape = (src.height, src.width)
            
            print(f"Reference raster loaded:")
            print(f"  CRS: {src.crs}")
            print(f"  Shape: {self._shape}")
            print(f"  Resolution: {src.res}")
            print(f"  Bounds: {src.bounds}")
    
    def load_slope_geometries(self, layer_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load slope polygons from GeoPackage
        
        Args:
            layer_name: Layer name in GeoPackage (optional)
        
        Returns:
            gdf: GeoDataFrame with slope geometries and attributes
        """
        if self._slope_gdf is not None:
            return self._slope_gdf
        
        print(f"Loading slope polygons from {self.slope_polygons_path}...")
        gdf = gpd.read_file(self.slope_polygons_path, layer=layer_name)
        
        # Ensure 'cat' column exists
        if 'cat' not in gdf.columns:
            raise ValueError("Slope polygons must have 'cat' column for identification")
        
        print(f"  Loaded {len(gdf):,} slope polygons")
        print(f"  CRS: {gdf.crs}")
        print(f"  Cat range: [{gdf['cat'].min()}, {gdf['cat'].max()}]")
        
        if self.cache_mapping:
            self._slope_gdf = gdf
        
        return gdf
    
    def create_slope_to_pixel_mapping(
        self,
        shape: Optional[Tuple[int, int]] = None,
        transform: Optional[Affine] = None,
        force_rebuild: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Create mapping from slope ID (cat) to pixel indices
        
        Args:
            shape: Raster shape (height, width)
            transform: Affine transform
            force_rebuild: Force rebuild even if cached
        
        Returns:
            slope_to_pixels: Dict mapping cat -> pixel indices (N, 2) array of (row, col)
        """
        # Return cached if available
        if self._slope_to_pixels is not None and not force_rebuild:
            return self._slope_to_pixels
        
        # Use reference metadata if not provided
        if shape is None:
            if self._shape is None:
                raise ValueError("Must provide shape or set reference_raster_path")
            shape = self._shape
        
        if transform is None:
            if self._transform is None:
                raise ValueError("Must provide transform or set reference_raster_path")
            transform = self._transform
        
        print("Creating slope-to-pixel mapping...")
        gdf = self.load_slope_geometries()
        
        # Rasterize slope IDs
        slope_raster = rasterize(
            [(geom, cat) for geom, cat in zip(gdf.geometry, gdf['cat'])],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.int32
        )
        
        # Create mapping: cat -> pixel indices (fully vectorized)
        # Extract all non-zero pixels at once
        rows, cols = np.where(slope_raster > 0)
        cat_values = slope_raster[rows, cols]
        
        # Use pandas for ultra-fast grouping (optimized C code)
        df = pd.DataFrame({
            'cat': cat_values,
            'row': rows,
            'col': cols
        })
        
        # Group by cat and create dictionary
        slope_to_pixels = {}
        for cat, group in df.groupby('cat'):
            slope_to_pixels[int(cat)] = group[['row', 'col']].values
        
        print(f"  Mapped {len(slope_to_pixels):,} slopes to pixels")
        print(f"  Avg pixels per slope: {np.mean([len(v) for v in slope_to_pixels.values()]):.1f}")
        
        if self.cache_mapping:
            self._slope_to_pixels = slope_to_pixels
        
        return slope_to_pixels
    
    def gnn_output_to_raster(
        self,
        gnn_outputs: Union[torch.Tensor, np.ndarray],
        cat_values: Union[torch.Tensor, np.ndarray],
        shape: Optional[Tuple[int, int]] = None,
        transform: Optional[Affine] = None,
        fill_value: float = 0.0,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert GNN node-level outputs to raster format
        
        Args:
            gnn_outputs: (num_nodes,) array of GNN predictions (0-1 probabilities)
            cat_values: (num_nodes,) array of slope IDs (cat)
            shape: Output raster shape (height, width)
            transform: Affine transform
            fill_value: Fill value for pixels not in any slope
            output_path: Optional path to save raster as GeoTIFF
        
        Returns:
            raster: (H, W) array of values
        """
        # Convert to numpy if needed
        if isinstance(gnn_outputs, torch.Tensor):
            gnn_outputs = gnn_outputs.detach().cpu().numpy()
        if isinstance(cat_values, torch.Tensor):
            cat_values = cat_values.detach().cpu().numpy()
        
        # Use reference metadata if not provided
        if shape is None:
            if self._shape is None:
                raise ValueError("Must provide shape or set reference_raster_path")
            shape = self._shape
        
        if transform is None:
            if self._transform is None:
                raise ValueError("Must provide transform or set reference_raster_path")
            transform = self._transform
        
        # Get or create slope-to-pixel mapping
        slope_to_pixels = self.create_slope_to_pixel_mapping(shape, transform)
        
        # Create output raster
        raster = np.full(shape, fill_value, dtype=np.float32)
        
        # Fill raster with GNN outputs
        cat_to_output = {int(cat): output for cat, output in zip(cat_values, gnn_outputs)}
        
        for cat, pixels in slope_to_pixels.items():
            if cat in cat_to_output:
                rows, cols = pixels[:, 0], pixels[:, 1]
                raster[rows, cols] = cat_to_output[cat]
        
        # Save if output_path provided
        if output_path is not None:
            self.save_raster(raster, output_path, transform)
        
        return raster
    
    def save_raster(
        self,
        data: np.ndarray,
        output_path: str,
        transform: Optional[Affine] = None,
        crs: Optional[str] = None,
        dtype: str = 'float32',
        nodata: Optional[float] = None
    ):
        """
        Save numpy array as GeoTIFF
        
        Args:
            data: 2D array to save
            output_path: Output path
            transform: Affine transform
            crs: Coordinate reference system
            dtype: Output data type
            nodata: NoData value
        """
        if transform is None:
            if self._transform is None:
                raise ValueError("Must provide transform or set reference_raster_path")
            transform = self._transform
        
        if crs is None:
            if self._raster_meta is None:
                raise ValueError("Must provide crs or set reference_raster_path")
            crs = self._raster_meta['crs']
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write raster
        meta = {
            'driver': 'GTiff',
            'dtype': dtype,
            'nodata': nodata,
            'width': data.shape[1],
            'height': data.shape[0],
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data, 1)
        
        print(f"Raster saved: {output_path}")


class DynamicRasterLoader:
    """
    Loads dynamic raster stacks for specific dates
    
    Args:
        raster_base_path: Base path to raster directory
        variables: List of variable names to load (e.g., ['acc3d', 'acc7d', 'peak1h'])
        statistics: List of statistics to load (e.g., ['mean', 'max'])
    """
    
    def __init__(
        self,
        raster_base_path: str,
        variables: List[str] = ['acc3d', 'acc7d', 'peak1h'],
        statistics: List[str] = []  # Empty by default for pixel-level prediction
    ):
        self.raster_base_path = Path(raster_base_path)
        self.variables = variables
        self.statistics = statistics if statistics else []  # Ensure it's a list
        
        # Validate paths
        if not self.raster_base_path.exists():
            raise FileNotFoundError(f"Raster base path not found: {self.raster_base_path}")
        
        print(f"Dynamic raster loader initialized:")
        print(f"  Base path: {self.raster_base_path}")
        print(f"  Variables: {self.variables}")
        if self.statistics:
            print(f"  Statistics: {self.statistics}")
        else:
            print(f"  Mode: Raw raster values (pixel-level)")
    
    def _get_raster_path(self, variable: str, date: str) -> Path:
        """
        Get raster file path for specific variable and date
        
        Args:
            variable: Variable name (e.g., 'acc3d')
            date: Date string in YYYYMMDD format
        
        Returns:
            path: Path to raster file
        
        Note:
            LDAPS rasters are raw pixel values, not slope-level statistics.
            Statistics (mean/max) are computed per slope and stored in CSV files.
            This is used for GNN-U-Net (pixel-level prediction), not GNN-RNN.
        """
        # Pattern for LDAPS data: base_path / variable / f"{date}_{variable}_mm_5179_30m.tif"
        filename = f"{date}_{variable}_mm_5179_30m.tif"
        path = self.raster_base_path / variable / filename
        
        return path
    
    def load_single_raster(self, variable: str, date: str) -> np.ndarray:
        """
        Load a single raster (raw pixel values)
        
        Args:
            variable: Variable name (e.g., 'acc3d', 'acc7d', 'peak1h')
            date: Date string in YYYYMMDD format
        
        Returns:
            data: 2D array of raw raster values
        
        Note:
            For GNN-U-Net pixel-level prediction.
            GNN-RNN uses slope-level statistics from CSV files instead.
        """
        path = self._get_raster_path(variable, date)
        
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        
        with rasterio.open(path) as src:
            # Read first band (raw pixel values)
            data = src.read(1)
        
        return data
    
    def load_dynamic_raster_stack(
        self,
        date: Union[str, datetime],
        output_channels_first: bool = True
    ) -> np.ndarray:
        """
        Load all dynamic features for a specific date as a multi-channel array
        
        Args:
            date: Date string (YYYYMMDD) or datetime object
            output_channels_first: If True, return (C, H, W), else (H, W, C)
        
        Returns:
            stack: Multi-channel array of shape (C, H, W) or (H, W, C)
                   where C = len(variables)
        
        Note:
            Loads raw raster values (pixel-level) for U-Net input.
            Does NOT use slope-level statistics (mean/max).
        """
        # Convert datetime to string if needed
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = date
        
        # Load all rasters (one per variable)
        channels = []
        
        for variable in self.variables:
            try:
                data = self.load_single_raster(variable, date_str)
                channels.append(data)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # Use zeros if file not found
                if len(channels) > 0:
                    data = np.zeros_like(channels[0])
                else:
                    raise ValueError("Cannot create zero array without reference shape")
                channels.append(data)
        
        # Stack channels
        if output_channels_first:
            stack = np.stack(channels, axis=0)  # (C, H, W)
        else:
            stack = np.stack(channels, axis=-1)  # (H, W, C)
        
        return stack
    
    def get_channel_names(self) -> List[str]:
        """Get list of channel names in order"""
        # For pixel-level prediction: just variable names
        return self.variables


# ============================================================
# Helper Functions
# ============================================================

def create_study_area_metadata(
    bounds: Tuple[float, float, float, float],
    resolution: float = 30.0,
    crs: str = 'EPSG:5179'
) -> Dict:
    """
    Create raster metadata for study area
    
    Args:
        bounds: (xmin, ymin, xmax, ymax) in CRS units
        resolution: Pixel resolution in CRS units (default: 30m)
        crs: Coordinate reference system
    
    Returns:
        metadata: Dict with shape, transform, crs
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Calculate shape
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    
    # Create transform
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    metadata = {
        'shape': (height, width),
        'transform': transform,
        'crs': crs,
        'resolution': resolution,
        'bounds': bounds
    }
    
    return metadata


def load_reference_raster_metadata(raster_path: str) -> Dict:
    """
    Load metadata from reference raster
    
    Args:
        raster_path: Path to reference raster
    
    Returns:
        metadata: Dict with shape, transform, crs, bounds
    """
    with rasterio.open(raster_path) as src:
        metadata = {
            'shape': (src.height, src.width),
            'transform': src.transform,
            'crs': str(src.crs),
            'resolution': src.res,
            'bounds': src.bounds
        }
    
    return metadata


def compute_slope_bboxes(
    slope_id_raster_path: str,
    cache_path: Optional[str] = None,
    patch_size: int = 128
) -> Dict[int, Dict]:
    """
    Compute bounding boxes for all slopes in the slope ID raster
    
    Args:
        slope_id_raster_path: Path to slope ID raster
        cache_path: Optional path to save/load cached bboxes
        patch_size: Patch size for logging oversized slopes
    
    Returns:
        slope_bboxes: Dict mapping cat -> bbox info
            {cat: {
                'row_min': int,
                'row_max': int,
                'col_min': int,
                'col_max': int,
                'centroid_row': int,
                'centroid_col': int,
                'height': int,
                'width': int
            }}
    """
    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached slope bboxes from {cache_path}...")
        with open(cache_path, 'rb') as f:
            slope_bboxes = pickle.load(f)
        print(f"  Loaded {len(slope_bboxes):,} slope bounding boxes")
        return slope_bboxes
    
    print(f"Computing slope bounding boxes from {slope_id_raster_path}...")
    
    # Load slope ID raster
    with rasterio.open(slope_id_raster_path) as src:
        slope_id_raster = src.read(1)
    
    # Get unique slope IDs
    unique_cats = np.unique(slope_id_raster[slope_id_raster > 0])
    print(f"  Found {len(unique_cats):,} unique slopes")
    
    # Compute bboxes using vectorized operations
    slope_bboxes = {}
    oversized_slopes = []
    
    for cat in unique_cats:
        # Find all pixels belonging to this slope
        rows, cols = np.where(slope_id_raster == cat)
        
        if len(rows) == 0:
            continue
        
        # Compute bbox
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        height = row_max - row_min + 1
        width = col_max - col_min + 1
        
        slope_bboxes[int(cat)] = {
            'row_min': int(row_min),
            'row_max': int(row_max),
            'col_min': int(col_min),
            'col_max': int(col_max),
            'centroid_row': int((row_min + row_max) // 2),
            'centroid_col': int((col_min + col_max) // 2),
            'height': int(height),
            'width': int(width)
        }
        
        # Check if oversized
        if height > patch_size or width > patch_size:
            oversized_slopes.append({
                'cat': int(cat),
                'height': int(height),
                'width': int(width),
                'pixels': int(len(rows))
            })
    
    print(f"  Computed {len(slope_bboxes):,} bounding boxes")
    
    # Log oversized slopes
    if oversized_slopes:
        log_dir = os.path.dirname(cache_path) if cache_path else "."
        log_path = os.path.join(log_dir, "oversized_slopes.log")
        
        with open(log_path, 'w') as f:
            f.write(f"Oversized Slopes Report (patch_size={patch_size})\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Total oversized slopes: {len(oversized_slopes)}\n")
            f.write(f"Max height: {max(s['height'] for s in oversized_slopes)}\n")
            f.write(f"Max width: {max(s['width'] for s in oversized_slopes)}\n")
            f.write(f"Max pixels: {max(s['pixels'] for s in oversized_slopes)}\n\n")
            
            f.write(f"{'Cat ID':<10} {'Height':<10} {'Width':<10} {'Pixels':<10}\n")
            f.write(f"{'-'*40}\n")
            
            for s in sorted(oversized_slopes, key=lambda x: x['pixels'], reverse=True):
                f.write(f"{s['cat']:<10} {s['height']:<10} {s['width']:<10} {s['pixels']:<10}\n")
        
        print(f"  ⚠️  Warning: {len(oversized_slopes)} slopes exceed {patch_size}×{patch_size}")
        print(f"      See: {log_path}")
    
    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(slope_bboxes, f)
        print(f"  Cached bboxes to {cache_path}")
    
    return slope_bboxes


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing Raster Utilities")
    print("="*70)
    
    # Test with dummy data
    print("\n[1] Testing RasterConverter with dummy data")
    
    # Create dummy converter (without real files)
    print("  Note: Requires actual GeoPackage and raster files for full test")
    print("  Skipping full test - use in actual pipeline")
    
    print("\n" + "="*70)
    print("Raster utilities module loaded successfully!")
    print("="*70)

