"""
Data loader for Hierarchical Fusion Model

Components:
- GeometricAugmentation: HorizontalFlip, VerticalFlip, RandomRotate90
- HierarchicalDataset: Loads static, dynamic, GNN embeddings, KFS prior
"""

import os
import numpy as np
import pandas as pd
import torch
import rasterio
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.normalization_utils import (
    load_or_compute_stats,
    normalize_static_variable,
    normalize_dynamic_variable
)


class GeometricAugmentation:
    """
    Geometric augmentation for spatial data

    3 augmentation techniques (NO interpolation):
    1. HorizontalFlip: Left-right flip
    2. VerticalFlip: Top-bottom flip
    3. RandomRotate90: 0/90/180/270 degree rotation

    Applied to both positive and negative samples
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability for each augmentation technique
        """
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to sample

        Args:
            sample: {
                'static': (8, H, W),
                'dynamic': (T=5, 3, H, W),
                'gnn_embedding': (128, H, W),
                'kfs_prior': (1, H, W),
                'slope_mask': (1, H, W),
                ...
            }

        Returns:
            augmented_sample
        """
        # Keys to apply spatial augmentation
        spatial_keys = ['static', 'dynamic', 'gnn_embedding', 'kfs_prior', 'slope_mask']

        # 1. HorizontalFlip (left-right)
        if np.random.rand() < self.p:
            for key in spatial_keys:
                if key not in sample:
                    continue
                if key == 'dynamic':
                    # (T, C, H, W) → flip along W
                    sample[key] = torch.flip(sample[key], dims=[-1])
                else:
                    # (C, H, W) → flip along W
                    sample[key] = torch.flip(sample[key], dims=[-1])

            # Aspect correction: eastness (sin) sign flip
            if 'static' in sample:
                sample['static'][0] = -sample['static'][0]  # eastness = -eastness

        # 2. VerticalFlip (top-bottom)
        if np.random.rand() < self.p:
            for key in spatial_keys:
                if key not in sample:
                    continue
                if key == 'dynamic':
                    # (T, C, H, W) → flip along H
                    sample[key] = torch.flip(sample[key], dims=[-2])
                else:
                    # (C, H, W) → flip along H
                    sample[key] = torch.flip(sample[key], dims=[-2])

            # Aspect correction: northness (cos) sign flip
            if 'static' in sample:
                sample['static'][1] = -sample['static'][1]  # northness = -northness

        # 3. RandomRotate90 (0/90/180/270 degrees)
        k = np.random.choice([0, 1, 2, 3])  # 0/90/180/270 degrees
        if k > 0:
            for key in spatial_keys:
                if key not in sample:
                    continue
                if key == 'dynamic':
                    # (T, C, H, W) → rotate along (H, W)
                    sample[key] = torch.rot90(sample[key], k=k, dims=[-2, -1])
                else:
                    # (C, H, W) → rotate along (H, W)
                    sample[key] = torch.rot90(sample[key], k=k, dims=[-2, -1])

            # Aspect correction: rotate eastness/northness
            if 'static' in sample and k > 0:
                eastness = sample['static'][0].clone()
                northness = sample['static'][1].clone()

                if k == 1:  # 90 degrees clockwise
                    sample['static'][0] = northness   # new eastness
                    sample['static'][1] = -eastness   # new northness
                elif k == 2:  # 180 degrees
                    sample['static'][0] = -eastness
                    sample['static'][1] = -northness
                elif k == 3:  # 270 degrees (= 90 counter-clockwise)
                    sample['static'][0] = -northness
                    sample['static'][1] = eastness

        return sample


class HierarchicalDataset(Dataset):
    """
    Dataset for Hierarchical Fusion Model

    Loads:
    - Static terrain rasters (7 channels + mask)
    - Dynamic temporal sequences (5-day window, 3 channels)
    - GNN embeddings (128d per pixel, rasterized)
    - KFS prior (probability map)
    - Slope masks (interior pixels)
    """

    def __init__(self, config: Dict, samples_df: pd.DataFrame,
                 gnn_embedding_path: Optional[str] = None, augment: bool = False,
                 aug_prob: float = 0.5):
        """
        Args:
            config: model config
            samples_df: DataFrame with columns [cat, event_date, label]
            gnn_embedding_path: Path to gnn_embeddings_128d.pt (Optional)
            augment: Whether to apply augmentation
            aug_prob: Probability for each augmentation technique
        """
        self.samples = samples_df.reset_index(drop=True)
        self.config = config
        self.augment = augment

        # Paths
        self.static_raster_dir = config['data']['static_raster_dir']
        self.dynamic_raster_dir = config['data']['dynamic_raster_dir']
        self.kfs_prior_path = config['data']['kfs_prior_path']
        self.slope_id_raster_path = config['data']['slope_id_raster_path']
        self.valid_area_mask_path = config['data']['valid_area_mask_path']

        # Additional directories for new features
        self.litho_raster_dir = config['data'].get('litho_raster_dir', 'data/model_ready/Litho')
        self.landcover_raster_dir = config['data'].get('landcover_raster_dir', 'data/model_ready/landcover')
        self.forestroad_raster_dir = config['data'].get('forestroad_raster_dir', 'data/model_ready/forestroad')

        # Feature names
        self.static_features = config['data']['static_features']
        self.dynamic_variables = config['data']['dynamic_variables']

        # Patch configuration
        self.patch_size = config['data']['patch_size']
        self.temporal_window = config['data']['temporal_window']

        # GNN embedding (조건부 로딩)
        self.use_gnn_embedding = config['data'].get('use_gnn_embedding', True)
        if self.use_gnn_embedding:
            if gnn_embedding_path is None:
                raise ValueError("gnn_embedding_path required when use_gnn_embedding=True")
            self.gnn_embeddings = self._load_gnn_embeddings(gnn_embedding_path)
        else:
            self.gnn_embeddings = None
            print("  GNN embedding disabled (use_gnn_embedding=False)")

        # Load slope ID raster
        with rasterio.open(self.slope_id_raster_path) as src:
            self.slope_id_raster = src.read(1)
            self.transform = src.transform
            self.crs = src.crs

        # Load KFS prior
        with rasterio.open(self.kfs_prior_path) as src:
            kfs_prior_raw = src.read(1, masked=True).astype(np.float32)
            # Fill NoData with 0 and clip to [0, 1] range
            self.kfs_prior = kfs_prior_raw.filled(0.0)
            self.kfs_prior = np.clip(self.kfs_prior, 0.0, 1.0)

        # Load valid area mask
        with rasterio.open(self.valid_area_mask_path) as src:
            self.valid_area_mask = src.read(1).astype(np.float32)

        # Load seasonal NDVI (pre-load all 4 seasons into memory)
        self.seasonal_ndvi = self._load_seasonal_ndvi()

        # Normalization stats
        norm_stats_path = config['data']['norm_stats_path']
        self.norm_stats = load_or_compute_stats()

        # Monthly normalization option
        self.use_monthly_norm = config['data'].get('use_monthly_normalization', False)
        if self.use_monthly_norm:
            print("  Monthly normalization ENABLED (seasonal rainfall patterns)")
        else:
            print("  Monthly normalization DISABLED (global statistics)")

        # Filter samples by valid pixel count (>= min_valid_pixels)
        min_valid_pixels = config['data'].get('min_valid_pixels', 5)
        self.samples = self._filter_samples_by_valid_pixels(min_valid_pixels)

        # Augmentation
        if self.augment:
            self.transform_fn = GeometricAugmentation(p=aug_prob)
        else:
            self.transform_fn = None

    def _filter_samples_by_valid_pixels(self, min_valid_pixels: int) -> pd.DataFrame:
        """
        Filter samples by valid pixel count within slope interior

        Args:
            min_valid_pixels: Minimum number of valid pixels required

        Returns:
            filtered_samples: DataFrame with valid samples only
        """
        print(f"\nFiltering samples with < {min_valid_pixels} valid pixels...")
        original_count = len(self.samples)

        valid_indices = []

        for idx, row in self.samples.iterrows():
            slope_id = int(row['cat'])

            # Count valid pixels in slope interior
            slope_mask = (self.slope_id_raster == slope_id)
            valid_pixels = (slope_mask & (self.valid_area_mask > 0)).sum()

            if valid_pixels >= min_valid_pixels:
                valid_indices.append(idx)

        filtered_samples = self.samples.loc[valid_indices].reset_index(drop=True)

        removed_count = original_count - len(filtered_samples)
        print(f"  Removed {removed_count} samples ({removed_count/original_count*100:.1f}%)")
        print(f"  Remaining: {len(filtered_samples)} samples")

        # Show label distribution
        if 'label' in filtered_samples.columns:
            pos_count = (filtered_samples['label'] == 1).sum()
            neg_count = (filtered_samples['label'] == 0).sum()
            print(f"  Positive: {pos_count}, Negative: {neg_count}")

        return filtered_samples

    def _load_seasonal_ndvi(self) -> Dict[str, np.ndarray]:
        """
        Load seasonal NDVI mean rasters (2018-2020) into memory

        Returns:
            seasonal_ndvi: dict {season: (H, W) array}
                seasons: 'spring', 'summer', 'autumn', 'winter'
        """
        ndvi_base_dir = self.config['data'].get('seasonal_ndvi_dir',
                                                 'data/model_ready/NDVI')

        seasons = ['spring', 'summer', 'autumn', 'winter']
        seasonal_ndvi = {}

        print("\nLoading seasonal NDVI rasters into memory...")
        for season in seasons:
            ndvi_path = os.path.join(ndvi_base_dir, f"{season}_ndvi_mean_clipped_2018-2020.tif")

            if not os.path.exists(ndvi_path):
                print(f"  Warning: {season} NDVI not found at {ndvi_path}, using zeros")
                # Use zeros if file not found
                seasonal_ndvi[season] = np.zeros_like(self.slope_id_raster, dtype=np.float32)
            else:
                with rasterio.open(ndvi_path) as src:
                    ndvi_raster = src.read(1).astype(np.float32)

                # Handle NoData
                if src.nodata is not None:
                    ndvi_raster = np.where(ndvi_raster == src.nodata, 0.0, ndvi_raster)

                ndvi_raster = np.nan_to_num(ndvi_raster, nan=0.0, posinf=0.0, neginf=0.0)

                # Apply valid area mask
                ndvi_raster = ndvi_raster * self.valid_area_mask

                seasonal_ndvi[season] = ndvi_raster
                print(f"  Loaded {season} NDVI: {ndvi_path}")

        print(f"  Loaded {len(seasonal_ndvi)} seasonal NDVI rasters successfully")
        return seasonal_ndvi

    def _get_season_from_month(self, month: int) -> str:
        """
        Determine season from month number

        Args:
            month: Month number (1-12)

        Returns:
            season: 'spring', 'summer', 'autumn', or 'winter'
        """
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'autumn'
        else:  # 12, 1, 2
            return 'winter'

    def _load_gnn_embeddings(self, path: str) -> Dict[int, np.ndarray]:
        """
        Load pre-computed GNN embeddings

        Args:
            path: Path to gnn_embeddings_128d.pt

        Returns:
            embeddings: dict {slope_id: (128,) array}
        """
        data = torch.load(path, weights_only=False)

        embeddings = {}
        # Check available keys and use appropriate one
        if 'embedding_dict' in data:
            # Direct dictionary format
            embeddings = {int(k): v.cpu().numpy() if torch.is_tensor(v) else v
                         for k, v in data['embedding_dict'].items()}
        elif 'cat_values' in data and 'embeddings' in data:
            # Parallel arrays format
            for slope_id, emb in zip(data['cat_values'], data['embeddings']):
                embeddings[int(slope_id)] = emb.cpu().numpy()  # (128,)
        else:
            raise KeyError(f"Unknown GNN embedding format. Available keys: {list(data.keys())}")

        return embeddings

    def _get_slope_bbox(self, slope_id: int) -> Tuple[int, int, int, int]:
        """
        Get bounding box for slope unit (centered patch)

        Args:
            slope_id: slope unit ID

        Returns:
            (row_min, row_max, col_min, col_max)
        """
        # Find slope pixels
        slope_pixels = np.where(self.slope_id_raster == slope_id)

        if len(slope_pixels[0]) == 0:
            raise ValueError(f"Slope {slope_id} not found in raster")

        # Calculate centroid
        row_center = int(np.mean(slope_pixels[0]))
        col_center = int(np.mean(slope_pixels[1]))

        # Create centered patch
        half_size = self.patch_size // 2

        row_min = max(0, row_center - half_size)
        row_max = min(self.slope_id_raster.shape[0], row_center + half_size)
        col_min = max(0, col_center - half_size)
        col_max = min(self.slope_id_raster.shape[1], col_center + half_size)

        # Ensure patch_size × patch_size
        if row_max - row_min < self.patch_size:
            if row_min == 0:
                row_max = min(self.patch_size, self.slope_id_raster.shape[0])
            else:
                row_min = max(0, row_max - self.patch_size)

        if col_max - col_min < self.patch_size:
            if col_min == 0:
                col_max = min(self.patch_size, self.slope_id_raster.shape[1])
            else:
                col_min = max(0, col_max - self.patch_size)

        return (row_min, row_max, col_min, col_max)

    def _load_static_features(self, bbox: Tuple[int, int, int, int],
                              event_month: int) -> np.ndarray:
        """
        Load static terrain features + new features + seasonal NDVI

        Args:
            bbox: (row_min, row_max, col_min, col_max)
            event_month: Month of event (1-12) for selecting seasonal NDVI

        Returns:
            static: (23, H, W) - 8 terrain (w/ aspect split) + 12 binary + 1 accm + 1 NDVI + 1 mask
        """
        row_min, row_max, col_min, col_max = bbox
        H, W = row_max - row_min, col_max - col_min

        static_channels = []

        for feature_name in self.static_features:
            # Determine raster path based on feature type
            if feature_name.startswith('rock'):
                # Litho features
                raster_path = os.path.join(self.litho_raster_dir, f"{feature_name}.tif")
            elif feature_name in ['agri', 'forest', 'grass', 'urban', 'water', 'bareland']:
                # Landcover features
                raster_path = os.path.join(self.landcover_raster_dir, f"{feature_name}.tif")
            elif feature_name == 'forestroad':
                # Forest road
                raster_path = os.path.join(self.forestroad_raster_dir, "forestroad.tif")
            else:
                # DEM_OUTPUTS features (aspect_deg, slope_deg, curv_plan, curv_prof, twi_use, tri_sd3, accm)
                raster_path = os.path.join(self.static_raster_dir, f"{feature_name}.tif")

            if not os.path.exists(raster_path):
                raise FileNotFoundError(f"Static raster not found: {raster_path}")

            with rasterio.open(raster_path) as src:
                data = src.read(1, window=((row_min, row_max), (col_min, col_max)))
                data = data.astype(np.float32)

                # Handle NoData values - replace with 0
                if src.nodata is not None:
                    data = np.where(data == src.nodata, 0.0, data)

                # Replace any remaining NaN/Inf with 0
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize using normalize_static_variable
            # This handles aspect_deg (circular), binary, log1p_robust (accm), robust_zscore
            normalized = normalize_static_variable(data, feature_name, self.norm_stats)

            # Ensure no NaN after normalization
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

            # normalize_static_variable returns (C, H, W) where C=2 for aspect, C=1 for others
            if normalized.shape[0] == 2:
                # aspect_deg returns [eastness, northness]
                static_channels.append(normalized[0])
                static_channels.append(normalized[1])
            else:
                # Other features return single channel
                static_channels.append(normalized[0])

        # Stack features: (22, H, W) - 8 terrain (w/ aspect split) + 12 binary + 1 accm + 1 NDVI
        static = np.stack(static_channels, axis=0)

        # Get valid area mask for this patch
        mask = self.valid_area_mask[row_min:row_max, col_min:col_max]

        # Apply mask to all static features (zero out invalid areas)
        # mask: 1 = valid, 0 = invalid
        static = static * mask[np.newaxis, ...]  # Broadcast mask to all channels

        # Add seasonal NDVI
        season = self._get_season_from_month(event_month)
        seasonal_ndvi_patch = self.seasonal_ndvi[season][row_min:row_max, col_min:col_max]

        # Normalize NDVI (use same normalization as other static features)
        # NDVI is already masked during loading
        seasonal_ndvi_normalized = normalize_static_variable(
            seasonal_ndvi_patch, 'seasonal_ndvi', self.norm_stats
        )
        seasonal_ndvi_normalized = np.nan_to_num(seasonal_ndvi_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Add mask as final channel
        static = np.concatenate([
            static,  # (22, H, W) - all features
            seasonal_ndvi_normalized[0:1, ...],  # (1, H, W) - seasonal NDVI
            mask[np.newaxis, ...]  # (1, H, W) - mask
        ], axis=0)  # (23, H, W)

        return static

    def _load_dynamic_sequence(self, event_date: datetime,
                               bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Load dynamic temporal sequence (5-day window)

        Args:
            event_date: target date
            bbox: (row_min, row_max, col_min, col_max)

        Returns:
            dynamic: (T=5, 3, H, W) - temporal rainfall sequences
        """
        row_min, row_max, col_min, col_max = bbox
        H, W = row_max - row_min, col_max - col_min

        temporal_sequence = []

        # 5-day window: [event_date - 4 days, ..., event_date]
        for t in range(self.temporal_window):
            current_date = event_date - timedelta(days=self.temporal_window - 1 - t)
            date_str = current_date.strftime('%Y%m%d')

            timestep_channels = []

            # Load rainfall features
            for channel_name in self.dynamic_variables['rainfall']:
                raster_path = os.path.join(
                    self.dynamic_raster_dir,
                    channel_name,
                    f"{date_str}_{channel_name}_mm_5179_30m.tif"
                )

                if os.path.exists(raster_path):
                    with rasterio.open(raster_path) as src:
                        data = src.read(1, window=((row_min, row_max), (col_min, col_max)))
                        data = data.astype(np.float32)

                        # Handle NoData values
                        if src.nodata is not None:
                            data = np.where(data == src.nodata, 0.0, data)

                        # Replace NaN/Inf
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Normalize (log1p + robust z-score)
                    # Extract month from current_date for monthly normalization
                    month = current_date.month if self.use_monthly_norm else None
                    normalized = normalize_dynamic_variable(
                        data, channel_name, self.norm_stats,
                        month=month, use_monthly=self.use_monthly_norm
                    )

                    # Ensure no NaN after normalization
                    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # Missing data: fill with zeros
                    normalized = np.zeros((H, W), dtype=np.float32)

                timestep_channels.append(normalized)

            # Stack channels for this timestep: (3, H, W)
            timestep_data = np.stack(timestep_channels, axis=0)
            temporal_sequence.append(timestep_data)

        # Stack temporal dimension: (T=5, 3, H, W)
        dynamic = np.stack(temporal_sequence, axis=0)

        # Apply valid area mask to all timesteps
        mask = self.valid_area_mask[row_min:row_max, col_min:col_max]
        # Broadcast: (T, C, H, W) * (1, 1, H, W)
        dynamic = dynamic * mask[np.newaxis, np.newaxis, ...]

        return dynamic

    def _rasterize_gnn_embedding(self, slope_id: int,
                                 bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Rasterize 128d GNN embedding to spatial feature map

        Args:
            slope_id: slope unit ID
            bbox: (row_min, row_max, col_min, col_max)

        Returns:
            gnn_raster: (128, H, W) - GNN embedding replicated across slope pixels
        """
        row_min, row_max, col_min, col_max = bbox
        H, W = row_max - row_min, col_max - col_min

        # GNN embedding (조건부)
        if self.use_gnn_embedding and self.gnn_embeddings is not None:
            # Get embedding vector for this slope
            emb_vector = self.gnn_embeddings.get(slope_id, np.zeros(128, dtype=np.float32))

            # Create slope mask for this patch
            slope_mask = (self.slope_id_raster[row_min:row_max, col_min:col_max] == slope_id)

            # Replicate embedding across spatial dimensions
            gnn_raster = np.zeros((128, H, W), dtype=np.float32)
            for c in range(128):
                gnn_raster[c] = emb_vector[c] * slope_mask  # Only apply to slope interior

            # Also apply valid area mask
            valid_mask = self.valid_area_mask[row_min:row_max, col_min:col_max]
            gnn_raster = gnn_raster * valid_mask[np.newaxis, ...]
        else:
            # GNN 비활성화: zero tensor 반환
            gnn_raster = np.zeros((128, H, W), dtype=np.float32)

        return gnn_raster

    def _load_kfs_prior(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Load KFS prior

        Args:
            bbox: (row_min, row_max, col_min, col_max)

        Returns:
            kfs_prior: (1, H, W)
        """
        row_min, row_max, col_min, col_max = bbox

        prior = self.kfs_prior[row_min:row_max, col_min:col_max]

        # Apply valid area mask
        valid_mask = self.valid_area_mask[row_min:row_max, col_min:col_max]
        prior = prior * valid_mask

        # Additional safety: ensure [0, 1] range (defensive)
        prior = np.clip(prior, 0.0, 1.0)

        prior = prior[np.newaxis, ...]  # (1, H, W)

        return prior

    def _create_slope_mask(self, slope_id: int,
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create slope interior mask

        Args:
            slope_id: slope unit ID
            bbox: (row_min, row_max, col_min, col_max)

        Returns:
            slope_mask: (1, H, W) - binary mask
        """
        row_min, row_max, col_min, col_max = bbox

        slope_mask = (self.slope_id_raster[row_min:row_max, col_min:col_max] == slope_id)
        slope_mask = slope_mask.astype(np.float32)[np.newaxis, ...]  # (1, H, W)

        return slope_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load single sample (slope-centered patch)

        Returns:
            {
                'static': (9, H, W) - terrain + seasonal_ndvi + mask,
                'dynamic': (T=5, 3, H, W) - temporal rainfall,
                'gnn_embedding': (128, H, W) - rasterized GNN embeddings,
                'kfs_prior': (1, H, W) - KFS prior,
                'slope_mask': (1, H, W) - slope interior mask,
                'label': int (0/1),
                'slope_id': int,
                'event_date': str (YYYYMMDD)
            }
        """
        sample_row = self.samples.iloc[idx]
        slope_id = int(sample_row['cat'])
        event_date = pd.to_datetime(sample_row['event_date'])
        label = int(sample_row['label'])

        # 1. Get slope bounding box (128×128 centered on slope)
        bbox = self._get_slope_bbox(slope_id)

        # 2. Load static features (with seasonal NDVI based on event month)
        static = self._load_static_features(bbox, event_date.month)  # (9, H, W)

        # 3. Load dynamic temporal sequence
        dynamic = self._load_dynamic_sequence(event_date, bbox)  # (5, 3, H, W)

        # 4. Rasterize GNN embeddings
        gnn_embedding = self._rasterize_gnn_embedding(slope_id, bbox)  # (128, H, W)

        # 5. Load KFS prior
        kfs_prior = self._load_kfs_prior(bbox)  # (1, H, W)

        # 6. Create slope mask
        slope_mask = self._create_slope_mask(slope_id, bbox)  # (1, H, W)

        # Convert to tensors
        sample = {
            'static': torch.from_numpy(static),
            'dynamic': torch.from_numpy(dynamic),
            'gnn_embedding': torch.from_numpy(gnn_embedding),
            'kfs_prior': torch.from_numpy(kfs_prior),
            'slope_mask': torch.from_numpy(slope_mask),
            'label': torch.tensor(label, dtype=torch.long),
            'slope_id': slope_id,
            'event_date': event_date.strftime('%Y%m%d')
        }

        # 7. Apply augmentation
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)

        # 8. Validate data for NaN/Inf
        for key in ['static', 'dynamic', 'gnn_embedding', 'kfs_prior', 'slope_mask']:
            if torch.isnan(sample[key]).any():
                print(f"[Data Validation] NaN detected in '{key}' for slope_id={slope_id}, label={label}")
                print(f"  NaN count: {torch.isnan(sample[key]).sum().item()}/{sample[key].numel()}")
                # Replace NaN with 0
                sample[key] = torch.nan_to_num(sample[key], nan=0.0)

            if torch.isinf(sample[key]).any():
                print(f"[Data Validation] Inf detected in '{key}' for slope_id={slope_id}, label={label}")
                print(f"  Inf count: {torch.isinf(sample[key]).sum().item()}/{sample[key].numel()}")
                # Replace Inf with 0
                sample[key] = torch.nan_to_num(sample[key], nan=0.0, posinf=0.0, neginf=0.0)

        return sample


class HierarchicalDatasetDynamic(HierarchicalDataset):
    """
    Hierarchical Dataset with Dynamic Negative Sampling

    Extends HierarchicalDataset to support dynamic negative sampling.
    Each epoch, negative samples are regenerated from the full slope pool.

    Key Features:
    - Positive samples: Fixed from CSV file
    - Negative samples: Dynamically generated each epoch
    - 80% of negatives: dates matched to positive distribution
    - 20% of negatives: completely random dates

    Usage:
        >>> from dynamic_negative_sampler import DynamicNegativeSampler
        >>>
        >>> # Initialize sampler
        >>> sampler = DynamicNegativeSampler(
        ...     positive_samples_df=positive_df,
        ...     all_slope_ids=list(range(1, 87697)),
        ...     date_range=('2019-01-01', '2020-09-30')
        ... )
        >>>
        >>> # Create dataset
        >>> dataset = HierarchicalDatasetDynamic(
        ...     config=config,
        ...     negative_sampler=sampler,
        ...     gnn_embedding_path='path/to/embeddings.pt'
        ... )
        >>>
        >>> # Refresh negatives each epoch (in training loop)
        >>> dataset.refresh_negative_samples(epoch=1)
    """

    def __init__(
        self,
        config: Dict,
        negative_sampler,  # DynamicNegativeSampler instance
        gnn_embedding_path: str,
        augment: bool = False,
        aug_prob: float = 0.5
    ):
        """
        Args:
            config: Model configuration dict
            negative_sampler: DynamicNegativeSampler instance
            gnn_embedding_path: Path to GNN embeddings
            augment: Whether to apply geometric augmentation
            aug_prob: Augmentation probability
        """
        self.negative_sampler = negative_sampler

        # Initialize with combined samples (pos + neg for epoch 0)
        initial_samples = self.negative_sampler.get_combined_samples(epoch=0)

        # Call parent init with initial samples
        super().__init__(
            config=config,
            samples_df=initial_samples,
            gnn_embedding_path=gnn_embedding_path,
            augment=augment,
            aug_prob=aug_prob
        )

        print(f"\n=== HierarchicalDatasetDynamic Initialized ===")
        print(f"Initial samples: {len(self.samples)}")
        print(f"  Positive: {(self.samples['label']==1).sum()}")
        print(f"  Negative: {(self.samples['label']==0).sum()}")

    def refresh_negative_samples(self, epoch: int):
        """
        Regenerate negative samples for the current epoch.

        This should be called at the beginning of each epoch in the training loop.

        Args:
            epoch: Current epoch number
        """
        print(f"\n{'='*80}")
        print(f"[Epoch {epoch}] Refreshing negative samples...")
        print(f"{'='*80}")

        # Generate new combined samples (pos + new negatives)
        new_samples = self.negative_sampler.get_combined_samples(epoch=epoch)

        # Update internal samples DataFrame
        self.samples = new_samples.reset_index(drop=True)

        print(f"Updated dataset size: {len(self.samples)}")
        print(f"  Positive: {(self.samples['label']==1).sum()}")
        print(f"  Negative: {(self.samples['label']==0).sum()}")
        print(f"{'='*80}\n")
