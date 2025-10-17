"""
Stage 2 Dual-Stream Model Inference Script

This script performs landslide risk prediction using the trained Dual-Stream model:
- Loads pre-trained model (Stage 2)
- Loads temporal sequences for specified date
- Generates pixel-level risk map with attention visualization

Usage:
    python src/inference/predict_dual.py \
        --checkpoint experiments/stage2_dual/<timestamp>/checkpoints/model_dual_best.pth \
        --date 20200728 \
        --output outputs/risk_maps/risk_map_dual_20200728.tif \
        --attention_output outputs/risk_maps/attention_map_20200728.tif

Author: Landslide Risk Analysis Project
Date: 2025-01-17
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_dual_unet import DualHierarchicalGNNUNet


# ============================================================
# Dual-Stream Predictor
# ============================================================

class DualStreamPredictor:
    """
    Dual-Stream model predictor with temporal window support
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to use
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']
        
        # Extract configuration
        self.slope_id_raster_path = self.config['data']['slope_id_raster_path']
        self.gnn_raster_path = self.config['data']['gnn_raster_path']
        self.static_raster_dir = self.config['data']['static_raster_dir']
        self.static_variables = self.config['data']['static_variables']
        self.dynamic_raster_base = self.config['data']['dynamic_raster_base']
        self.dynamic_variables = self.config['data']['dynamic_variables']
        self.window_size = self.config['data']['window_size']
        self.patch_size = self.config['data']['patch_size']
        
        # Load raster metadata
        print(f"Loading raster metadata...")
        with rasterio.open(self.slope_id_raster_path) as src:
            self.slope_id_raster = src.read(1)
            self.raster_height, self.raster_width = src.shape
            self.raster_transform = src.transform
            self.raster_crs = src.crs
            self.raster_bounds = src.bounds
        
        with rasterio.open(self.gnn_raster_path) as src:
            gnn_raster_full = src.read(1)
        
        # Clip GNN raster to match slope_id_raster size (공간정합 유지하며 범위만 조정)
        gnn_h, gnn_w = gnn_raster_full.shape
        target_h, target_w = self.raster_height, self.raster_width
        
        if gnn_h != target_h or gnn_w != target_w:
            print(f"  GNN raster size mismatch: ({gnn_h}, {gnn_w}) → clipping to ({target_h}, {target_w})")
            # Clip to target size (no interpolation, spatial alignment preserved)
            clip_h = min(gnn_h, target_h)
            clip_w = min(gnn_w, target_w)
            self.gnn_raster = gnn_raster_full[:clip_h, :clip_w]
            
            # If GNN is smaller, pad with zeros
            if clip_h < target_h or clip_w < target_w:
                gnn_padded = np.zeros((target_h, target_w), dtype=gnn_raster_full.dtype)
                gnn_padded[:clip_h, :clip_w] = self.gnn_raster
                self.gnn_raster = gnn_padded
        else:
            self.gnn_raster = gnn_raster_full
        
        print(f"  Raster size: {self.raster_height} × {self.raster_width}")
        
        # Build model
        print(f"Building model...")
        graph_data = torch.load(self.config['data']['graph_path'], weights_only=False)
        static_dim = graph_data.x.shape[1]
        
        self.model = DualHierarchicalGNNUNet(
            static_dim=static_dim,
            gnn_hidden=self.config['model_dual']['gnn_hidden'],
            gnn_layers=self.config['model_dual']['gnn_layers'],
            gnn_type=self.config['model_dual']['gnn_type'],
            gnn_dropout=self.config['model_dual']['gnn_dropout'],
            gat_heads=self.config['model_dual']['gat_heads'],
            state_channels=len(self.static_variables),
            trigger_channels=sum(len(chs) for chs in self.dynamic_variables.values()),
            state_base_channels=self.config['model_dual']['state_base_channels'],
            trigger_hidden_channels=self.config['model_dual']['trigger_hidden_channels'],
            trigger_num_layers=self.config['model_dual']['trigger_num_layers'],
            fusion_num_heads=self.config['model_dual']['fusion_num_heads'],
            decoder_depth=self.config['model_dual']['decoder_depth']
        )
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Model loaded successfully")
        print(f"  Training metrics: AUC-ROC={self.checkpoint['metrics']['auc_roc']:.4f}, "
              f"F1={self.checkpoint['metrics']['f1']:.4f}")
    
    def _load_temporal_sequence(
        self,
        event_date: datetime,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int
    ) -> np.ndarray:
        """
        Load temporal sequence of dynamic rasters
        
        Args:
            event_date: Target prediction date
            row_start, row_end, col_start, col_end: Patch bounds
        
        Returns:
            temporal_data: (T, C_trigger, H, W) array
        """
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        temporal_patches = []
        
        for t in range(self.window_size):
            current_date = event_date - timedelta(days=self.window_size - 1 - t)
            date_str = current_date.strftime('%Y%m%d')
            
            timestep_patches = []
            
            for var_group, channels in self.dynamic_variables.items():
                for channel in channels:
                    raster_path = os.path.join(
                        self.dynamic_raster_base,
                        channel,
                        f"{date_str}_{channel}_mm_5179_30m.tif"
                    )
                    
                    if not os.path.exists(raster_path):
                        patch = np.zeros((row_end - row_start, col_end - col_start), dtype=np.float32)
                    else:
                        with rasterio.open(raster_path) as src:
                            patch = src.read(1, window=window).astype(np.float32)
                        
                        # Pad if needed (dynamic 래스터가 작을 수 있음)
                        expected_h = row_end - row_start
                        expected_w = col_end - col_start
                        if patch.shape[0] < expected_h or patch.shape[1] < expected_w:
                            patch_padded = np.zeros((expected_h, expected_w), dtype=np.float32)
                            h, w = patch.shape
                            patch_padded[:h, :w] = patch
                            patch = patch_padded
                    
                    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                    timestep_patches.append(patch)
            
            timestep_data = np.stack(timestep_patches, axis=0)
            temporal_patches.append(timestep_data)
        
        temporal_data = np.stack(temporal_patches, axis=0)
        return temporal_data
    
    def _load_state_patch(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int
    ) -> np.ndarray:
        """
        Load static terrain features (State stream)
        
        Returns:
            state_patch: (C_state, H, W) array
        """
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        state_patches = []
        
        for var in self.static_variables:
            raster_path = os.path.join(self.static_raster_dir, f"{var}.tif")
            
            with rasterio.open(raster_path) as src:
                patch = src.read(1, window=window).astype(np.float32)
            
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
        
        state_patch = np.stack(state_patches, axis=0)
        state_patch = np.clip(state_patch, -10, 10)
        
        return state_patch
    
    def predict_full_extent(
        self,
        event_date: datetime,
        stride: int = None,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict full extent with sliding window
        
        Args:
            event_date: Target prediction date
            stride: Stride for sliding window (default: patch_size // 2)
            return_attention: Whether to return attention maps
        
        Returns:
            risk_map: (H, W) probability map
            attention_map: (H, W) aggregated attention weights (if return_attention=True)
        """
        if stride is None:
            stride = self.patch_size // 2
        
        print(f"Predicting for date: {event_date.strftime('%Y-%m-%d')}")
        print(f"  Temporal window: {self.window_size} days")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Stride: {stride}")
        
        # Initialize output arrays
        risk_map = np.zeros((self.raster_height, self.raster_width), dtype=np.float32)
        attention_map = np.zeros((self.raster_height, self.raster_width), dtype=np.float32)
        count_map = np.zeros((self.raster_height, self.raster_width), dtype=np.float32)
        
        # Calculate number of patches
        num_rows = (self.raster_height - self.patch_size) // stride + 1
        num_cols = (self.raster_width - self.patch_size) // stride + 1
        total_patches = num_rows * num_cols
        
        print(f"  Total patches: {total_patches} ({num_rows} × {num_cols})")
        
        with torch.no_grad():
            with tqdm(total=total_patches, desc="Processing") as pbar:
                for row_idx in range(num_rows):
                    for col_idx in range(num_cols):
                        row_start = row_idx * stride
                        col_start = col_idx * stride
                        row_end = row_start + self.patch_size
                        col_end = col_start + self.patch_size
                        
                        # Handle boundary
                        if row_end > self.raster_height:
                            row_end = self.raster_height
                            row_start = row_end - self.patch_size
                        if col_end > self.raster_width:
                            col_end = self.raster_width
                            col_start = col_end - self.patch_size
                        
                        # Extract patches (all rasters are aligned, always 128×128)
                        gnn_patch = self.gnn_raster[row_start:row_end, col_start:col_end]
                        gnn_patch = np.where(gnn_patch < 0, 0.0, gnn_patch)
                        gnn_patch = gnn_patch[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
                        
                        state_patch = self._load_state_patch(row_start, row_end, col_start, col_end)
                        state_patch = state_patch[np.newaxis, ...]  # (1, C, H, W)
                        
                        trigger_sequence = self._load_temporal_sequence(
                            event_date, row_start, row_end, col_start, col_end
                        )
                        trigger_sequence = trigger_sequence[np.newaxis, ...]  # (1, T, C, H, W)
                        
                        # Convert to tensors
                        gnn_patch = torch.from_numpy(gnn_patch).float().to(self.device)
                        state_patch = torch.from_numpy(state_patch).float().to(self.device)
                        trigger_sequence = torch.from_numpy(trigger_sequence).float().to(self.device)
                        
                        # Forward pass
                        pixel_logits, intermediate_features = self.model(
                            gnn_patch=gnn_patch,
                            state_patch=state_patch,
                            trigger_sequence=trigger_sequence
                        )
                        
                        # Convert to probabilities
                        pixel_probs = torch.sigmoid(pixel_logits[0, 0]).cpu().numpy()
                        
                        # Calculate attention (if requested)
                        if return_attention:
                            zone_patch = self.slope_id_raster[row_start:row_end, col_start:col_end]
                            zone_patch = torch.from_numpy(zone_patch[np.newaxis, ...]).long().to(self.device)
                            
                            # Dummy cat and label
                            cats = torch.tensor([1], dtype=torch.long, device=self.device)
                            labels = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                            
                            _, attention_weights = self.model.compute_loss(
                                pixel_logits,
                                intermediate_features,
                                zone_patch,
                                cats,
                                labels
                            )
                            
                            attention_weights = attention_weights[0].cpu().numpy()
                            attention_map[row_start:row_end, col_start:col_end] += attention_weights
                        
                        # Accumulate
                        risk_map[row_start:row_end, col_start:col_end] += pixel_probs
                        count_map[row_start:row_end, col_start:col_end] += 1
                        
                        pbar.update(1)
        
        # Average overlapping regions
        risk_map = np.divide(risk_map, count_map, where=count_map > 0)
        
        if return_attention:
            attention_map = np.divide(attention_map, count_map, where=count_map > 0)
            return risk_map, attention_map
        
        return risk_map, None
    
    def save_risk_map(
        self,
        risk_map: np.ndarray,
        output_path: str
    ):
        """Save risk map to GeoTIFF"""
        print(f"\nSaving risk map to: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self.raster_height,
            width=self.raster_width,
            count=1,
            dtype=np.float32,
            crs=self.raster_crs,
            transform=self.raster_transform,
            compress='lzw'
        ) as dst:
            dst.write(risk_map, 1)
        
        print(f"  ✓ Risk map saved")


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dual-Stream Model Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, required=True,
                       help='Prediction date (YYYYMMDD)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output risk map path')
    parser.add_argument('--attention_output', type=str, default=None,
                       help='Optional: Output attention map path')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (default: patch_size // 2)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Parse date
    event_date = datetime.strptime(args.date, '%Y%m%d')
    
    # Create predictor
    predictor = DualStreamPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Predict
    return_attention = args.attention_output is not None
    risk_map, attention_map = predictor.predict_full_extent(
        event_date=event_date,
        stride=args.stride,
        return_attention=return_attention
    )
    
    # Save risk map
    predictor.save_risk_map(risk_map, args.output)
    
    # Save attention map
    if attention_map is not None:
        predictor.save_risk_map(attention_map, args.attention_output)
        print(f"  ✓ Attention map saved to: {args.attention_output}")
    
    # Statistics
    print("\n" + "="*70)
    print("Prediction Statistics")
    print("="*70)
    print(f"  Min risk: {np.nanmin(risk_map):.4f}")
    print(f"  Max risk: {np.nanmax(risk_map):.4f}")
    print(f"  Mean risk: {np.nanmean(risk_map):.4f}")
    print(f"  Std risk: {np.nanstd(risk_map):.4f}")
    
    if attention_map is not None:
        print(f"\n  Attention stats:")
        print(f"    Min: {np.nanmin(attention_map):.4f}")
        print(f"    Max: {np.nanmax(attention_map):.4f}")
        print(f"    Mean: {np.nanmean(attention_map):.4f}")
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

