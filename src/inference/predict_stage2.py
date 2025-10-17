"""
Stage 2 U-Net Inference Script

This script performs inference using the Stage 2 trained model (frozen GNN + U-Net).
It uses the same multi-channel input as training:
- Channel 0: GNN susceptibility (pre-generated)
- Channels 1-8: High-resolution static rasters
- Channels 9-11: Dynamic rainfall rasters

Usage:
    python src/inference/predict_stage2.py \
        --checkpoint experiments/stage2_unet/<timestamp>/checkpoints/model_stage2_best.pth \
        --date 20200728 \
        --output outputs/risk_maps/risk_map_stage2_20200728.tif

Author: Landslide Risk Analysis Project
Date: 2025-01-17
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import HierarchicalGNNUNet


class Stage2Predictor:
    """
    Stage 2 U-Net predictor with frozen GNN
    
    Args:
        checkpoint_path: Path to Stage 2 best checkpoint
        gnn_raster_path: Path to GNN susceptibility raster
        static_raster_dir: Directory containing static rasters
        static_raster_files: List of static raster filenames
        slope_id_raster_path: Path to slope ID raster
        dynamic_raster_base: Base path to dynamic rasters
        dynamic_variables: List of dynamic variables
        device: Device (cuda or cpu)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        gnn_raster_path: str,
        static_raster_dir: str,
        static_raster_files: list,
        slope_id_raster_path: str,
        dynamic_raster_base: str,
        dynamic_variables: list,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store paths
        self.slope_id_raster_path = slope_id_raster_path
        self.gnn_raster_path = gnn_raster_path
        self.dynamic_raster_base = dynamic_raster_base
        self.dynamic_variables = dynamic_variables
        self.static_raster_paths = [
            os.path.join(static_raster_dir, fname) for fname in static_raster_files
        ]
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: AUC-ROC={checkpoint['metrics']['auc_roc']:.4f}, "
              f"F1={checkpoint['metrics']['f1']:.4f}")
        
        # Load rasters
        print("Loading rasters...")
        
        # GNN raster
        print(f"  GNN raster: {gnn_raster_path}")
        with rasterio.open(gnn_raster_path) as src:
            self.gnn_raster = src.read(1)
        
        # Slope ID raster
        print(f"  Slope ID raster: {slope_id_raster_path}")
        with rasterio.open(slope_id_raster_path) as src:
            self.slope_id_raster = src.read(1)
            self.raster_profile = src.profile
            self.height, self.width = src.shape
        
        print(f"  Raster size: {self.height} × {self.width}")
        print(f"  Static rasters: {len(self.static_raster_paths)} files")
        
        # Build model
        print("Building model...")
        
        # Get graph data for static_dim
        graph_data = torch.load(self.config['data']['graph_path'], weights_only=False)
        static_dim = graph_data.x.shape[1]
        
        # Total channels
        num_static = len(static_raster_files)
        num_dynamic = len(dynamic_variables)
        unet_in_channels = 1 + num_static + num_dynamic
        
        # Build model
        model = HierarchicalGNNUNet(
            static_dim=static_dim,
            gnn_hidden=self.config['model_unet']['gnn_hidden'],
            gnn_layers=self.config['model_unet']['gnn_layers'],
            gnn_type=self.config['model_unet']['gnn_type'],
            gnn_dropout=self.config['model_unet']['gnn_dropout'],
            gat_heads=self.config['model_unet']['gat_heads'],
            dynamic_channels=unet_in_channels - 1,
            unet_base_channels=self.config['model_unet']['unet_base_channels'],
            unet_depth=self.config['model_unet']['unet_depth']
        )
        
        # Modify U-Net input layer (same as training)
        model.unet.inc = nn.Sequential(
            nn.Conv2d(unet_in_channels, self.config['model_unet']['unet_base_channels'], 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.config['model_unet']['unet_base_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.config['model_unet']['unet_base_channels'], 
                     self.config['model_unet']['unet_base_channels'], 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.config['model_unet']['unet_base_channels']),
            nn.ReLU(inplace=True)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.unet_in_channels = unet_in_channels
        
        print(f"  Model loaded: {unet_in_channels} input channels")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def load_dynamic_rasters(self, date_str: str) -> np.ndarray:
        """Load dynamic rasters for given date"""
        dynamic_rasters = []
        
        for var in self.dynamic_variables:
            raster_path = os.path.join(
                self.dynamic_raster_base,
                var,
                f"{date_str}_{var}_mm_5179_30m.tif"
            )
            
            if not os.path.exists(raster_path):
                print(f"  Warning: {raster_path} not found, using zeros")
                dynamic_rasters.append(np.zeros((self.height, self.width), dtype=np.float32))
            else:
                with rasterio.open(raster_path) as src:
                    raster = src.read(1).astype(np.float32)
                    # Handle NaN
                    raster = np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)
                    dynamic_rasters.append(raster)
        
        return np.stack(dynamic_rasters, axis=0)  # (N_dynamic, H, W)
    
    def load_static_rasters(self) -> np.ndarray:
        """
        Load all static rasters using coordinate-based reading
        
        This ensures exact spatial alignment without interpolation.
        Uses the same transform-based approach as training.
        """
        from rasterio.windows import from_bounds
        
        static_rasters = []
        
        # Get bounds from slope ID raster
        with rasterio.open(self.slope_id_raster_path) as ref:
            ref_bounds = ref.bounds
            ref_transform = ref.transform
        
        for raster_path in self.static_raster_paths:
            with rasterio.open(raster_path) as src:
                # Calculate window based on bounds (coordinate-based, not pixel-based)
                window = from_bounds(
                    ref_bounds.left, ref_bounds.bottom,
                    ref_bounds.right, ref_bounds.top,
                    transform=src.transform
                )
                
                # Read using window (may be float, rasterio handles it)
                raster = src.read(1, window=window, 
                                 out_shape=(self.height, self.width),
                                 resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
                
                # Handle NaN
                raster = np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize per-channel (same as training)
                valid_mask = ~np.isnan(raster) & (raster != 0)
                if valid_mask.sum() > 0:
                    mean = raster[valid_mask].mean()
                    std = raster[valid_mask].std()
                    if std > 1e-6:
                        raster = (raster - mean) / (std + 1e-6)
                    else:
                        raster = raster - mean
                
                static_rasters.append(raster)
        
        return np.stack(static_rasters, axis=0)  # (N_static, H, W)
    
    def predict(
        self,
        date_str: str,
        patch_size: int = 512,
        overlap: int = 64
    ) -> np.ndarray:
        """
        Predict risk map for given date using patch-based inference
        
        Args:
            date_str: Date string (YYYYMMDD)
            patch_size: Patch size for inference
            overlap: Overlap between patches
        
        Returns:
            risk_map: (H, W) risk probability map
        """
        print(f"\nGenerating risk map for {date_str}...")
        
        # Load all rasters
        print("  Loading rasters...")
        gnn_raster = self.gnn_raster.copy()
        gnn_raster = np.where(gnn_raster < 0, 0.0, gnn_raster)  # Handle NoData
        
        static_rasters = self.load_static_rasters()
        dynamic_rasters = self.load_dynamic_rasters(date_str)
        
        # Combine all channels
        print("  Combining channels...")
        full_input = np.concatenate([
            gnn_raster[np.newaxis, ...],  # (1, H, W)
            static_rasters,                # (N_static, H, W)
            dynamic_rasters                # (N_dynamic, H, W)
        ], axis=0)  # (C_total, H, W)
        
        # Clip extreme values
        full_input = np.clip(full_input, -10, 10)
        
        print(f"  Input shape: {full_input.shape}")
        print(f"  Input range: [{full_input.min():.2f}, {full_input.max():.2f}]")
        
        # Patch-based prediction
        print(f"  Running patch-based inference (patch_size={patch_size}, overlap={overlap})...")
        
        risk_map = np.zeros((self.height, self.width), dtype=np.float32)
        count_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        stride = patch_size - overlap
        
        # Calculate total patches
        num_patches_h = (self.height - overlap) // stride + 1
        num_patches_w = (self.width - overlap) // stride + 1
        total_patches = num_patches_h * num_patches_w
        
        with torch.no_grad():
            with tqdm(total=total_patches, desc="  Predicting") as pbar:
                for i in range(0, self.height, stride):
                    for j in range(0, self.width, stride):
                        # Extract patch
                        i_end = min(i + patch_size, self.height)
                        j_end = min(j + patch_size, self.width)
                        
                        patch = full_input[:, i:i_end, j:j_end]
                        
                        # Pad if necessary
                        if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                            padded = np.zeros((self.unet_in_channels, patch_size, patch_size), dtype=np.float32)
                            padded[:, :patch.shape[1], :patch.shape[2]] = patch
                            patch = padded
                        
                        # Convert to tensor
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)  # (1, C, H, W)
                        
                        # Predict
                        logits = self.model.forward_stage2(patch_tensor)  # (1, 1, H, W)
                        probs = torch.sigmoid(logits[0, 0]).cpu().numpy()  # (H, W)
                        
                        # Merge into full map
                        pred_h, pred_w = i_end - i, j_end - j
                        risk_map[i:i_end, j:j_end] += probs[:pred_h, :pred_w]
                        count_map[i:i_end, j:j_end] += 1
                        
                        pbar.update(1)
        
        # Average overlapping regions
        risk_map = np.divide(risk_map, count_map, where=count_map > 0)
        
        # Mask NoData regions
        valid_mask = (self.slope_id_raster > 0)
        risk_map[~valid_mask] = -9999
        
        print(f"  ✓ Prediction complete!")
        print(f"    Valid pixels: {valid_mask.sum():,} ({valid_mask.sum()/(self.height*self.width)*100:.1f}%)")
        print(f"    Risk range: [{risk_map[valid_mask].min():.4f}, {risk_map[valid_mask].max():.4f}]")
        print(f"    Risk mean: {risk_map[valid_mask].mean():.4f}")
        
        return risk_map
    
    def save_risk_map(self, risk_map: np.ndarray, output_path: str):
        """Save risk map as GeoTIFF"""
        print(f"\nSaving risk map to {output_path}...")
        
        # Update profile
        output_profile = self.raster_profile.copy()
        output_profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=-9999
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(risk_map.astype(np.float32), 1)
        
        print(f"  ✓ Risk map saved!")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stage 2 U-Net Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 2 best checkpoint')
    parser.add_argument('--gnn_raster', type=str,
                       default='experiments/stage1_gnn/20251017_150408/gnn_susceptibility_30m.tif',
                       help='Path to GNN susceptibility raster')
    parser.add_argument('--static_raster_dir', type=str,
                       default='data/model_ready/DEM_OUTPUTS',
                       help='Directory containing static rasters')
    parser.add_argument('--slope_id_raster', type=str,
                       default='data/model_ready/zone_raster_gyeongnam_5179_30m.tif',
                       help='Path to slope ID raster')
    parser.add_argument('--dynamic_raster_base', type=str,
                       default='data/processed/gyeongnam/LDAPS/TIF_5179_30m_20190101_20200930_clipped',
                       help='Base path to dynamic rasters')
    parser.add_argument('--date', type=str, required=True,
                       help='Target date (YYYYMMDD)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output risk map path')
    parser.add_argument('--patch_size', type=int, default=512,
                       help='Patch size for inference')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between patches')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Static raster files (must match training)
    static_raster_files = [
        "slope_deg.tif",
        "aspect_deg.tif",
        "curv_plan.tif",
        "curv_prof.tif",
        "twi_use.tif",
        "lnspi_use.tif",
        "tri_sd3.tif",
        "accm.tif"
    ]
    
    # Dynamic variables (must match training)
    dynamic_variables = ["acc3d", "acc7d", "peak1h"]
    
    print("="*70)
    print("Stage 2 U-Net Inference")
    print("="*70)
    
    # Create predictor
    predictor = Stage2Predictor(
        checkpoint_path=args.checkpoint,
        gnn_raster_path=args.gnn_raster,
        static_raster_dir=args.static_raster_dir,
        static_raster_files=static_raster_files,
        slope_id_raster_path=args.slope_id_raster,
        dynamic_raster_base=args.dynamic_raster_base,
        dynamic_variables=dynamic_variables,
        device=args.device
    )
    
    # Predict
    risk_map = predictor.predict(
        date_str=args.date,
        patch_size=args.patch_size,
        overlap=args.overlap
    )
    
    # Save
    predictor.save_risk_map(risk_map, args.output)
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print(f"Output: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()

