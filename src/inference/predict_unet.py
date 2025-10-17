"""
Inference Script for Hierarchical GNN-U-Net Model

Generates pixel-level landslide risk maps using trained model.
Uses sliding window with overlap to avoid seam artifacts.

Usage:
    # Single date prediction
    python src/inference/predict_unet.py --checkpoint experiments/stage2_unet/20251017_171645/checkpoints/model_stage2_best.pth --date 20200728 --output outputs/risk_maps/risk_map_20200728.tif
    
    # Multiple dates
    python src/inference/predict_unet.py \\
        --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \\
        --dates 20200615 20200715 20200815 \\
        --output_dir outputs/risk_maps/

"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import HierarchicalGNNUNet
from src.utils.config import load_config


class UNetPredictor:
    """
    Hierarchical GNN-U-Net predictor with sliding window inference
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        patch_size: int = 512,
        overlap: int = 64
    ):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
            patch_size: Patch size for sliding window
            overlap: Overlap between patches
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load config
        self.config = checkpoint['config']
        
        # Load graph data
        print(f"Loading graph data from {self.config['data']['graph_path']}...")
        graph_data = torch.load(
            self.config['data']['graph_path'],
            map_location=self.device,
            weights_only=False
        )
        
        self.static_x = graph_data.x.to(self.device)
        self.edge_index = graph_data.edge_index.to(self.device)
        self.edge_attr = graph_data.edge_attr.to(self.device) if graph_data.edge_attr is not None else None
        self.cat_values = graph_data.cat.cpu().numpy()
        
        # Create cat to node mapping
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.cat_values)}
        
        # Load slope ID raster
        print(f"Loading slope ID raster from {self.config['data']['slope_id_raster_path']}...")
        with rasterio.open(self.config['data']['slope_id_raster_path']) as src:
            self.slope_id_raster = src.read(1)
            self.height, self.width = src.shape
            self.transform = src.transform
            self.crs = src.crs
            self.profile = src.profile.copy()
        
        print(f"  Raster size: {self.height} × {self.width}")
        
        # Build model
        print("Building model...")
        self.model = HierarchicalGNNUNet(
            static_dim=self.static_x.shape[1],
            gnn_hidden=self.config['model_unet']['gnn_hidden'],
            gnn_layers=self.config['model_unet']['gnn_layers'],
            gnn_type=self.config['model_unet']['gnn_type'],
            gnn_dropout=self.config['model_unet']['gnn_dropout'],
            gat_heads=self.config['model_unet']['gat_heads'],
            dynamic_channels=len(self.config['data']['dynamic_variables']),
            unet_base_channels=self.config['model_unet']['unet_base_channels'],
            unet_depth=self.config['model_unet']['unet_depth']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Device: {self.device}")
        print(f"  Patch size: {self.patch_size}, Overlap: {self.overlap}, Stride: {self.stride}")
    
    def _compute_gnn_susceptibility(self) -> np.ndarray:
        """
        Compute GNN-based static susceptibility map (once for all predictions)
        
        Returns:
            gnn_raster: (H, W) susceptibility map
        """
        print("Computing GNN static susceptibility...")
        
        with torch.no_grad():
            gnn_logits = self.model.forward_stage1(
                self.static_x,
                self.edge_index,
                self.edge_attr
            )
            gnn_probs = torch.sigmoid(gnn_logits).cpu().numpy()
        
        # Map to raster
        gnn_raster = np.zeros((self.height, self.width), dtype=np.float32)
        
        unique_cats = np.unique(self.slope_id_raster)
        for cat in tqdm(unique_cats, desc="  Mapping GNN to raster"):
            if cat == 0:
                continue
            if int(cat) in self.cat_to_node:
                node_idx = self.cat_to_node[int(cat)]
                gnn_raster[self.slope_id_raster == cat] = gnn_probs[node_idx]
        
        print(f"  ✓ GNN susceptibility computed")
        print(f"    Range: [{gnn_raster.min():.4f}, {gnn_raster.max():.4f}]")
        print(f"    Mean: {gnn_raster.mean():.4f}")
        
        return gnn_raster
    
    def _load_dynamic_rasters(self, date: str) -> np.ndarray:
        """
        Load dynamic raster stack for a given date
        
        Args:
            date: Date string (YYYYMMDD)
        
        Returns:
            dynamic_stack: (C, H, W) dynamic features
        """
        print(f"Loading dynamic rasters for {date}...")
        
        dynamic_rasters = []
        raster_base = self.config['data']['raster_base_path']
        
        for var in self.config['data']['dynamic_variables']:
            raster_path = os.path.join(raster_base, var, f"{date}_{var}_mm_5179_30m.tif")
            
            if not os.path.exists(raster_path):
                print(f"  Warning: {raster_path} not found, using zeros")
                dynamic_rasters.append(np.zeros((self.height, self.width), dtype=np.float32))
            else:
                with rasterio.open(raster_path) as src:
                    data = src.read(1).astype(np.float32)
                    dynamic_rasters.append(data)
                print(f"  ✓ Loaded {var}")
        
        dynamic_stack = np.stack(dynamic_rasters, axis=0)  # (C, H, W)
        return dynamic_stack
    
    def _predict_with_sliding_window(
        self,
        multi_channel_input: np.ndarray
    ) -> np.ndarray:
        """
        Predict using sliding window with overlap
        
        Args:
            multi_channel_input: (C, H, W) multi-channel input
        
        Returns:
            risk_map: (H, W) predicted risk probabilities
        """
        print(f"Predicting with sliding window (patch={self.patch_size}, overlap={self.overlap})...")
        
        C, H, W = multi_channel_input.shape
        risk_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Calculate number of patches
        n_rows = int(np.ceil((H - self.overlap) / self.stride))
        n_cols = int(np.ceil((W - self.overlap) / self.stride))
        total_patches = n_rows * n_cols
        
        print(f"  Total patches: {total_patches} ({n_rows} × {n_cols})")
        
        with torch.no_grad():
            pbar = tqdm(total=total_patches, desc="  Processing patches")
            
            for i in range(0, H, self.stride):
                for j in range(0, W, self.stride):
                    # Extract patch
                    i_end = min(i + self.patch_size, H)
                    j_end = min(j + self.patch_size, W)
                    
                    patch = multi_channel_input[:, i:i_end, j:j_end]
                    
                    # Pad if necessary
                    if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                        padded = np.zeros((C, self.patch_size, self.patch_size), dtype=np.float32)
                        padded[:, :patch.shape[1], :patch.shape[2]] = patch
                        patch = padded
                    
                    # Predict
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                    logits = self.model.forward_stage2(patch_tensor)
                    probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
                    
                    # Merge (accumulate)
                    pred_h = i_end - i
                    pred_w = j_end - j
                    risk_map[i:i_end, j:j_end] += probs[:pred_h, :pred_w]
                    count_map[i:i_end, j:j_end] += 1
                    
                    pbar.update(1)
            
            pbar.close()
        
        # Average overlapping regions
        risk_map = np.divide(risk_map, count_map, where=count_map > 0)
        
        print(f"  ✓ Prediction complete")
        print(f"    Range: [{risk_map.min():.4f}, {risk_map.max():.4f}]")
        print(f"    Mean: {risk_map.mean():.4f}")
        
        return risk_map
    
    def predict(
        self,
        date: str,
        output_path: str,
        save_gnn_susceptibility: bool = False
    ):
        """
        Generate risk map for a given date
        
        Args:
            date: Date string (YYYYMMDD)
            output_path: Output GeoTIFF path
            save_gnn_susceptibility: Whether to save GNN susceptibility map separately
        """
        print("\n" + "="*70)
        print(f"Generating Risk Map for {date}")
        print("="*70)
        
        # Step 1: GNN susceptibility (computed once)
        if not hasattr(self, 'gnn_susceptibility'):
            self.gnn_susceptibility = self._compute_gnn_susceptibility()
        
        # Step 2: Load dynamic rasters
        dynamic_stack = self._load_dynamic_rasters(date)
        
        # Step 3: Combine GNN + dynamic
        multi_channel_input = np.vstack([
            self.gnn_susceptibility[np.newaxis, :, :],
            dynamic_stack
        ])  # (1+C, H, W)
        
        # Step 4: Predict with sliding window
        risk_map = self._predict_with_sliding_window(multi_channel_input)
        
        # Step 4.5: Mask NoData regions (zone_raster == 0)
        print("Masking NoData regions...")
        valid_mask = (self.slope_id_raster > 0)
        risk_map[~valid_mask] = -9999  # Set NoData to -9999
        
        n_valid = valid_mask.sum()
        n_total = valid_mask.size
        print(f"  Valid pixels: {n_valid:,} ({n_valid/n_total*100:.1f}%)")
        print(f"  NoData pixels: {n_total - n_valid:,} ({(n_total-n_valid)/n_total*100:.1f}%)")
        
        # Step 5: Save risk map
        print(f"Saving risk map to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=-9999
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(risk_map.astype(np.float32), 1)
        
        print(f"✓ Risk map saved!")
        
        # Optional: Save GNN susceptibility
        if save_gnn_susceptibility:
            gnn_output_path = output_path.replace('.tif', '_gnn_susceptibility.tif')
            print(f"Saving GNN susceptibility to {gnn_output_path}...")
            
            with rasterio.open(gnn_output_path, 'w', **profile) as dst:
                dst.write(self.gnn_susceptibility.astype(np.float32), 1)
            
            print(f"✓ GNN susceptibility saved!")
        
        print("="*70)
        
        return risk_map


def main():
    parser = argparse.ArgumentParser(description='Generate landslide risk maps using trained GNN-U-Net model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--date', type=str,
                        help='Single date for prediction (YYYYMMDD)')
    parser.add_argument('--dates', type=str, nargs='+',
                        help='Multiple dates for prediction (YYYYMMDD YYYYMMDD ...)')
    parser.add_argument('--output', type=str,
                        help='Output path for single date prediction')
    parser.add_argument('--output_dir', type=str, default='outputs/risk_maps',
                        help='Output directory for multiple date predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Patch size for sliding window')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between patches')
    parser.add_argument('--save_gnn', action='store_true',
                        help='Save GNN susceptibility map separately')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.date and not args.dates:
        parser.error("Either --date or --dates must be specified")
    
    if args.date and not args.output:
        parser.error("--output must be specified when using --date")
    
    # Initialize predictor
    predictor = UNetPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        patch_size=args.patch_size,
        overlap=args.overlap
    )
    
    # Predict
    if args.date:
        # Single date
        predictor.predict(
            date=args.date,
            output_path=args.output,
            save_gnn_susceptibility=args.save_gnn
        )
    else:
        # Multiple dates
        os.makedirs(args.output_dir, exist_ok=True)
        
        for date in args.dates:
            output_path = os.path.join(args.output_dir, f"risk_map_{date}.tif")
            predictor.predict(
                date=date,
                output_path=output_path,
                save_gnn_susceptibility=args.save_gnn
            )
    
    print("\n" + "="*70)
    print("All predictions completed!")
    print("="*70)


if __name__ == "__main__":
    main()

