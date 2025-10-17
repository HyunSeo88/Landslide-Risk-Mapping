"""
GNN Raster Generation Script

Generates a full-coverage GNN susceptibility raster from a trained Stage 1 model.
This raster will be used as input to Stage 2 U-Net training.

Usage:
    python src/models/generate_gnn_raster.py \
        --checkpoint experiments/stage1_gnn/<timestamp>/checkpoints/model_stage1_best.pth \
        --graph_path data/model_ready/graph_data_v3.pt \
        --slope_id_raster data/model_ready/zone_raster_gyeongnam_5179_30m.tif \
        --output experiments/stage1_gnn/<timestamp>/gnn_susceptibility_30m.tif

Author: Landslide Risk Analysis Project
Date: 2025-01-17
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import rasterio
from rasterio.transform import Affine

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import StaticSusceptibilityGNN


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: AUC-ROC={checkpoint['metrics']['auc_roc']:.4f}, "
          f"F1={checkpoint['metrics']['f1']:.4f}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict, static_dim: int, device: torch.device) -> StaticSusceptibilityGNN:
    """Build model from checkpoint"""
    config = checkpoint['config']
    
    model = StaticSusceptibilityGNN(
        in_channels=static_dim,
        hidden_channels=config['model_gnn']['gnn_hidden'],
        num_layers=config['model_gnn']['gnn_layers'],
        gnn_type=config['model_gnn']['gnn_type'],
        dropout=config['model_gnn']['gnn_dropout'],
        gat_heads=config['model_gnn']['gat_heads']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_graph_data(graph_path: str, device: torch.device):
    """Load graph data"""
    print(f"Loading graph data from {graph_path}...")
    graph_data = torch.load(graph_path, weights_only=False)
    
    static_x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None
    cat_values = graph_data.cat.numpy()
    
    # Create cat to node mapping
    cat_to_node = {int(cat): idx for idx, cat in enumerate(cat_values)}
    
    print(f"  Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    print(f"  Static features: {static_x.shape}")
    
    return static_x, edge_index, edge_attr, cat_to_node


def load_slope_id_raster(slope_id_raster_path: str):
    """Load slope ID raster"""
    print(f"Loading slope ID raster from {slope_id_raster_path}...")
    
    with rasterio.open(slope_id_raster_path) as src:
        slope_id_raster = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        height, width = src.shape
    
    print(f"  Shape: {height} × {width}")
    print(f"  CRS: {crs}")
    print(f"  Unique slopes: {len(np.unique(slope_id_raster[slope_id_raster > 0]))}")
    
    return slope_id_raster, profile, transform, crs


def generate_gnn_raster(
    model: StaticSusceptibilityGNN,
    static_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    slope_id_raster: np.ndarray,
    cat_to_node: Dict[int, int],
    device: torch.device
) -> np.ndarray:
    """
    Generate GNN susceptibility raster
    
    Args:
        model: Trained GNN model
        static_x: Static features (num_nodes, static_dim)
        edge_index: Graph connectivity
        edge_attr: Edge weights
        slope_id_raster: Slope ID raster (H, W)
        cat_to_node: Mapping from slope ID to node index
        device: Device
    
    Returns:
        gnn_raster: GNN susceptibility raster (H, W)
    """
    print("\nGenerating GNN susceptibility raster...")
    
    # GNN forward pass (entire graph, once)
    print("  Running GNN forward pass...")
    with torch.no_grad():
        gnn_logits = model(static_x, edge_index, edge_attr)  # (num_nodes,)
        gnn_probs = torch.sigmoid(gnn_logits).cpu().numpy()  # (num_nodes,)
    
    print(f"  GNN probabilities: min={gnn_probs.min():.4f}, "
          f"max={gnn_probs.max():.4f}, mean={gnn_probs.mean():.4f}")
    
    # Create output raster
    print("  Mapping GNN values to raster pixels...")
    H, W = slope_id_raster.shape
    gnn_raster = np.full((H, W), -9999, dtype=np.float32)
    
    # Get unique slope IDs
    unique_cats = np.unique(slope_id_raster)
    valid_cats = unique_cats[unique_cats > 0]  # Exclude NoData (0)
    
    print(f"  Processing {len(valid_cats)} slopes...")
    
    # Map GNN values to pixels
    mapped_count = 0
    unmapped_count = 0
    
    for cat in valid_cats:
        cat_int = int(cat)
        
        if cat_int in cat_to_node:
            node_idx = cat_to_node[cat_int]
            gnn_value = gnn_probs[node_idx]
            
            # Assign value to all pixels of this slope
            gnn_raster[slope_id_raster == cat] = gnn_value
            mapped_count += 1
        else:
            # Slope not in graph (should not happen if data is consistent)
            unmapped_count += 1
    
    print(f"  Mapped slopes: {mapped_count}")
    if unmapped_count > 0:
        print(f"  Warning: {unmapped_count} slopes not found in graph")
    
    # Calculate statistics
    valid_mask = (gnn_raster != -9999)
    valid_pixels = valid_mask.sum()
    total_pixels = H * W
    
    print(f"  Valid pixels: {valid_pixels:,} / {total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
    print(f"  Raster statistics:")
    print(f"    Min: {gnn_raster[valid_mask].min():.4f}")
    print(f"    Max: {gnn_raster[valid_mask].max():.4f}")
    print(f"    Mean: {gnn_raster[valid_mask].mean():.4f}")
    print(f"    Std: {gnn_raster[valid_mask].std():.4f}")
    
    return gnn_raster


def save_geotiff(
    raster: np.ndarray,
    output_path: str,
    profile: Dict,
    nodata: float = -9999
):
    """Save raster as GeoTIFF"""
    print(f"\nSaving GeoTIFF to {output_path}...")
    
    # Update profile
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=nodata
    )
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write raster
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(raster.astype(np.float32), 1)
    
    print(f"  ✓ GeoTIFF saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate GNN Susceptibility Raster")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint (model_stage1_best.pth)')
    parser.add_argument('--graph_path', type=str, 
                       default='data/model_ready/graph_data_v3.pt',
                       help='Path to graph data')
    parser.add_argument('--slope_id_raster', type=str,
                       default='data/model_ready/zone_raster_gyeongnam_5179_30m.tif',
                       help='Path to slope ID raster')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for GNN raster')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("GNN Susceptibility Raster Generation")
    print("="*70)
    
    # Load data
    static_x, edge_index, edge_attr, cat_to_node = load_graph_data(args.graph_path, device)
    slope_id_raster, profile, transform, crs = load_slope_id_raster(args.slope_id_raster)
    
    # Load checkpoint and build model
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, static_x.shape[1], device)
    
    # Generate raster
    gnn_raster = generate_gnn_raster(
        model=model,
        static_x=static_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        slope_id_raster=slope_id_raster,
        cat_to_node=cat_to_node,
        device=device
    )
    
    # Save raster
    save_geotiff(gnn_raster, args.output, profile, nodata=-9999)
    
    print("\n" + "="*70)
    print("GNN Raster Generation Complete!")
    print(f"Output: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()

