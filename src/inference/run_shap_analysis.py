"""
SHAP Analysis Runner for Landslide Risk Model

This script performs comprehensive SHAP analysis on a trained model:
1. Dynamic feature importance (temporal patterns)
2. Static feature importance (spatial characteristics)
3. Integrated analysis and reporting

Usage:
    python src/inference/run_shap_analysis.py --checkpoint path/to/model_best.pth

Author: Landslide Risk Analysis Project
Date: 2025-01-16
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import LandslideRiskModel
from src.models.data_loader import LandslideDataset
from src.models.shap_analysis import run_full_shap_analysis


def main(args):
    """Main SHAP analysis pipeline"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Landslide Risk Model - SHAP Analysis")
    print("="*70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create dataset
    print(f"\nLoading dataset...")
    dataset = LandslideDataset(
        graph_path=config['data']['graph_path'],
        rainfall_path=config['data']['rainfall_path'],
        samples_path=config['data']['samples_path'],
        window_size=config['data']['window_size'],
        use_insar=config['data']['use_insar'],
        use_ndvi=config['data']['use_ndvi'],
        insar_path=config['data'].get('insar_path'),
        ndvi_path=config['data'].get('ndvi_path'),
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Build model
    print(f"\nBuilding model...")
    model = LandslideRiskModel(
        static_dim=dataset.static_features.shape[1],
        dynamic_dim=dataset.dynamic_dim,
        gnn_type=config['model']['gnn_type'],
        gnn_hidden=config['model']['gnn_hidden'],
        gnn_layers=config['model']['gnn_layers'],
        rnn_hidden=config['model']['rnn_hidden'],
        rnn_layers=config['model']['rnn_layers'],
        dropout=config['model']['dropout'],
        gat_heads=config['model']['gat_heads']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # Note: Don't call eval() here - SHAP analysis needs train mode for gradient computation
    # The SHAP analyzer will handle train/eval mode switching
    
    print(f"Model loaded successfully!")
    
    # Select samples for SHAP analysis
    total_samples = len(dataset)
    
    # Test samples: random subset
    np.random.seed(args.seed)
    test_indices = np.random.choice(total_samples, 
                                   size=min(args.n_test, total_samples), 
                                   replace=False).tolist()
    
    # Background samples: random subset (different from test)
    remaining = list(set(range(total_samples)) - set(test_indices))
    background_indices = np.random.choice(remaining,
                                         size=min(args.n_background, len(remaining)),
                                         replace=False).tolist()
    
    print(f"\nSHAP Analysis Configuration:")
    print(f"  Test samples: {len(test_indices)}")
    print(f"  Background samples: {len(background_indices)}")
    print(f"  Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run SHAP analysis
    results = run_full_shap_analysis(
        model=model,
        dataset=dataset,
        test_indices=test_indices,
        background_indices=background_indices,
        save_dir=output_dir,
        device=device
    )
    
    print("\n" + "="*70)
    print("SHAP Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    # Print summary
    print("\nGenerated files:")
    print("  - shap_dynamic_summary.png: Dynamic feature SHAP summary")
    print("  - shap_dynamic_importance.png: Dynamic feature importance (aggregated)")
    print("  - shap_temporal_importance.png: Temporal importance pattern")
    print("  - shap_static_summary.png: Static feature SHAP summary")
    print("  - shap_static_importance.png: Static feature importance")
    print("  - shap_integrated_comparison.png: Integrated comparison")
    print("  - shap_report.txt: Detailed text report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Analysis for Landslide Risk Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='outputs/shap_analysis',
                       help='Output directory for SHAP results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--n_test', type=int, default=100,
                       help='Number of test samples for SHAP')
    parser.add_argument('--n_background', type=int, default=100,
                       help='Number of background samples for SHAP')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    
    args = parser.parse_args()
    main(args)

