"""
Inference Script for Landslide Risk Model

This script loads a trained model and generates predictions for:
1. Risk maps (spatial predictions)
2. Time series predictions
3. Evaluation reports

"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import LandslideRiskModel
from src.models.data_loader import LandslideDataset, LandslideCollator
from src.utils.metrics import compute_metrics


def load_model_and_config(checkpoint_path: str, device: str = 'cuda') -> Tuple[LandslideRiskModel, Dict]:
    """
    Load trained model and configuration
    
    Returns:
        model: Loaded model
        config: Configuration dictionary
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create dataset to get dimensions
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
        end_date=config['data']['end_date'],
        include_target_day=True  # For training compatibility
    )
    
    # Build model
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
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: AUC-ROC={checkpoint['metrics'].get('auc_roc', 'N/A'):.4f}")
    
    return model, config, dataset


def generate_predictions(
    model: LandslideRiskModel,
    dataset: LandslideDataset,
    device: str = 'cuda',
    batch_size: int = 256,
    include_target_day: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions for all samples
    
    Args:
        include_target_day: If False, use real-time prediction mode (exclude target day)
    
    Returns:
        predictions: Probability predictions
        labels: True labels
        cats: Slope unit IDs
        attentions: Attention weights (N, 2)
    """
    print(f"\nGenerating predictions...")
    print(f"  Mode: {'Training (include target day)' if include_target_day else 'Real-time (exclude target day)'}")
    
    # Update dataset mode
    dataset.include_target_day = include_target_day
    
    # Create dataloader
    collator = LandslideCollator(dataset.graph_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    
    all_preds = []
    all_labels = []
    all_cats = []
    all_attentions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch, return_attention=True)
            logits = outputs['logits']
            attention = outputs['attention']
            
            # Predictions
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_cats.extend(batch['cats'].cpu().numpy())
            all_attentions.append(attention.cpu().numpy())
    
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    cats = np.array(all_cats)
    attentions = np.vstack(all_attentions)
    
    print(f"  Predictions generated: {len(predictions)} samples")
    
    return predictions, labels, cats, attentions


def save_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    cats: np.ndarray,
    attentions: np.ndarray,
    dataset: LandslideDataset,
    output_path: Path
):
    """Save predictions to CSV"""
    # Get event dates
    event_dates = [dataset.samples[i]['event_date'].strftime('%Y-%m-%d') 
                   for i in range(len(dataset))]
    
    df = pd.DataFrame({
        'cat': cats,
        'event_date': event_dates,
        'true_label': labels,
        'predicted_prob': predictions,
        'predicted_label': (predictions > 0.5).astype(int),
        'attention_gnn': attentions[:, 0],
        'attention_rnn': attentions[:, 1]
    })
    
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved: {output_path}")


def generate_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path
):
    """Generate evaluation metrics report"""
    from sklearn.metrics import classification_report, roc_curve, auc
    
    # Compute metrics (need probabilities for AUC)
    metrics = compute_metrics(
        torch.tensor(predictions),  # Use probabilities, not binary
        torch.tensor(labels),
        threshold=0.5
    )
    
    # Binary predictions for classification report
    preds_binary = (predictions > 0.5).astype(int)
    
    # Classification report
    class_report = classification_report(labels, preds_binary, 
                                        target_names=['Stable', 'Landslide'])
    
    # Save report
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Landslide Risk Model - Evaluation Report\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC:   {metrics['auc']:.4f}\n")
        f.write(f"AUC-PR:    {metrics['ap']:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-"*70 + "\n")
        f.write(class_report + "\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*70 + "\n")
        f.write(f"              Predicted\n")
        f.write(f"              Stable  Landslide\n")
        f.write(f"Actual Stable    {metrics['true_negative']:6d}  {metrics['false_positive']:6d}\n")
        f.write(f"       Landslide {metrics['false_negative']:6d}  {metrics['true_positive']:6d}\n")
    
    print(f"Evaluation report saved: {output_path}")


def plot_roc_curve(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROC curve saved: {output_path}")


def plot_risk_distribution(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Plot risk score distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(predictions[labels == 0], bins=50, alpha=0.5, label='Stable', color='blue')
    axes[0].hist(predictions[labels == 1], bins=50, alpha=0.5, label='Landslide', color='red')
    axes[0].set_xlabel('Predicted Risk Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Risk Score Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [predictions[labels == 0], predictions[labels == 1]]
    axes[1].boxplot(data_to_plot, labels=['Stable', 'Landslide'])
    axes[1].set_ylabel('Predicted Risk Probability')
    axes[1].set_title('Risk Score by Class')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Risk distribution plot saved: {output_path}")


def main(args):
    """Main inference pipeline"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    predictions_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Landslide Risk Model - Inference")
    print("="*70)
    
    # Load model
    model, config, dataset = load_model_and_config(args.checkpoint, device)
    
    # Generate predictions
    predictions, labels, cats, attentions = generate_predictions(
        model, dataset, device, 
        batch_size=args.batch_size,
        include_target_day=not args.realtime_mode
    )
    
    # Save predictions
    pred_filename = "predictions_realtime.csv" if args.realtime_mode else "predictions_training.csv"
    save_predictions(
        predictions, labels, cats, attentions, dataset,
        predictions_dir / pred_filename
    )
    
    # Generate evaluation report
    report_filename = "evaluation_report_realtime.txt" if args.realtime_mode else "evaluation_report_training.txt"
    generate_evaluation_report(
        predictions, labels,
        reports_dir / report_filename
    )
    
    # Plot ROC curve
    roc_filename = "roc_curve_realtime.png" if args.realtime_mode else "roc_curve_training.png"
    plot_roc_curve(predictions, labels, reports_dir / roc_filename)
    
    # Plot risk distribution
    dist_filename = "risk_distribution_realtime.png" if args.realtime_mode else "risk_distribution_training.png"
    plot_risk_distribution(predictions, labels, reports_dir / dist_filename)
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landslide Risk Model Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--realtime_mode', action='store_true',
                       help='Use real-time prediction mode (exclude target day)')
    
    args = parser.parse_args()
    main(args)

