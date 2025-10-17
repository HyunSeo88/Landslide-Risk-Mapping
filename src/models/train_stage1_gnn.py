"""
Stage 1: GNN Pre-training Script for Static Susceptibility

This script trains the GNN component of the HierarchicalGNNUNet model
using slope-level labels with standard BCE loss (not MIL).

Key features:
- Trains GNN only (U-Net is not used)
- Uses BCEWithLogitsLoss for slope-level binary classification
- Saves best model checkpoint for Stage 2 initialization

Usage:
    python src/models/train_stage1_gnn.py --config configs/stage1_gnn.yaml

Author: Landslide Risk Analysis Project
Date: 2025-01-17
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import StaticSusceptibilityGNN


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(save_dir: str) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(save_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    return exp_dir


# ============================================================
# Dataset for Stage 1 (Slope-level)
# ============================================================

class SlopeLevelDataset(Dataset):
    """
    Slope-level dataset for Stage 1 GNN training
    
    Each sample is a slope with its static features and label.
    Multiple temporal samples for the same slope are aggregated.
    
    Args:
        graph_path: Path to graph_data.pt
        samples_path: Path to samples CSV (cat, event_date, label)
        start_date: Start date filter
        end_date: End date filter
    """
    
    def __init__(
        self,
        graph_path: str,
        samples_path: str,
        start_date: str = '20190101',
        end_date: str = '20200930'
    ):
        super().__init__()
        
        # Load graph
        print(f"Loading graph from {graph_path}...")
        self.graph_data = torch.load(graph_path, weights_only=False)
        self.static_features = self.graph_data.x
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr
        self.cat_values = self.graph_data.cat
        
        # Create mappings
        self.cat_to_node = {int(cat): idx for idx, cat in enumerate(self.cat_values.numpy())}
        self.node_to_cat = {idx: int(cat) for cat, idx in self.cat_to_node.items()}
        
        print(f"  Graph: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        
        # Load and process samples
        print(f"Loading samples from {samples_path}...")
        samples_df = pd.read_csv(samples_path, encoding='utf-8-sig')
        samples_df['event_date'] = pd.to_datetime(samples_df['event_date'], format='%Y-%m-%d')
        
        # Date filtering
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')
        samples_df = samples_df[
            (samples_df['event_date'] >= start_dt) &
            (samples_df['event_date'] <= end_dt)
        ]
        
        # Filter by valid cat
        samples_df = samples_df[samples_df['cat'].isin(self.cat_to_node.keys())]
        
        # Aggregate by slope: if any positive label exists, mark as positive
        slope_labels = {}
        for _, row in samples_df.iterrows():
            cat = int(row['cat'])
            label = int(row['label'])
            
            if cat not in slope_labels:
                slope_labels[cat] = label
            else:
                # If any sample is positive, mark slope as positive
                slope_labels[cat] = max(slope_labels[cat], label)
        
        # Create sample list
        self.samples = []
        for cat, label in slope_labels.items():
            node_idx = self.cat_to_node[cat]
            self.samples.append({
                'cat': cat,
                'node_idx': node_idx,
                'label': label
            })
        
        print(f"  Valid slopes: {len(self.samples)}")
        print(f"  Positive: {sum(s['label'] == 1 for s in self.samples)}")
        print(f"  Negative: {sum(s['label'] == 0 for s in self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single slope sample"""
        sample = self.samples[idx]
        return {
            'node_idx': torch.tensor(sample['node_idx'], dtype=torch.long),
            'cat': torch.tensor(sample['cat'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float32)
        }


class Stage1Collator:
    """Collator for Stage 1 slope-level batches"""
    
    def __init__(self, graph_data):
        self.graph_data = graph_data
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch"""
        node_indices = torch.stack([s['node_idx'] for s in batch])
        cats = torch.stack([s['cat'] for s in batch])
        labels = torch.stack([s['label'] for s in batch])
        
        return {
            'static_x': self.graph_data.x,
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr,
            'node_indices': node_indices,
            'cats': cats,
            'labels': labels
        }


def create_stage1_dataloaders(
    graph_path: str,
    samples_path: str,
    start_date: str,
    end_date: str,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, SlopeLevelDataset]:
    """Create train and validation data loaders for Stage 1"""
    
    # Create dataset
    dataset = SlopeLevelDataset(
        graph_path=graph_path,
        samples_path=samples_path,
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
    print(f"  Training: {train_size} slopes")
    print(f"  Validation: {val_size} slopes")
    
    # Create collator
    collator = Stage1Collator(dataset.graph_data)
    
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
# Training Functions
# ============================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
    log_interval: int = 10
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp and scaler is not None:
            with autocast('cuda'):
                # GNN forward on entire graph
                gnn_logits = model(
                    batch['static_x'],
                    batch['edge_index'],
                    batch.get('edge_attr')
                )  # (num_nodes,)
                
                # Extract logits for batch samples
                batch_logits = gnn_logits[batch['node_indices']]  # (B,)
                
                # BCE loss
                loss = criterion(batch_logits, batch['labels'])
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            gnn_logits = model(
                batch['static_x'],
                batch['edge_index'],
                batch.get('edge_attr')
            )
            
            batch_logits = gnn_logits[batch['node_indices']]
            loss = criterion(batch_logits, batch['labels'])
            
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # GNN forward
            gnn_logits = model(
                batch['static_x'],
                batch['edge_index'],
                batch.get('edge_attr')
            )
            
            batch_logits = gnn_logits[batch['node_indices']]
            loss = criterion(batch_logits, batch['labels'])
            total_loss += loss.item()
            
            # Predictions
            batch_probs = torch.sigmoid(batch_logits).cpu().numpy()
            batch_preds = (batch_probs > 0.5).astype(float)
            batch_labels = batch['labels'].cpu().numpy()
            
            all_probs.extend(batch_probs)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pr': average_precision_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics


# ============================================================
# Checkpointing and Plotting
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    config: Dict,
    exp_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    if is_best:
        checkpoint_path = exp_dir / "checkpoints" / "model_stage1_best.pth"
        print(f"  → Saving best model: {checkpoint_path}")
    else:
        checkpoint_path = exp_dir / "checkpoints" / "model_stage1_final.pth"
    
    torch.save(checkpoint, checkpoint_path)


def plot_training_curves(history: Dict, exp_dir: Path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC-ROC curve
    axes[0, 1].plot(history['val_auc_roc'], label='Val AUC-ROC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('AUC-ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision-Recall
    axes[1, 1].plot(history['val_precision'], label='Precision')
    axes[1, 1].plot(history['val_recall'], label='Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(exp_dir / "plots" / "training_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, exp_dir: Path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(exp_dir / "plots" / "confusion_matrix.png", dpi=300)
    plt.close()


# ============================================================
# Main Training Loop
# ============================================================

def train(config: Dict, exp_dir: Path):
    """Main training function"""
    
    device = torch.device(config['experiment']['device'])
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, dataset = create_stage1_dataloaders(
        graph_path=config['data']['graph_path'],
        samples_path=config['data']['samples_path'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        batch_size=config['training_stage1']['batch_size'],
        train_ratio=1.0 - config['training_stage1']['val_ratio'],
        random_seed=config['experiment']['seed'],
        num_workers=0
    )
    
    # Build model
    print("\nBuilding GNN model...")
    model = StaticSusceptibilityGNN(
        in_channels=dataset.static_features.shape[1],
        hidden_channels=config['model_gnn']['gnn_hidden'],
        num_layers=config['model_gnn']['gnn_layers'],
        gnn_type=config['model_gnn']['gnn_type'],
        dropout=config['model_gnn']['gnn_dropout'],
        gat_heads=config['model_gnn']['gat_heads']
    )
    model = model.to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training_stage1']['learning_rate'],
        weight_decay=config['training_stage1']['weight_decay']
    )
    
    # Loss function
    pos_weight = config['training_stage1'].get('pos_weight')
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight})")
    
    # Scheduler
    scheduler = None
    if config['training_stage1'].get('use_scheduler', False):
        scheduler_type = config['training_stage1'].get('scheduler_type', 'warmup_cosine')
        
        if scheduler_type == 'warmup_cosine':
            warmup_epochs = config['training_stage1'].get('warmup_epochs', 10)
            total_epochs = config['training_stage1']['epochs']
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=config['training_stage1'].get('min_lr', 1e-7)
            )
            
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            
            print(f"  Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing")
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config['training_stage1'].get('use_amp', False) else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc_roc': [],
        'val_auc_pr': []
    }
    
    # Early stopping
    best_metric = 0.0
    patience_counter = 0
    monitor_metric = config['training_stage1']['monitor_metric']
    
    epochs = config['training_stage1']['epochs']
    
    print("\n" + "="*70)
    print("Starting Stage 1 Training (GNN Pre-training)")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            use_amp=config['training_stage1'].get('use_amp', False),
            grad_clip=config['training_stage1'].get('grad_clip_value'),
            log_interval=config['training_stage1'].get('log_interval', 10)
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Log
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics:")
        print(f"    Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"    Precision: {val_metrics['precision']:.4f}")
        print(f"    Recall:    {val_metrics['recall']:.4f}")
        print(f"    F1:        {val_metrics['f1']:.4f}")
        print(f"    AUC-ROC:   {val_metrics['auc_roc']:.4f}")
        print(f"    AUC-PR:    {val_metrics['auc_pr']:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_auc_pr'].append(val_metrics['auc_pr'])
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Check best model
        current_metric = val_metrics[monitor_metric.replace('val_', '')]
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(model, optimizer, epoch, val_metrics, config, exp_dir, is_best=True)
            patience_counter = 0
            print(f"  ✓ New best {monitor_metric}: {best_metric:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training_stage1']['early_stopping_patience']:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    save_checkpoint(model, optimizer, epochs-1, val_metrics, config, exp_dir, is_best=False)
    
    # Plot curves
    plot_training_curves(history, exp_dir)
    
    # Plot confusion matrix
    checkpoint = torch.load(exp_dir / "checkpoints" / "model_stage1_best.pth", weights_only=False)
    cm = checkpoint['metrics']['confusion_matrix']
    plot_confusion_matrix(np.array(cm), exp_dir)
    
    # Save history
    with open(exp_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Stage 1 Training Complete!")
    print(f"Best {monitor_metric}: {best_metric:.4f}")
    print("="*70)
    
    return history


# ============================================================
# Main Function
# ============================================================

def main(config_path: str):
    """Main training pipeline"""
    # Load config
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config['experiment']['save_dir'])
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    shutil.copy(config_path, exp_dir / "config.yaml")
    
    # Train
    results = train(config, exp_dir)
    
    print("\n" + "="*70)
    print("Training Completed!")
    print(f"Results saved to: {exp_dir}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 GNN Model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args.config)

