"""
Stage 2: Dual-Stream U-Net Training Script

This script trains the Dual-Stream Hierarchical GNN-U-Net model:
- State Encoder (CNN) for static terrain features
- Trigger Encoder (ConvLSTM) for temporal dynamic sequences
- Cross-Attention Fusion for State-Trigger interaction
- Attention-MIL for interpretable weakly supervised learning

Key Features:
- Loads Stage 1 GNN weights and freezes them
- Temporal window (5 days) for ConvLSTM
- Attention maps for interpretability
- Comprehensive evaluation metrics

Usage:
    python src/models/train_stage2_dual.py --config configs/stage2_dual_unet.yaml

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
from typing import Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_dual_unet import DualHierarchicalGNNUNet
from src.models.data_loader_dual import create_dual_dataloaders


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
    (exp_dir / "attention_maps").mkdir(exist_ok=True)
    
    return exp_dir


# ============================================================
# Training Functions
# ============================================================

def train_epoch_dual(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
    grad_clip: float = None,
    log_interval: int = 5
) -> float:
    """
    Train for one epoch with dual-stream architecture
    
    Returns:
        avg_loss: Average training loss
    """
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
                # Dual-stream forward
                pixel_logits, intermediate_features = model(
                    gnn_patch=batch['gnn_patches'],
                    state_patch=batch['state_patches'],
                    trigger_sequence=batch['trigger_sequences']
                )
                
                # Attention-MIL loss
                loss, attention_maps = model.compute_loss(
                    pixel_logits,
                    intermediate_features,
                    batch['zone_patches'],
                    batch['cats'],
                    batch['labels']
                )
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            pixel_logits, intermediate_features = model(
                gnn_patch=batch['gnn_patches'],
                state_patch=batch['state_patches'],
                trigger_sequence=batch['trigger_sequences']
            )
            
            loss, attention_maps = model.compute_loss(
                pixel_logits,
                intermediate_features,
                batch['zone_patches'],
                batch['cats'],
                batch['labels']
            )
            
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


def validate_epoch_dual(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict]:
    """
    Validate for one epoch
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # Forward
            pixel_logits, intermediate_features = model(
                gnn_patch=batch['gnn_patches'],
                state_patch=batch['state_patches'],
                trigger_sequence=batch['trigger_sequences']
            )
            
            # Loss
            loss, attention_maps = model.compute_loss(
                pixel_logits,
                intermediate_features,
                batch['zone_patches'],
                batch['cats'],
                batch['labels']
            )
            total_loss += loss.item()
            
            # Extract representative scores
            batch_size = len(batch['cats'])
            for i in range(batch_size):
                cat = batch['cats'][i].item()
                zone_patch = batch['zone_patches'][i]
                attention_map = attention_maps[i]
                
                # Valid mask
                valid_mask = (zone_patch > 0).float()
                slope_mask = ((zone_patch == cat) & (zone_patch > 0)).float()
                
                if slope_mask.sum() == 0:
                    representative_score = 0.0
                else:
                    # Weighted aggregation with attention
                    slope_logits = pixel_logits[i, 0] * slope_mask
                    slope_attention = attention_map * slope_mask
                    
                    # Normalize attention
                    attention_sum = slope_attention.sum()
                    if attention_sum > 0:
                        slope_attention = slope_attention / attention_sum
                    
                    representative_score = (slope_logits * slope_attention).sum().item()
                
                representative_prob = torch.sigmoid(torch.tensor(representative_score)).item()
                representative_pred = 1.0 if representative_prob > 0.5 else 0.0
                
                all_probs.append(representative_prob)
                all_preds.append(representative_pred)
                all_labels.append(batch['labels'][i].item())
    
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
        checkpoint_path = exp_dir / "checkpoints" / "model_dual_best.pth"
        print(f"  → Saving best model: {checkpoint_path}")
    else:
        checkpoint_path = exp_dir / "checkpoints" / "model_dual_final.pth"
    
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
    print("\nCreating data loaders (Dual-Stream)...")
    train_loader, val_loader, dataset = create_dual_dataloaders(
        graph_path=config['data']['graph_path'],
        samples_path=config['data']['samples_path'],
        slope_id_raster_path=config['data']['slope_id_raster_path'],
        slope_bboxes_cache_path=config['data']['slope_bboxes_cache_path'],
        gnn_raster_path=config['data']['gnn_raster_path'],
        static_raster_dir=config['data']['static_raster_dir'],
        static_variables=config['data']['static_variables'],
        dynamic_raster_base=config['data']['dynamic_raster_base'],
        dynamic_variables=config['data']['dynamic_variables'],
        patch_size=config['data']['patch_size'],
        window_size=config['data']['window_size'],
        batch_size=config['training_dual']['batch_size'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        train_ratio=1.0 - config['training_dual']['val_ratio'],
        random_seed=config['experiment']['seed'],
        num_workers=4  # ← 0에서 4로 변경
    )
    
    # Build model
    print("\nBuilding Dual-Stream Hierarchical GNN-U-Net model...")
    
    static_dim = dataset.graph_data.x.shape[1]
    
    model = DualHierarchicalGNNUNet(
        static_dim=static_dim,
        gnn_hidden=config['model_dual']['gnn_hidden'],
        gnn_layers=config['model_dual']['gnn_layers'],
        gnn_type=config['model_dual']['gnn_type'],
        gnn_dropout=config['model_dual']['gnn_dropout'],
        gat_heads=config['model_dual']['gat_heads'],
        state_channels=dataset.num_state_channels,
        trigger_channels=dataset.num_trigger_channels,
        state_base_channels=config['model_dual']['state_base_channels'],
        trigger_hidden_channels=config['model_dual']['trigger_hidden_channels'],
        trigger_num_layers=config['model_dual']['trigger_num_layers'],
        fusion_num_heads=config['model_dual']['fusion_num_heads'],
        decoder_depth=config['model_dual']['decoder_depth']
    )
    
    print(f"  State channels: {dataset.num_state_channels}")
    print(f"  Trigger channels: {dataset.num_trigger_channels}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load Stage 1 GNN weights
    stage1_checkpoint_path = config['stage2']['stage1_checkpoint_path']
    print(f"\nLoading Stage 1 GNN weights from:")
    print(f"  {stage1_checkpoint_path}")
    
    stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location=device, weights_only=False)
    
    # Load GNN weights
    model.gnn.load_state_dict(stage1_checkpoint['model_state_dict'], strict=True)
    print(f"  ✓ Stage 1 GNN weights loaded successfully")
    print(f"  Stage 1 metrics: AUC-ROC={stage1_checkpoint['metrics']['auc_roc']:.4f}, "
          f"F1={stage1_checkpoint['metrics']['f1']:.4f}")
    
    # Freeze GNN if requested
    if config['stage2'].get('freeze_gnn', True):
        model.freeze_gnn()
    
    model = model.to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    
    # Optimizer
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=config['training_dual']['learning_rate'],
        weight_decay=config['training_dual']['weight_decay']
    )
    
    # Scheduler
    scheduler = None
    if config['training_dual'].get('use_scheduler', False):
        scheduler_type = config['training_dual'].get('scheduler_type', 'warmup_cosine')
        
        if scheduler_type == 'warmup_cosine':
            warmup_epochs = config['training_dual'].get('warmup_epochs', 10)
            total_epochs = config['training_dual']['epochs']
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=config['training_dual'].get('min_lr', 1e-7)
            )
            
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            
            print(f"  Using Warmup + CosineAnnealing scheduler")
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config['training_dual'].get('use_amp', False) else None
    
    # Tensorboard
    writer = None
    if config['training_dual'].get('use_tensorboard', False):
        writer = SummaryWriter(log_dir=exp_dir / "logs")
    
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
    monitor_metric = config['training_dual']['monitor_metric']
    
    epochs = config['training_dual']['epochs']
    
    print("\n" + "="*70)
    print("Starting Stage 2 Training (Dual-Stream with ConvLSTM)")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch_dual(
            model, train_loader, optimizer, device,
            scaler=scaler,
            use_amp=config['training_dual'].get('use_amp', False),
            grad_clip=config['training_dual'].get('grad_clip_value'),
            log_interval=config['training_dual'].get('log_interval', 5)
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch_dual(
            model, val_loader, device
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
        
        # Tensorboard
        if writer is not None:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            writer.add_scalar('Val/AUC-ROC', val_metrics['auc_roc'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
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
        if patience_counter >= config['training_dual']['early_stopping_patience']:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    save_checkpoint(model, optimizer, epochs-1, val_metrics, config, exp_dir, is_best=False)
    
    # Close tensorboard
    if writer is not None:
        writer.close()
    
    # Plot curves
    plot_training_curves(history, exp_dir)
    
    # Plot confusion matrix
    checkpoint = torch.load(exp_dir / "checkpoints" / "model_dual_best.pth", weights_only=False)
    cm = checkpoint['metrics']['confusion_matrix']
    plot_confusion_matrix(np.array(cm), exp_dir)
    
    # Save history
    with open(exp_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Stage 2 Training Complete (Dual-Stream)!")
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
    parser = argparse.ArgumentParser(description="Train Stage 2 Dual-Stream U-Net Model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args.config)

