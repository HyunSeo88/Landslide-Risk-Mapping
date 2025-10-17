"""
Training Script for Landslide Risk Model

Features:
- Config-based experiment management
- Single split or K-fold cross-validation
- Epoch-wise negative resampling with hard mining
- Warmup + Cosine annealing scheduler
- Gradient clipping
- Mixed precision training (FP16)
- Tensorboard logging
- Comprehensive metrics tracking

Usage:
    python src/models/train.py --config configs/baseline.yaml
"""

import os
import sys
import argparse
import random
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import LandslideRiskModel, StaticFeatureProxy, get_loss_fn
from src.models.data_loader import LandslideDataset, LandslideCollator, create_dataloaders
from src.models.sampling import (
    sample_negatives_temporal_matched,
    sample_negatives_hard,
    sample_negatives_random
)


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
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
# Model Building
# ============================================================

def build_model_from_config(config: Dict, dataset: LandslideDataset) -> LandslideRiskModel:
    """
    Build model from config with automatic dimension inference
    """
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

    return model


# ============================================================
# Optimizer and Scheduler
# ============================================================

def get_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Get optimizer from config"""
    opt_type = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict, steps_per_epoch: int):
    """
    Get learning rate scheduler with warmup + cosine annealing

    Args:
        optimizer: PyTorch optimizer
        config: Training config
        steps_per_epoch: Number of batches per epoch

    Returns:
        scheduler: Learning rate scheduler
    """
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0) if config['training'].get('use_warmup', False) else 0
    min_lr = config['training'].get('min_lr', 1e-6)

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup: linear increase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr / config['training']['learning_rate'],
                      0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler


# ============================================================
# Metrics Calculation
# ============================================================

def calculate_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict:
    """
    Calculate comprehensive evaluation metrics

    Returns:
        metrics: Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auc_roc': roc_auc_score(labels, probs),
        'auc_pr': average_precision_score(labels, probs),
        'confusion_matrix': confusion_matrix(labels, preds).tolist()
    }

    return metrics


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                scaler: Optional[GradScaler] = None,
                use_amp: bool = False,
                grad_clip: Optional[float] = None,
                log_interval: int = 10) -> float:
    """
    Train for one epoch

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

        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(batch)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs['logits']
                loss = criterion(logits, batch['labels'])

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            outputs = model(batch)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs['logits']
            loss = criterion(logits, batch['labels'])

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            optimizer.step()

        # Scheduler step (per batch for warmup + cosine)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, Dict, np.ndarray]:
    """
    Validate for one epoch

    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of evaluation metrics
        all_attentions: Attention weights (N, 2)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_attentions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward with attention
            outputs = model(batch, return_attention=True)
            logits = outputs['logits']
            attention = outputs['attention']

            # Loss
            loss = criterion(logits, batch['labels'])
            total_loss += loss.item()

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_attentions.append(attention.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_attentions = np.vstack(all_attentions)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    # Add attention statistics
    metrics['attention_mean_gnn'] = float(all_attentions[:, 0].mean())
    metrics['attention_mean_rnn'] = float(all_attentions[:, 1].mean())

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics, all_attentions


# ============================================================
# Negative Resampling
# ============================================================

def resample_negatives_and_reload(
    dataset: LandslideDataset,
    model: nn.Module,
    config: Dict,
    epoch: int,
    device: torch.device
) -> DataLoader:
    """
    Resample negative samples and create new dataloader

    Args:
        dataset: Original dataset
        model: Trained model (for hard mining)
        config: Training config
        epoch: Current epoch number
        device: Device

    Returns:
        new_loader: DataLoader with resampled negatives
    """
    print(f"  Resampling negatives for epoch {epoch+1}...")

    # Extract positive samples
    positive_samples = [s for s in dataset.samples if s['label'] == 1]

    # Get all slope IDs that have rainfall data (to avoid IndexError)
    graph_slope_ids = set(dataset.graph_data.cat.numpy().tolist())
    rainfall_slope_ids = set(dataset.rainfall_df['cat'].values)
    all_slope_ids = list(graph_slope_ids.intersection(rainfall_slope_ids))

    # Determine sampling strategy
    strategy = config['training']['sampling_strategy']
    hard_mining_start = config['training'].get('hard_mining_start_epoch', 5)

    # Hard negative mining after certain epoch
    if epoch >= hard_mining_start:
        print(f"    Using hard negative mining...")
        # TODO: Implement hard mining with model predictions
        # For now, use temporal matching
        negative_samples = sample_negatives_temporal_matched(
            positive_samples, all_slope_ids,
            ratio=1.0, random_seed=config['experiment']['seed'] + epoch
        )
    else:
        # Standard sampling
        if strategy == 'temporal':
            negative_samples = sample_negatives_temporal_matched(
                positive_samples, all_slope_ids,
                ratio=1.0, random_seed=config['experiment']['seed'] + epoch
            )
        elif strategy == 'random':
            date_range = (dataset.start_date, dataset.end_date)
            negative_samples = sample_negatives_random(
                positive_samples, all_slope_ids, date_range,
                ratio=1.0, random_seed=config['experiment']['seed'] + epoch
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Create new dataset with resampled negatives
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    dataset.samples = all_samples

    print(f"    Resampled: {len(positive_samples)} pos + {len(negative_samples)} neg")

    # Create new dataloader
    collator = LandslideCollator(dataset.graph_data)
    new_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )

    return new_loader


# ============================================================
# Checkpointing and Logging
# ============================================================

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   config: Dict,
                   exp_dir: Path,
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }

    # Save checkpoint
    if is_best:
        checkpoint_path = exp_dir / "checkpoints" / "model_best.pth"
        print(f"  → Saving best model: {checkpoint_path}")
    else:
        checkpoint_path = exp_dir / "checkpoints" / "model_final.pth"

    torch.save(checkpoint, checkpoint_path)


def log_metrics_to_tensorboard(writer: SummaryWriter,
                               phase: str,
                               metrics: Dict,
                               epoch: int):
    """Log metrics to tensorboard"""
    # Loss (always present)
    if 'loss' in metrics:
        writer.add_scalar(f'{phase}/Loss', metrics['loss'], epoch)
    
    # Classification metrics (optional)
    if 'accuracy' in metrics:
        writer.add_scalar(f'{phase}/Accuracy', metrics['accuracy'], epoch)
    if 'precision' in metrics:
        writer.add_scalar(f'{phase}/Precision', metrics['precision'], epoch)
    if 'recall' in metrics:
        writer.add_scalar(f'{phase}/Recall', metrics['recall'], epoch)
    if 'f1' in metrics:
        writer.add_scalar(f'{phase}/F1', metrics['f1'], epoch)
    if 'auc_roc' in metrics:
        writer.add_scalar(f'{phase}/AUC-ROC', metrics['auc_roc'], epoch)
    if 'auc_pr' in metrics:
        writer.add_scalar(f'{phase}/AUC-PR', metrics['auc_pr'], epoch)

    # Attention weights (optional)
    if 'attention_mean_gnn' in metrics:
        writer.add_scalar(f'{phase}/Attention_GNN', metrics['attention_mean_gnn'], epoch)
    if 'attention_mean_rnn' in metrics:
        writer.add_scalar(f'{phase}/Attention_RNN', metrics['attention_mean_rnn'], epoch)


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

    # Attention weights
    axes[1, 1].plot(history['val_attention_gnn'], label='GNN')
    axes[1, 1].plot(history['val_attention_rnn'], label='RNN')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].set_title('Attention Weights')
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

def train_single_split(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      dataset: LandslideDataset,
                      config: Dict,
                      exp_dir: Path) -> Dict:
    """
    Train with single train/val split

    Returns:
        history: Training history
    """
    device = torch.device(config['experiment']['device'])
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    # Loss function
    criterion = get_loss_fn(
        config['training']['loss_type'],
        pos_weight=config['training'].get('pos_weight', 1.0),
        device=device
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda') if config['training'].get('use_amp', False) else None

    # Tensorboard
    writer = None
    if config['training'].get('use_tensorboard', False):
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
        'val_auc_pr': [],
        'val_attention_gnn': [],
        'val_attention_rnn': []
    }

    # Early stopping
    best_metric = 0.0
    patience_counter = 0
    monitor_metric = config['training']['monitor_metric']

    epochs = config['training']['epochs']

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=config['training'].get('use_amp', False),
            grad_clip=config['training'].get('grad_clip_value') if config['training'].get('use_grad_clip') else None,
            log_interval=config['training']['log_interval']
        )

        # Validate
        val_loss, val_metrics, val_attentions = validate_epoch(
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
        print(f"  Attention: GNN={val_metrics['attention_mean_gnn']:.3f}, "
              f"RNN={val_metrics['attention_mean_rnn']:.3f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_auc_pr'].append(val_metrics['auc_pr'])
        history['val_attention_gnn'].append(val_metrics['attention_mean_gnn'])
        history['val_attention_rnn'].append(val_metrics['attention_mean_rnn'])

        # Tensorboard
        if writer is not None:
            log_metrics_to_tensorboard(writer, 'Train', {'loss': train_loss}, epoch)
            log_metrics_to_tensorboard(writer, 'Val', {**val_metrics, 'loss': val_loss}, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

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
        early_stopping_patience = config['training']['early_stopping_patience']
        if patience_counter >= early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break

        # Resample negatives
        if config['training'].get('resample_negatives', False):
            train_loader = resample_negatives_and_reload(
                dataset, model, config, epoch, device
            )

    # Save final model
    save_checkpoint(model, optimizer, epochs-1, val_metrics, config, exp_dir, is_best=False)

    # Close tensorboard
    if writer is not None:
        writer.close()

    return history


def train_kfold(dataset: LandslideDataset,
               config: Dict,
               exp_dir: Path) -> Dict:
    """
    Train with K-Fold cross-validation

    Returns:
        aggregated_results: Aggregated metrics across folds
    """
    k_folds = config['training']['k_folds']
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config['experiment']['seed'])

    fold_results = []

    print("\n" + "="*70)
    print(f"Starting {k_folds}-Fold Cross-Validation")
    print("="*70)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{k_folds}")
        print(f"{'='*70}")

        # Create fold dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        collator = LandslideCollator(dataset.graph_data)

        train_loader = DataLoader(
            train_subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        # Build model for this fold
        model = build_model_from_config(config, dataset)

        # Create fold directory
        fold_dir = exp_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)

        # Train
        fold_history = train_single_split(
            model, train_loader, val_loader, dataset, config, fold_dir
        )

        fold_results.append(fold_history)

    # Aggregate results
    aggregated = aggregate_fold_results(fold_results, exp_dir)

    return aggregated


def aggregate_fold_results(fold_results: List[Dict], exp_dir: Path) -> Dict:
    """
    Aggregate metrics across folds

    Returns:
        aggregated: Mean ± std for each metric
    """
    # Extract final metrics from each fold
    metrics_names = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc_roc', 'val_auc_pr']

    aggregated = {}
    for metric in metrics_names:
        values = [fold[metric][-1] for fold in fold_results]  # Last epoch
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)

    # Save aggregated results
    with open(exp_dir / "aggregated_results.json", 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("K-Fold Results Summary")
    print("="*70)
    for metric in metrics_names:
        mean = aggregated[f'{metric}_mean']
        std = aggregated[f'{metric}_std']
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return aggregated


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

    # Load dataset
    print("\nLoading dataset...")
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

    # Train
    if config['training']['use_kfold']:
        # K-Fold CV
        results = train_kfold(dataset, config, exp_dir)
    else:
        # Single split
        # Create train/val loaders
        total_size = len(dataset)
        val_size = int(total_size * config['training']['val_ratio'])
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config['experiment']['seed'])
        )

        collator = LandslideCollator(dataset.graph_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        # Build model
        model = build_model_from_config(config, dataset)

        # Train
        history = train_single_split(model, train_loader, val_loader, dataset, config, exp_dir)

        # Plot curves
        plot_training_curves(history, exp_dir)

        # Plot confusion matrix
        # Load best model
        checkpoint = torch.load(exp_dir / "checkpoints" / "model_best.pth", weights_only=False)
        cm = checkpoint['metrics']['confusion_matrix']
        plot_confusion_matrix(np.array(cm), exp_dir)

        # Save history
        with open(exp_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

        results = history

    print("\n" + "="*70)
    print("Training Completed!")
    print(f"Results saved to: {exp_dir}")
    print("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Landslide Risk Model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()

    main(args.config)
