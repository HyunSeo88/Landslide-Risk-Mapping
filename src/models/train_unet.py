"""
Training Script for Hierarchical GNN-U-Net Model with MIL

This script implements Multiple Instance Learning (MIL) for the hierarchical model.
Key features:
- End-to-end or two-stage training
- MIL aggregation (max/mean/LSE) for slope-level supervision
- Pixel-level risk prediction
- Comprehensive evaluation metrics

Usage:
    python src/models/train_unet.py --config configs/hierarchical_unet.yaml

Author: Landslide Risk Analysis Project
Date: 2025-01-16
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.windows import Window

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import HierarchicalGNNUNet, MILLoss
from src.models.data_loader_mil import create_mil_dataloaders


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
    (exp_dir / "risk_maps").mkdir(exist_ok=True)
    
    return exp_dir


# ============================================================
# Training Functions
# ============================================================

def train_epoch_mil(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MILLoss,
    device: torch.device,
    cat_to_node: Dict[int, int],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
    log_interval: int = 5
) -> float:
    """
    Train for one epoch with patch-based MIL and end-to-end GNN learning
    
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
                # Stage 1: GNN forward pass (entire graph)
                gnn_logits = model.forward_stage1(
                    batch['static_x'],
                    batch['edge_index'],
                    batch.get('edge_attr')
                )  # (num_nodes,)
                
                gnn_probs = torch.sigmoid(gnn_logits)
                
                # Map GNN outputs to patches
                batch_size = len(batch['cats'])
                zone_patches = batch['zone_patches']  # (B, H, W)
                dynamic_patches = batch['dynamic_patches']  # (B, C, H, W)
                
                # Create GNN patch for each sample
                gnn_patches = []
                for i in range(batch_size):
                    zone_patch = zone_patches[i]  # (H, W)
                    
                    # Create empty GNN patch with same dtype as gnn_probs
                    gnn_patch = torch.zeros_like(zone_patch, dtype=gnn_probs.dtype, device=device)
                    
                    # Get unique slope IDs in this patch
                    unique_cats = torch.unique(zone_patch)
                    
                    # Map GNN values to patch pixels
                    for cat_id in unique_cats:
                        if cat_id.item() == 0:  # Skip NoData
                            continue
                        
                        cat_val = cat_id.item()
                        if cat_val in cat_to_node:
                            node_idx = cat_to_node[cat_val]
                            gnn_patch[zone_patch == cat_id] = gnn_probs[node_idx]
                    
                    gnn_patches.append(gnn_patch)
                
                gnn_patches = torch.stack(gnn_patches).unsqueeze(1)  # (B, 1, H, W)
                
                # Stage 2: U-Net forward pass
                multi_channel_input = torch.cat([gnn_patches, dynamic_patches], dim=1)  # (B, 1+C, H, W)
                pixel_logits = model.forward_stage2(multi_channel_input)  # (B, 1, H, W)
                
                # MIL loss (patch-based)
                loss = criterion(
                    pixel_logits,
                    zone_patches,
                    batch['cats'],
                    batch['labels']
                )
            
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
            # Stage 1: GNN forward pass
            gnn_logits = model.forward_stage1(
                batch['static_x'],
                batch['edge_index'],
                batch.get('edge_attr')
            )
            
            gnn_probs = torch.sigmoid(gnn_logits)
            
            # Map GNN outputs to patches
            batch_size = len(batch['cats'])
            zone_patches = batch['zone_patches']
            dynamic_patches = batch['dynamic_patches']
            
            gnn_patches = []
            for i in range(batch_size):
                zone_patch = zone_patches[i]
                gnn_patch = torch.zeros_like(zone_patch, dtype=gnn_probs.dtype, device=device)
                unique_cats = torch.unique(zone_patch)
                
                for cat_id in unique_cats:
                    if cat_id.item() == 0:
                        continue
                    cat_val = cat_id.item()
                    if cat_val in cat_to_node:
                        node_idx = cat_to_node[cat_val]
                        gnn_patch[zone_patch == cat_id] = gnn_probs[node_idx]
                
                gnn_patches.append(gnn_patch)
            
            gnn_patches = torch.stack(gnn_patches).unsqueeze(1)
            
            # Stage 2: U-Net forward pass
            multi_channel_input = torch.cat([gnn_patches, dynamic_patches], dim=1)
            pixel_logits = model.forward_stage2(multi_channel_input)
            
            # MIL loss
            loss = criterion(
                pixel_logits,
                zone_patches,
                batch['cats'],
                batch['labels']
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
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


def validate_epoch_mil(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MILLoss,
    device: torch.device,
    cat_to_node: Dict[int, int]
) -> Tuple[float, Dict]:
    """
    Validate for one epoch with patch-based MIL
    
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
            
            # Stage 1: GNN forward pass
            gnn_logits = model.forward_stage1(
                batch['static_x'],
                batch['edge_index'],
                batch.get('edge_attr')
            )
            
            gnn_probs = torch.sigmoid(gnn_logits)
            
            # Map GNN outputs to patches
            batch_size = len(batch['cats'])
            zone_patches = batch['zone_patches']
            dynamic_patches = batch['dynamic_patches']
            
            gnn_patches = []
            for i in range(batch_size):
                zone_patch = zone_patches[i]
                gnn_patch = torch.zeros_like(zone_patch, dtype=gnn_probs.dtype, device=device)
                unique_cats = torch.unique(zone_patch)
                
                for cat_id in unique_cats:
                    if cat_id.item() == 0:
                        continue
                    cat_val = cat_id.item()
                    if cat_val in cat_to_node:
                        node_idx = cat_to_node[cat_val]
                        gnn_patch[zone_patch == cat_id] = gnn_probs[node_idx]
                
                gnn_patches.append(gnn_patch)
            
            gnn_patches = torch.stack(gnn_patches).unsqueeze(1)
            
            # Stage 2: U-Net forward pass
            multi_channel_input = torch.cat([gnn_patches, dynamic_patches], dim=1)
            pixel_logits = model.forward_stage2(multi_channel_input)
            
            # MIL loss
            loss = criterion(
                pixel_logits,
                zone_patches,
                batch['cats'],
                batch['labels']
            )
            total_loss += loss.item()
            
            # Extract representative scores for each slope
            for i in range(batch_size):
                cat = batch['cats'][i].item()
                zone_patch = zone_patches[i]  # (H, W)
                
                # Create valid mask (exclude NoData = 0)
                valid_mask = (zone_patch > 0)
                
                # Create mask for this slope (only valid pixels)
                slope_mask = (zone_patch == cat) & valid_mask
                
                if slope_mask.sum() == 0:
                    representative_score = 0.0
                else:
                    # Extract slope pixels from prediction (only valid, no NoData)
                    slope_logits = pixel_logits[i, 0, slope_mask]  # (num_pixels,)
                    
                    # Use same aggregation as training
                    if criterion.aggregation == 'max':
                        representative_score = slope_logits.max().item()
                    elif criterion.aggregation == 'mean':
                        representative_score = slope_logits.mean().item()
                    elif criterion.aggregation == 'lse':
                        representative_score = torch.logsumexp(slope_logits, dim=0).item()
                    else:
                        representative_score = slope_logits.max().item()  # Default
                
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
# Checkpointing and Logging
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
        checkpoint_path = exp_dir / "checkpoints" / "model_best.pth"
        print(f"  → Saving best model: {checkpoint_path}")
    else:
        checkpoint_path = exp_dir / "checkpoints" / "model_final.pth"
    
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


def generate_risk_map(
    model: nn.Module,
    config: Dict,
    dataset,
    cat_to_node: Dict[int, int],
    device: torch.device,
    output_path: str,
    target_date: str = '20200615'
):
    """
    Generate full risk map for entire study area
    
    Args:
        model: Trained model
        config: Configuration dict
        dataset: Dataset object (for slope_id_raster, etc.)
        cat_to_node: Mapping from slope ID to node index
        device: Device
        output_path: Output GeoTIFF path
        target_date: Target date for prediction (YYYYMMDD)
    """
    print(f"\nGenerating risk map for {target_date}...")
    model.eval()
    
    # Load slope ID raster
    slope_id_raster = dataset.slope_id_raster  # (H, W)
    H, W = slope_id_raster.shape
    
    # Load graph data
    static_x = dataset.static_features.to(device)
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device) if dataset.edge_attr is not None else None
    
    # GNN forward (once for all slopes)
    print("  Running GNN forward pass...")
    with torch.no_grad():
        gnn_logits = model.forward_stage1(static_x, edge_index, edge_attr)
        gnn_probs = torch.sigmoid(gnn_logits).cpu().numpy()
    
    # Create GNN susceptibility raster
    print("  Creating GNN susceptibility raster...")
    gnn_raster = np.zeros((H, W), dtype=np.float32)
    for cat in np.unique(slope_id_raster):
        if cat == 0:
            continue
        if int(cat) in cat_to_node:
            node_idx = cat_to_node[int(cat)]
            gnn_raster[slope_id_raster == cat] = gnn_probs[node_idx]
    
    # Load dynamic rasters for target date
    print(f"  Loading dynamic rasters for {target_date}...")
    dynamic_rasters = []
    raster_base = config['data']['raster_base_path']
    
    for var in config['data']['dynamic_variables']:
        raster_path = os.path.join(raster_base, var, f"{target_date}_{var}_mm_5179_30m.tif")
        if not os.path.exists(raster_path):
            print(f"    Warning: {raster_path} not found, using zeros")
            dynamic_rasters.append(np.zeros((H, W), dtype=np.float32))
        else:
            with rasterio.open(raster_path) as src:
                dynamic_rasters.append(src.read(1).astype(np.float32))
    
    dynamic_stack = np.stack(dynamic_rasters, axis=0)  # (C, H, W)
    
    # Create full input stack
    full_input = np.vstack([gnn_raster[np.newaxis, :, :], dynamic_stack])  # (1+C, H, W)
    full_input_tensor = torch.from_numpy(full_input).unsqueeze(0).to(device)  # (1, 1+C, H, W)
    
    # U-Net forward (full image - might need patching for very large images)
    print("  Running U-Net forward pass...")
    with torch.no_grad():
        if H * W > 30000000:  # If too large, use patch-based prediction
            print("    Image too large, using patch-based prediction...")
            risk_map = predict_with_patches(model, full_input_tensor, patch_size=512, device=device)
        else:
            pixel_logits = model.forward_stage2(full_input_tensor)  # (1, 1, H, W)
            risk_probs = torch.sigmoid(pixel_logits[0, 0]).cpu().numpy()  # (H, W)
            risk_map = risk_probs
    
    # Mask NoData regions
    print("  Masking NoData regions...")
    valid_mask = (slope_id_raster > 0)
    risk_map[~valid_mask] = -9999
    print(f"    Valid: {valid_mask.sum():,} pixels ({valid_mask.sum()/(H*W)*100:.1f}%)")
    
    # Save as GeoTIFF
    print(f"  Saving risk map to {output_path}...")
    with rasterio.open(config['data']['slope_id_raster_path']) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=-9999)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(risk_map.astype(np.float32), 1)
    
    print(f"  ✓ Risk map saved!")
    print(f"    Shape: {risk_map.shape}")
    print(f"    Range: [{risk_map.min():.4f}, {risk_map.max():.4f}]")
    print(f"    Mean: {risk_map.mean():.4f}")


def predict_with_patches(model, full_input, patch_size=512, overlap=64, device='cuda'):
    """Patch-based prediction for very large images"""
    _, C, H, W = full_input.shape
    risk_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    stride = patch_size - overlap
    
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # Extract patch
            i_end = min(i + patch_size, H)
            j_end = min(j + patch_size, W)
            
            patch = full_input[:, :, i:i_end, j:j_end]
            
            # Pad if necessary
            if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                padded = torch.zeros(1, C, patch_size, patch_size, device=device)
                padded[:, :, :patch.shape[2], :patch.shape[3]] = patch
                patch = padded
            
            # Predict
            with torch.no_grad():
                logits = model.forward_stage2(patch)
                probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
            
            # Merge
            pred_h, pred_w = i_end - i, j_end - j
            risk_map[i:i_end, j:j_end] += probs[:pred_h, :pred_w]
            count_map[i:i_end, j:j_end] += 1
    
    # Average overlapping regions
    risk_map = np.divide(risk_map, count_map, where=count_map > 0)
    
    return risk_map


# ============================================================
# Main Training Loop
# ============================================================

def train(config: Dict, exp_dir: Path):
    """Main training function"""
    
    device = torch.device(config['experiment']['device'])
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, dataset = create_mil_dataloaders(
        graph_path=config['data']['graph_path'],
        samples_path=config['data']['samples_path'],
        slope_polygons_path=config['data']['slope_polygons_path'],
        raster_base_path=config['data']['raster_base_path'],
        reference_raster_path=config['data']['reference_raster_path'],
        slope_id_raster_path=config['data']['slope_id_raster_path'],
        slope_bboxes_cache_path=config['data']['slope_bboxes_cache_path'],
        patch_size=config['data']['patch_size'],
        batch_size=config['training_mil']['batch_size'],
        dynamic_variables=config['data']['dynamic_variables'],
        dynamic_statistics=config['data']['dynamic_statistics'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        train_ratio=1.0 - config['training_mil']['val_ratio'],
        random_seed=config['experiment']['seed'],
        num_workers=0
    )
    
    # Get cat_to_node mapping for GNN output mapping
    cat_to_node = dataset.cat_to_node
    
    # Build model
    print("\nBuilding model...")
    model = HierarchicalGNNUNet(
        static_dim=dataset.static_features.shape[1],
        gnn_hidden=config['model_unet']['gnn_hidden'],
        gnn_layers=config['model_unet']['gnn_layers'],
        gnn_type=config['model_unet']['gnn_type'],
        gnn_dropout=config['model_unet']['gnn_dropout'],
        gat_heads=config['model_unet']['gat_heads'],
        dynamic_channels=dataset.num_dynamic_channels,
        unet_base_channels=config['model_unet']['unet_base_channels'],
        unet_depth=config['model_unet']['unet_depth']
    )
    model = model.to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training_mil']['learning_rate'],
        weight_decay=config['training_mil']['weight_decay']
    )
    
    # Loss function
    criterion = MILLoss(
        aggregation=config['training_mil']['mil_aggregation'],
        base_loss=config['training_mil']['loss_type'],
        pos_weight=config['training_mil'].get('pos_weight')
    ).to(device)
    
    # Scheduler (optional)
    scheduler = None
    warmup_scheduler = None
    scheduler_type = None
    
    if config['training_mil'].get('use_scheduler', False):
        scheduler_type = config['training_mil'].get('scheduler_type', 'warmup_cosine')
        
        if scheduler_type == 'plateau':
            # ReduceLROnPlateau: Adaptive learning rate based on validation metric
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # 'max' for AUC-ROC/F1, 'min' for loss
                factor=config['training_mil'].get('scheduler_factor', 0.5),
                patience=config['training_mil'].get('scheduler_patience', 5)
            )
            print(f"  Using ReduceLROnPlateau scheduler (patience={config['training_mil'].get('scheduler_patience', 5)}, factor={config['training_mil'].get('scheduler_factor', 0.5)})")
            print(f"  Learning rate will be reduced by factor {config['training_mil'].get('scheduler_factor', 0.5)} if no improvement for {config['training_mil'].get('scheduler_patience', 5)} epochs")
        
        elif scheduler_type == 'warmup_cosine':
            # Warmup + Cosine Annealing
            warmup_epochs = config['training_mil'].get('warmup_epochs', 10)
            total_epochs = config['training_mil']['epochs']
            
            # Warmup: Linear warmup from 0 to initial LR
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of LR
                end_factor=1.0,     # End at 100% of LR
                total_iters=warmup_epochs
            )
            
            # Cosine Annealing: After warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=config['training_mil'].get('min_lr', 1e-7)
            )
            
            # Sequential scheduler: Warmup → Cosine
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            
            print(f"  Using Warmup + CosineAnnealing scheduler")
            print(f"    Phase 1 - Warmup ({warmup_epochs} epochs):")
            print(f"      LR increases: {config['training_mil']['learning_rate']*0.01:.6f} (1%) → {config['training_mil']['learning_rate']:.6f} (100%)")
            print(f"    Phase 2 - Cosine Annealing ({total_epochs - warmup_epochs} epochs):")
            print(f"      LR decreases: {config['training_mil']['learning_rate']:.6f} → {config['training_mil'].get('min_lr', 1e-7):.6f}")
            print(f"    Total: Epoch 1-{warmup_epochs} (warmup) → Epoch {warmup_epochs+1}-{total_epochs} (cosine)")
        
        else:
            # Simple Cosine Annealing: Fixed schedule
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training_mil']['epochs'],
                eta_min=config['training_mil'].get('min_lr', 1e-7)
            )
            print(f"  Using CosineAnnealingLR scheduler")
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config['training_mil'].get('use_amp', False) else None
    
    # Tensorboard
    writer = None
    if config['training_mil'].get('use_tensorboard', False):
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
    monitor_metric = config['training_mil']['monitor_metric']
    
    epochs = config['training_mil']['epochs']
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch_mil(
            model, train_loader, optimizer, criterion, device,
            cat_to_node,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=config['training_mil'].get('use_amp', False),
            grad_clip=config['training_mil'].get('grad_clip_value'),
            log_interval=config['training_mil']['log_interval']
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch_mil(
            model, val_loader, criterion, device, cat_to_node
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
            if scheduler_type == 'plateau':
                # ReduceLROnPlateau: needs validation metric
                scheduler.step(val_metrics[monitor_metric.replace('val_', '')])
            elif scheduler_type == 'warmup_cosine' or scheduler_type == 'cosine':
                # Warmup+Cosine or Cosine: step every epoch
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
        if patience_counter >= config['training_mil']['early_stopping_patience']:
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
    checkpoint = torch.load(exp_dir / "checkpoints" / "model_best.pth", weights_only=False)
    cm = checkpoint['metrics']['confusion_matrix']
    plot_confusion_matrix(np.array(cm), exp_dir)
    
    # Save history
    with open(exp_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate risk maps
    print("\n" + "="*70)
    print("Generating Risk Maps")
    print("="*70)
    
    # Load best model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    risk_map_dir = exp_dir / "risk_maps"
    risk_map_dir.mkdir(exist_ok=True)
    
    # Generate risk maps for selected dates
    target_dates = ['20200615', '20200715', '20200815']  # Mid-season dates
    
    for date in target_dates:
        output_path = risk_map_dir / f"risk_map_{date}.tif"
        try:
            generate_risk_map(
                model=model,
                config=config,
                dataset=dataset,
                cat_to_node=cat_to_node,
                device=device,
                output_path=str(output_path),
                target_date=date
            )
        except Exception as e:
            print(f"  ✗ Failed to generate risk map for {date}: {e}")
    
    print("\n" + "="*70)
    print("Risk Map Generation Complete!")
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
    parser = argparse.ArgumentParser(description="Train Hierarchical GNN-U-Net Model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args.config)

