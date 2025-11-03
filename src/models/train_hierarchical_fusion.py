"""
Training script for Hierarchical Fusion Model

Features:
- Spatial split (70/30 by slopes, not time)
- Balanced sampling (1:1 or 1:2 ratio per epoch)
- Focal Loss with hard example mining
- Mixed precision training
- TensorBoard logging
- Checkpointing and visualization
"""

import os
import sys
import yaml
import json
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_hierarchical_fusion import HierarchicalFusionModel
from models.losses_hierarchical import HierarchicalCombinedLoss
from models.data_loader_hierarchical import HierarchicalDataset, HierarchicalDatasetDynamic
from models.dynamic_negative_sampler import DynamicNegativeSampler
import rasterio


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_balanced_sampler(samples_df: pd.DataFrame, ratio: float = 1.0):
    """
    Create weighted sampler for balanced training

    Args:
        samples_df: DataFrame with 'label' column
        ratio: negative:positive ratio (1.0 = 1:1, 2.0 = 2:1)

    Returns:
        WeightedRandomSampler
    """
    labels = samples_df['label'].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    print(f"Original distribution - Positive: {n_pos}, Negative: {n_neg}")

    # Weight calculation
    # Positive samples: weight = 1.0
    # Negative samples: weight = (n_pos * ratio) / n_neg
    weights = np.zeros(len(labels))
    weights[labels == 1] = 1.0
    weights[labels == 0] = (n_pos * ratio) / n_neg

    print(f"Sampling ratio - Positive:Negative = 1:{ratio}")
    print(f"Samples per epoch: {int(n_pos * (1 + ratio))}")

    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=int(n_pos * (1 + ratio)),
        replacement=True
    )

    return sampler


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Compute evaluation metrics

    Args:
        probs: predicted probabilities
        labels: ground truth labels
        threshold: classification threshold

    Returns:
        metrics dict
    """
    preds = (probs >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auc_roc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        'auc_pr': average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics.update({
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp)
    })

    return metrics


class HierarchicalTrainer:
    """
    Trainer for Hierarchical Fusion Model
    """

    def __init__(self, config_path: str, resume_checkpoint: str = None):
        self.config = load_config(config_path)
        self.resume_checkpoint = resume_checkpoint
        self.setup_experiment()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }

        # Resume from checkpoint if provided
        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)

    def setup_experiment(self):
        """Setup experiment directories"""
        exp_config = self.config['experiment']

        # Set random seed
        set_seed(exp_config['seed'])

        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{exp_config['name']}_{timestamp}"
        self.exp_dir = Path(exp_config['save_dir']) / exp_name

        self.checkpoint_dir = self.exp_dir / self.config['output']['checkpoint_dir']
        self.plot_dir = self.exp_dir / self.config['output']['plot_dir']
        self.log_dir = self.exp_dir / self.config['output']['log_dir']

        for d in [self.checkpoint_dir, self.plot_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config
        config_save_path = self.exp_dir / 'config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"Experiment directory: {self.exp_dir}")

    def setup_device(self):
        """Setup device (CUDA/CPU)"""
        device_name = self.config['experiment']['device']
        self.device = torch.device(
            device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu'
        )
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def setup_data(self):
        """Setup datasets and dataloaders"""
        print("\n=== Setting up data ===")

        # Load positive samples (fixed for all epochs)
        samples_path = self.config['data']['samples_path']
        positive_samples_df = pd.read_csv(samples_path)
        print(f"Loaded {len(positive_samples_df)} positive samples from {samples_path}")

        train_config = self.config['training']
        use_dynamic_sampling = train_config.get('dynamic_negative_sampling', False)

        if use_dynamic_sampling:
            print("\n=== Using Dynamic Negative Sampling ===")

            # Date range for training/validation (LDAPS available dates)
            date_range = train_config.get('dynamic_date_range', ('2019-01-01', '2020-09-30'))
            print(f"Training period: {date_range[0]} ~ {date_range[1]}")

            # Filter positive samples to training period only
            print(f"\nOriginal positive samples (all years): {len(positive_samples_df)}")
            print(f"  Date range: {positive_samples_df['event_date'].min()} ~ {positive_samples_df['event_date'].max()}")

            train_period_positive = positive_samples_df[
                (positive_samples_df['event_date'] >= date_range[0]) &
                (positive_samples_df['event_date'] <= date_range[1])
            ].reset_index(drop=True)

            print(f"Filtered to training period: {len(train_period_positive)}")
            print(f"  Unique slopes: {train_period_positive['cat'].nunique()}")
            print(f"  Date range: {train_period_positive['event_date'].min()} ~ {train_period_positive['event_date'].max()}")

            # Extract all slope IDs from slope_id_raster
            slope_id_raster_path = self.config['data']['slope_id_raster_path']
            print(f"\nLoading slope IDs from: {slope_id_raster_path}")
            with rasterio.open(slope_id_raster_path) as src:
                slope_id_raster = src.read(1)
            all_slope_ids = np.unique(slope_id_raster[slope_id_raster > 0]).tolist()
            print(f"Total slopes in raster: {len(all_slope_ids):,}")

            # Define negative pool using ALL positive samples (entire history)
            # This ensures slopes with landslides in ANY year are excluded
            all_positive_slope_ids = set(positive_samples_df['cat'].unique())
            print(f"\nSlopes with landslides (entire history): {len(all_positive_slope_ids):,}")
            print("  These slopes will be EXCLUDED from negative pool")

            # Spatial split: group by slope_id (training period positive only)
            unique_pos_slopes = train_period_positive['cat'].unique()

            train_pos_slopes, val_pos_slopes = train_test_split(
                unique_pos_slopes,
                test_size=train_config['val_ratio'],
                random_state=self.config['experiment']['seed'],
                shuffle=True
            )

            train_pos_samples = train_period_positive[train_period_positive['cat'].isin(train_pos_slopes)].reset_index(drop=True)
            val_pos_samples = train_period_positive[train_period_positive['cat'].isin(val_pos_slopes)].reset_index(drop=True)

            print(f"\nTrain positive slopes: {len(train_pos_slopes)}, Val positive slopes: {len(val_pos_slopes)}")
            print(f"Train positive samples: {len(train_pos_samples)}, Val positive samples: {len(val_pos_samples)}")

            # Create dynamic negative samplers
            print("\n=== Creating Dynamic Negative Samplers ===")

            train_neg_sampler = DynamicNegativeSampler(
                positive_samples_df=train_pos_samples,
                all_slope_ids=all_slope_ids,
                date_range=date_range,
                ratio=train_config.get('negative_ratio', 1.0),
                random_date_ratio=train_config.get('random_date_ratio', 0.2),
                seed=self.config['experiment']['seed'],
                exclude_slope_ids=all_positive_slope_ids  # NEW: exclude all historical positives
            )

            val_neg_sampler = DynamicNegativeSampler(
                positive_samples_df=val_pos_samples,
                all_slope_ids=all_slope_ids,
                date_range=date_range,
                ratio=train_config.get('negative_ratio', 1.0),
                random_date_ratio=train_config.get('random_date_ratio', 0.2),
                seed=self.config['experiment']['seed'] + 1000,  # Different seed for val
                exclude_slope_ids=all_positive_slope_ids  # NEW: exclude all historical positives
            )

            # Create datasets
            # Make GNN embedding optional
            use_gnn = self.config['data'].get('use_gnn_embedding', True)
            gnn_embedding_path = self.config['data'].get('gnn_embedding_path') if use_gnn else None

            print(f"\n=== Feature Configuration ===")
            print(f"Static features: {len(self.config['data']['static_features'])}")
            print(f"Static channels: {self.config['model']['static_encoder']['in_channels']}")
            print(f"Use GNN embedding: {use_gnn}")
            if not use_gnn:
                print("  → GNN encoder will be DISABLED")

            self.train_dataset = HierarchicalDatasetDynamic(
                config=self.config,
                negative_sampler=train_neg_sampler,
                gnn_embedding_path=gnn_embedding_path,
                augment=train_config.get('use_augmentation', True),
                aug_prob=train_config.get('augmentation_prob', 0.5)
            )

            self.val_dataset = HierarchicalDatasetDynamic(
                config=self.config,
                negative_sampler=val_neg_sampler,
                gnn_embedding_path=gnn_embedding_path,
                augment=False
            )

            print(f"\nAfter filtering - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

            # Create dataloaders (NO sampler needed - already balanced by DynamicNegativeSampler)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=train_config['batch_size'],
                shuffle=True,  # Shuffle entire combined dataset
                num_workers=train_config['num_workers'],
                pin_memory=train_config['pin_memory'],
                drop_last=True
            )

        else:
            # Original static sampling logic (backward compatibility)
            print("\n=== Using Static Sampling (Legacy Mode) ===")

            samples_df = positive_samples_df  # Assumes CSV has both pos and neg

            # Spatial split: group by slope_id
            unique_slopes = samples_df['cat'].unique()

            train_slopes, val_slopes = train_test_split(
                unique_slopes,
                test_size=train_config['val_ratio'],
                random_state=self.config['experiment']['seed'],
                shuffle=True
            )

            train_samples = samples_df[samples_df['cat'].isin(train_slopes)].reset_index(drop=True)
            val_samples = samples_df[samples_df['cat'].isin(val_slopes)].reset_index(drop=True)

            print(f"Train slopes: {len(train_slopes)}, Val slopes: {len(val_slopes)}")
            print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
            print(f"Train positive: {(train_samples['label']==1).sum()}, "
                  f"negative: {(train_samples['label']==0).sum()}")
            print(f"Val positive: {(val_samples['label']==1).sum()}, "
                  f"negative: {(val_samples['label']==0).sum()}")

            # Create datasets
            # Make GNN embedding optional
            use_gnn = self.config['data'].get('use_gnn_embedding', True)
            gnn_embedding_path = self.config['data'].get('gnn_embedding_path') if use_gnn else None

            print(f"\n=== Feature Configuration ===")
            print(f"Static features: {len(self.config['data']['static_features'])}")
            print(f"Static channels: {self.config['model']['static_encoder']['in_channels']}")
            print(f"Use GNN embedding: {use_gnn}")
            if not use_gnn:
                print("  → GNN encoder will be DISABLED")

            self.train_dataset = HierarchicalDataset(
                self.config,
                train_samples,
                gnn_embedding_path,
                augment=train_config.get('use_augmentation', True),
                aug_prob=train_config.get('augmentation_prob', 0.5)
            )

            self.val_dataset = HierarchicalDataset(
                self.config,
                val_samples,
                gnn_embedding_path,
                augment=False
            )

            print(f"After filtering - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

            # Create dataloaders with balanced sampler
            if train_config.get('use_balanced_sampling', True):
                train_sampler = create_balanced_sampler(
                    self.train_dataset.samples,
                    ratio=train_config['negative_ratio']
                )
                train_shuffle = False
            else:
                train_sampler = None
                train_shuffle = True

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=train_config['batch_size'],
                sampler=train_sampler,
                shuffle=train_shuffle if train_sampler is None else False,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'],
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory']
        )

        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

    def setup_model(self):
        """Setup model and loss function"""
        print("\n=== Setting up model ===")

        # Model
        self.model = HierarchicalFusionModel(self.config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss function
        train_config = self.config['training']
        self.criterion = HierarchicalCombinedLoss(
            alpha=train_config['focal_alpha'],
            gamma=train_config['focal_gamma'],
            alpha_loss_weight=train_config.get('alpha_loss_weight', 0.01),
            use_alpha_loss=train_config.get('use_alpha_loss', True)
        )

        print(f"Loss: Hierarchical MIL (Focal α={train_config['focal_alpha']}, γ={train_config['focal_gamma']}) "
              f"+ Alpha Detail (λ={train_config.get('alpha_loss_weight', 0.01)}, gradient-based)")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        train_config = self.config['training']

        # Optimizer
        if train_config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                betas=train_config.get('betas', [0.9, 0.999]),
                eps=train_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")

        # Scheduler
        if train_config.get('use_scheduler', True):
            if train_config['scheduler_type'] == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=train_config['epochs'],
                    eta_min=train_config['min_lr']
                )
            else:
                raise ValueError(f"Unknown scheduler: {train_config['scheduler_type']}")
        else:
            self.scheduler = None

        # Mixed precision scaler
        self.use_amp = train_config.get('use_amp', True)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler('cuda')
            print("Using mixed precision training (AMP)")
        else:
            self.scaler = None
            print("Using full precision training")

    def setup_logging(self):
        """Setup logging"""
        if self.config['training'].get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        else:
            self.writer = None

    def _compute_4way_summary(self, accumulated: dict) -> dict:
        """
        Compute 4-way summary statistics (Label x Rainfall)

        Groups:
        1. Pos+HighRain: Landslide events with heavy rainfall
        2. Pos+LowRain: Landslide events with light rainfall (terrain-sensitive)
        3. Neg+HighRain: Non-landslide slopes despite heavy rainfall (stable terrain)
        4. Neg+LowRain: Non-landslide slopes with light rainfall (baseline)

        Args:
            accumulated: dict with keys ['alpha_maps', 'p_model_maps', 'labels', 'rainfalls']
                - Each value is a list of tensors (cpu) accumulated across batches

        Returns:
            dict: {
                'Pos_HighRain': {'alpha_mean', 'alpha_std', 'p_model_mean', 'p_model_std', 'count'},
                'Pos_LowRain': {...},
                'Neg_HighRain': {...},
                'Neg_LowRain': {...}
            }
        """
        # Concatenate all accumulated data
        alpha_maps = torch.cat(accumulated['alpha_maps'], dim=0)  # (N,)
        p_model_maps = torch.cat(accumulated['p_model_maps'], dim=0)  # (N,)
        labels = torch.cat(accumulated['labels'], dim=0)  # (N,)
        rainfalls = torch.cat(accumulated['rainfalls'], dim=0)  # (N,)

        # Determine rainfall threshold (median)
        rain_threshold = rainfalls.median()

        # Create masks for 4 groups
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        high_rain_mask = (rainfalls > rain_threshold)
        low_rain_mask = (rainfalls <= rain_threshold)

        groups = {
            'Pos_HighRain': pos_mask & high_rain_mask,
            'Pos_LowRain': pos_mask & low_rain_mask,
            'Neg_HighRain': neg_mask & high_rain_mask,
            'Neg_LowRain': neg_mask & low_rain_mask
        }

        # Compute statistics for each group
        summary = {}
        for group_name, mask in groups.items():
            if mask.sum() > 0:
                alpha_group = alpha_maps[mask]
                p_model_group = p_model_maps[mask]

                summary[group_name] = {
                    'alpha_mean': alpha_group.mean().item(),
                    'alpha_std': alpha_group.std().item(),
                    'p_model_mean': p_model_group.mean().item(),
                    'p_model_std': p_model_group.std().item(),
                    'count': mask.sum().item()
                }
            else:
                # No samples in this group
                summary[group_name] = {
                    'alpha_mean': 0.0,
                    'alpha_std': 0.0,
                    'p_model_mean': 0.0,
                    'p_model_std': 0.0,
                    'count': 0
                }

        return summary

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        # Refresh negative samples if using dynamic sampling
        if hasattr(self.train_dataset, 'refresh_negative_samples'):
            self.train_dataset.refresh_negative_samples(epoch)

        self.model.train()

        epoch_losses = {
            'total': [], 'mil': [], 'alpha': []
        }

        all_probs = []
        all_labels = []

        # === 4-Way Analysis: Accumulation variables ===
        accumulated = {
            'alpha_maps': [],
            'p_model_maps': [],
            'labels': [],
            'rainfalls': []
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            # Check for NaN in input batch (first 5 batches only)
            if batch_idx < 5:
                for key in ['static', 'dynamic', 'gnn_embedding', 'kfs_prior']:
                    if torch.isnan(batch[key]).any():
                        print(f"[Batch {batch_idx}] NaN in input '{key}'")
                        print(f"  Shape: {batch[key].shape}, NaN count: {torch.isnan(batch[key]).sum()}")

            # Forward pass
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(batch)
                    loss_dict = self.criterion(
                        outputs,
                        batch['label'],
                        batch['slope_mask'],
                        batch['kfs_prior']
                    )
                    loss = loss_dict['total_loss']
            else:
                outputs = self.model(batch)

                # Check model output for NaN
                if torch.isnan(outputs['final_output']).any():
                    print(f"[Batch {batch_idx}] NaN in model output! Stopping training.")
                    print(f"  final_output NaN count: {torch.isnan(outputs['final_output']).sum()}")
                    # Print gradients of last few layers
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"    NaN gradient in: {name}")
                    raise RuntimeError("NaN detected in model output")

                loss_dict = self.criterion(
                    outputs,
                    batch['label'],
                    batch['slope_mask'],
                    batch['kfs_prior']
                )
                loss = loss_dict['total_loss']

            # === Dynamic Behavior Monitoring (Batch Level) ===
            with torch.no_grad():
                alpha_map = outputs['alpha_map'].detach()
                p_model_map = outputs['p_model_map'].detach()

                # Extract rainfall: acc7d from most recent timestep (t=-1)
                # batch['dynamic']: (B, T=5, 3, H, W) where 3 = [acc3d, acc7d, peak1h]
                acc7d_recent = batch['dynamic'][:, -1, 1, :, :]  # (B, H, W)
                slope_mask = batch['slope_mask']  # (B, 1, H, W)

                # Compute average rainfall per sample (within slope interior)
                rainfall_per_sample = (
                    (acc7d_recent * slope_mask.squeeze(1)).sum(dim=(1, 2)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )  # (B,)

                # Compute per-sample spatial means (within slope interior)
                alpha_mean_per_sample = (
                    (alpha_map * slope_mask).sum(dim=(1, 2, 3)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )  # (B,)
                p_model_mean_per_sample = (
                    (p_model_map * slope_mask).sum(dim=(1, 2, 3)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )  # (B,)

                # Classify by rainfall: High vs Low (median threshold)
                rain_threshold = rainfall_per_sample.median()
                high_rain_idx = (rainfall_per_sample > rain_threshold)  # (B,) boolean
                low_rain_idx = ~high_rain_idx

                # Compute Alpha/P_model averages by rainfall group (1D indexing)
                avg_alpha_high = alpha_mean_per_sample[high_rain_idx].mean() if high_rain_idx.sum() > 0 else torch.tensor(0.0, device=alpha_map.device)
                avg_alpha_low = alpha_mean_per_sample[low_rain_idx].mean() if low_rain_idx.sum() > 0 else torch.tensor(0.0, device=alpha_map.device)
                avg_p_high = p_model_mean_per_sample[high_rain_idx].mean() if high_rain_idx.sum() > 0 else torch.tensor(0.0, device=p_model_map.device)
                avg_p_low = p_model_mean_per_sample[low_rain_idx].mean() if low_rain_idx.sum() > 0 else torch.tensor(0.0, device=p_model_map.device)

                # === Accumulate for 4-Way Analysis (Epoch Level) ===
                accumulated['alpha_maps'].append(alpha_mean_per_sample.cpu())
                accumulated['p_model_maps'].append(p_model_mean_per_sample.cpu())
                accumulated['labels'].append(batch['label'].cpu())
                accumulated['rainfalls'].append(rainfall_per_sample.cpu())

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('use_grad_clip', True):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip_value']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config['training'].get('use_grad_clip', True):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip_value']
                    )

                self.optimizer.step()

            # Record metrics
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['mil'].append(loss_dict['mil_loss'].item())
            epoch_losses['alpha'].append(loss_dict['alpha_loss'].item())

            all_probs.extend(loss_dict['slope_probs'].cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'mil': f"{loss_dict['mil_loss'].item():.4f}",
                'alpha_l': f"{loss_dict['alpha_loss'].item():.4f}",
                'A_H': f"{avg_alpha_high.item():.2f}",
                'A_L': f"{avg_alpha_low.item():.2f}",
                'P_H': f"{avg_p_high.item():.2f}",
                'P_L': f"{avg_p_low.item():.2f}"
            })

            # TensorBoard logging (per batch)
            if self.writer and batch_idx % self.config['training']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_Total', loss_dict['total_loss'].item(), global_step)
                self.writer.add_scalar('Train/Loss_MIL', loss_dict['mil_loss'].item(), global_step)
                self.writer.add_scalar('Train/Loss_Alpha', loss_dict['alpha_loss'].item(), global_step)

                # Dynamic monitoring (batch level)
                self.writer.add_scalars('Batch/Alpha_ByRain', {
                    'high_rain': avg_alpha_high.item(),
                    'low_rain': avg_alpha_low.item()
                }, global_step)

                self.writer.add_scalars('Batch/PModel_ByRain', {
                    'high_rain': avg_p_high.item(),
                    'low_rain': avg_p_low.item()
                }, global_step)

        # === 4-Way Summary Analysis (Epoch Level) ===
        summary_4way = self._compute_4way_summary(accumulated)

        # Log 4-way statistics to TensorBoard
        if self.writer:
            for group_name, stats in summary_4way.items():
                self.writer.add_scalars(f'Epoch/Alpha_{group_name}', {
                    'mean': stats['alpha_mean'],
                    'std': stats['alpha_std']
                }, epoch)
                self.writer.add_scalars(f'Epoch/PModel_{group_name}', {
                    'mean': stats['p_model_mean'],
                    'std': stats['p_model_std']
                }, epoch)

        # Compute epoch metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = compute_metrics(all_probs, all_labels)
        metrics.update({
            'loss_total': np.mean(epoch_losses['total']),
            'loss_mil': np.mean(epoch_losses['mil']),
            'loss_alpha': np.mean(epoch_losses['alpha'])
        })

        # Add 4-way summary to metrics for logging
        metrics['4way_summary'] = summary_4way

        return metrics

    def validate(self, epoch: int) -> dict:
        """Validation loop"""
        self.model.eval()

        epoch_losses = {
            'total': [], 'mil': [], 'alpha': []
        }

        all_probs = []
        all_labels = []

        # === 4-Way Analysis: Accumulation variables ===
        accumulated = {
            'alpha_maps': [],
            'p_model_maps': [],
            'labels': [],
            'rainfalls': []
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation")

            for batch in pbar:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                loss_dict = self.criterion(
                    outputs,
                    batch['label'],
                    batch['slope_mask'],
                    batch['kfs_prior']
                )

                # Record metrics
                epoch_losses['total'].append(loss_dict['total_loss'].item())
                epoch_losses['mil'].append(loss_dict['mil_loss'].item())
                epoch_losses['alpha'].append(loss_dict['alpha_loss'].item())

                all_probs.extend(loss_dict['slope_probs'].cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())

                # === Accumulate for 4-Way Analysis ===
                alpha_map = outputs['alpha_map'].detach()
                p_model_map = outputs['p_model_map'].detach()
                slope_mask = batch['slope_mask']

                # Extract rainfall: acc7d from most recent timestep (t=-1)
                acc7d_recent = batch['dynamic'][:, -1, 1, :, :]  # (B, H, W)

                # Compute average per sample
                rainfall_per_sample = (
                    (acc7d_recent * slope_mask.squeeze(1)).sum(dim=(1, 2)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )

                alpha_mean_per_sample = (
                    (alpha_map * slope_mask).sum(dim=(1, 2, 3)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )
                p_model_mean_per_sample = (
                    (p_model_map * slope_mask).sum(dim=(1, 2, 3)) /
                    slope_mask.sum(dim=(1, 2, 3)).clamp(min=1)
                )

                accumulated['alpha_maps'].append(alpha_mean_per_sample.cpu())
                accumulated['p_model_maps'].append(p_model_mean_per_sample.cpu())
                accumulated['labels'].append(batch['label'].cpu())
                accumulated['rainfalls'].append(rainfall_per_sample.cpu())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}"
                })

        # === 4-Way Summary Analysis (Epoch Level) ===
        summary_4way = self._compute_4way_summary(accumulated)

        # Log 4-way statistics to TensorBoard
        if self.writer:
            for group_name, stats in summary_4way.items():
                self.writer.add_scalars(f'Epoch_Val/Alpha_{group_name}', {
                    'mean': stats['alpha_mean'],
                    'std': stats['alpha_std']
                }, epoch)
                self.writer.add_scalars(f'Epoch_Val/PModel_{group_name}', {
                    'mean': stats['p_model_mean'],
                    'std': stats['p_model_std']
                }, epoch)

        # Compute metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = compute_metrics(all_probs, all_labels)
        metrics.update({
            'loss_total': np.mean(epoch_losses['total']),
            'loss_mil': np.mean(epoch_losses['mil']),
            'loss_alpha': np.mean(epoch_losses['alpha'])
        })

        # Add 4-way summary to metrics for logging
        metrics['4way_summary'] = summary_4way

        return metrics

    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to console and TensorBoard"""
        # Console
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss_total']:.4f}, "
              f"AUC-ROC: {train_metrics['auc_roc']:.4f}, "
              f"AUC-PR: {train_metrics['auc_pr']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss_total']:.4f}, "
              f"AUC-ROC: {val_metrics['auc_roc']:.4f}, "
              f"AUC-PR: {val_metrics['auc_pr']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Val   - Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")

        # Log alpha loss weight
        alpha_loss_weight = self.criterion.alpha_loss_weight
        print(f"  Alpha Loss Weight: {alpha_loss_weight:.4f}")

        # === Log 4-Way Dynamic Monitoring Summary ===
        if '4way_summary' in train_metrics:
            print("\n  4-Way Dynamic Monitoring (Train):")
            for group_name, stats in train_metrics['4way_summary'].items():
                print(f"    [{group_name:12s}] n={stats['count']:4d} | "
                      f"Alpha: {stats['alpha_mean']:.3f}±{stats['alpha_std']:.3f} | "
                      f"P_model: {stats['p_model_mean']:.3f}±{stats['p_model_std']:.3f}")

        if '4way_summary' in val_metrics:
            print("\n  4-Way Dynamic Monitoring (Val):")
            for group_name, stats in val_metrics['4way_summary'].items():
                print(f"    [{group_name:12s}] n={stats['count']:4d} | "
                      f"Alpha: {stats['alpha_mean']:.3f}±{stats['alpha_std']:.3f} | "
                      f"P_model: {stats['p_model_mean']:.3f}±{stats['p_model_std']:.3f}")

        # TensorBoard
        if self.writer:
            # Losses
            self.writer.add_scalars('Loss/Total', {
                'train': train_metrics['loss_total'],
                'val': val_metrics['loss_total']
            }, epoch)

            # Metrics
            for metric in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']:
                self.writer.add_scalars(f'Metrics/{metric}', {
                    'train': train_metrics[metric],
                    'val': val_metrics[metric]
                }, epoch)

            # Learning rate
            if self.scheduler:
                self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)

            # Alpha loss weight
            self.writer.add_scalar('Alpha_Loss_Weight', alpha_loss_weight, epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'model_hierarchical_best.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best checkpoint to {checkpoint_path}")

        # Save every N epochs
        if epoch % self.config['training']['save_interval'] == 0:
            checkpoint_path = self.checkpoint_dir / f'model_hierarchical_epoch{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        print(f"\n=== Resuming from checkpoint: {checkpoint_path} ===")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model state")

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Loaded optimizer state")

        # Load scheduler state if exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Loaded scheduler state")

        # Load scaler state if exists
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  Loaded scaler state")

        # Resume training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)

        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best metric so far: {self.best_metric:.4f}")
        print("="*70 + "\n")

    def train(self):
        """Main training loop"""
        print("\n=== Starting training ===\n")

        train_config = self.config['training']
        monitor_metric = train_config['monitor_metric']
        early_stopping_patience = train_config.get('early_stopping_patience', 20)
        patience_counter = 0

        start_epoch = self.current_epoch + 1  # Resume from next epoch
        for epoch in range(start_epoch, train_config['epochs'] + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Log
            self.log_metrics(epoch, train_metrics, val_metrics)

            # Save history
            self.history['train_loss'].append(train_metrics['loss_total'])
            self.history['val_loss'].append(val_metrics['loss_total'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            # Check if best model
            current_metric = val_metrics[monitor_metric.replace('val_', '')]
            is_best = current_metric > self.best_metric

            if is_best:
                self.best_metric = current_metric
                patience_counter = 0
                print(f"  ★ New best {monitor_metric}: {self.best_metric:.4f}")
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if train_config.get('early_stopping', True) and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

        # Save final checkpoint
        final_checkpoint_path = self.checkpoint_dir / 'model_hierarchical_final.pth'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, final_checkpoint_path)
        print(f"\n✓ Saved final checkpoint to {final_checkpoint_path}")

        # Save history
        history_path = self.exp_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        print(f"\n=== Training completed ===")
        print(f"Best {monitor_metric}: {self.best_metric:.4f}")
        print(f"Experiment directory: {self.exp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Fusion Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Train
    trainer = HierarchicalTrainer(args.config, resume_checkpoint=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
