"""
SHAP Analysis for Landslide Risk Model

Two-stage SHAP analysis:
1. Dynamic features: SHAP on RNN input (with fixed GNN embeddings)
2. Static features: SHAP on Proxy model (simple MLP)

Author: Landslide Risk Analysis Project
Date: 2025-01-15
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    print("WARNING: SHAP not installed. Install with: pip install shap")
    shap = None

from src.models.model import LandslideRiskModel, StaticFeatureProxy
from src.models.data_loader import LandslideDataset, LandslideCollator


# ============================================================
# Stage 1: Dynamic Feature SHAP
# ============================================================

class DynamicFeatureSHAP:
    """
    Analyze dynamic feature importance with fixed GNN embeddings

    Method:
    1. Fix GNN embeddings (h_gnn) for all samples
    2. Create wrapper function: dynamic_x -> logits
    3. Apply SHAP DeepExplainer on RNN + Fusion + Classifier
    """

    def __init__(self, model: LandslideRiskModel, dataset: LandslideDataset, device: str = 'cuda'):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Feature names
        self.dynamic_feature_names = self._get_dynamic_feature_names()
        self.dynamic_temporal_names = self._get_temporal_feature_names()

    def _get_dynamic_feature_names(self) -> List[str]:
        """Get base dynamic feature names"""
        names = [
            'acc3d_mean', 'acc3d_max',
            'acc7d_mean', 'acc7d_max',
            'peak1h_mean', 'peak1h_max'
        ]

        if self.dataset.use_insar:
            names.extend(['cumulative_displacement', 'displacement_delay_days'])

        if self.dataset.use_ndvi:
            names.append('ndvi_diff')

        return names

    def _get_temporal_feature_names(self) -> List[str]:
        """Get temporal feature names (with day offset)"""
        names = []
        window_size = self.dataset.window_size

        for t in range(window_size):
            day_label = f"day_{t - window_size + 1}"  # day_-4, day_-3, ..., day_0
            for feat in self.dynamic_feature_names:
                names.append(f'{feat}_{day_label}')

        return names

    def prepare_fixed_gnn_embeddings(self, samples: List[int]) -> torch.Tensor:
        """
        Compute fixed GNN embeddings for given samples

        Args:
            samples: List of sample indices

        Returns:
            h_gnn_fixed: (num_samples, gnn_hidden) - fixed GNN embeddings
        """
        # Create subset dataloader
        subset = Subset(self.dataset, samples)
        collator = LandslideCollator(self.dataset.graph_data)
        loader = DataLoader(subset, batch_size=len(samples), shuffle=False,
                          collate_fn=collator, num_workers=0)

        # Get batch
        batch = next(iter(loader))
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        # Compute GNN embeddings
        with torch.no_grad():
            static_x = batch['static_x']
            edge_index = batch['edge_index']
            edge_attr = batch['edge_attr'] if self.model.gnn_type == 'sage' else None
            node_indices = batch['node_indices']

            h_gnn_all = self.model.gnn(static_x, edge_index, edge_attr)
            h_gnn = h_gnn_all[node_indices]

        return h_gnn

    def analyze(self,
                test_samples: List[int],
                background_samples: List[int],
                save_dir: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Perform SHAP analysis on dynamic features

        Args:
            test_samples: Sample indices to analyze
            background_samples: Background samples for SHAP
            save_dir: Directory to save plots (optional)

        Returns:
            shap_values: SHAP values for test samples
            feature_names: Temporal feature names
        """
        if shap is None:
            raise ImportError("SHAP not installed")

        print("\n" + "="*70)
        print("Stage 1: Dynamic Feature SHAP Analysis")
        print("="*70)

        # Prepare data
        print(f"\nPreparing data...")
        print(f"  Test samples: {len(test_samples)}")
        print(f"  Background samples: {len(background_samples)}")

        # Get GNN embeddings (fixed)
        h_gnn_test = self.prepare_fixed_gnn_embeddings(test_samples)
        h_gnn_background = self.prepare_fixed_gnn_embeddings(background_samples)

        # Get dynamic features
        test_dynamic = []
        background_dynamic = []

        for idx in test_samples:
            sample = self.dataset[idx]
            test_dynamic.append(sample['dynamic_features'].numpy())

        for idx in background_samples:
            sample = self.dataset[idx]
            background_dynamic.append(sample['dynamic_features'].numpy())

        test_dynamic = np.array(test_dynamic)  # (n_test, window_size, dynamic_dim)
        background_dynamic = np.array(background_dynamic)  # (n_bg, window_size, dynamic_dim)

        # Flatten temporal dimension
        test_dynamic_flat = test_dynamic.reshape(len(test_samples), -1)
        background_dynamic_flat = background_dynamic.reshape(len(background_samples), -1)

        print(f"\nDynamic features shape: {test_dynamic.shape}")
        print(f"Flattened shape: {test_dynamic_flat.shape}")

        # SHAP Gradient Explainer (for PyTorch)
        print(f"\nInitializing SHAP explainer...")
        
        # Convert to torch tensors for GradientExplainer
        background_tensor = torch.FloatTensor(background_dynamic_flat).to(self.device)
        test_tensor = torch.FloatTensor(test_dynamic_flat).to(self.device)
        
        # Create a wrapper model that takes flattened input and maintains gradient
        class FlattenedModel(nn.Module):
            def __init__(self, model, h_gnn_test, h_gnn_background, window_size, dynamic_dim, n_test):
                super().__init__()
                self.model = model
                self.h_gnn_test = h_gnn_test
                self.h_gnn_background = h_gnn_background
                self.window_size = window_size
                self.dynamic_dim = dynamic_dim
                self.n_test = n_test
            
            def forward(self, x):
                """
                x: (batch, window_size * dynamic_dim) - flattened dynamic features
                Returns: (batch, 1) - probabilities
                """
                # Reshape to (batch, window_size, dynamic_dim)
                batch_size = x.shape[0]
                dynamic_x = x.reshape(batch_size, self.window_size, self.dynamic_dim)
                
                # Determine which h_gnn to use (test or background)
                if batch_size == self.n_test:
                    h_gnn = self.h_gnn_test
                else:
                    # Use background embeddings (repeated if needed)
                    h_gnn = self.h_gnn_background[:batch_size]
                
                # Forward pass (without no_grad to maintain gradient)
                logits = self.model.forward_from_embeddings(h_gnn, dynamic_x)
                probs = torch.sigmoid(logits)
                
                # Ensure 2D output
                if probs.dim() == 1:
                    probs = probs.unsqueeze(1)
                
                return probs
        
        # IMPORTANT: Set main model to train mode FIRST for gradient computation with RNN/LSTM
        # Force all submodules to train mode (especially LSTM)
        self.model.train()
        for module in self.model.modules():
            module.train()
        
        wrapped_model = FlattenedModel(
            self.model,
            h_gnn_test,
            h_gnn_background,
            self.dataset.window_size,
            self.dataset.dynamic_dim,
            len(test_samples)
        ).to(self.device)
        
        # Set wrapper to train mode as well
        wrapped_model.train()
        for module in wrapped_model.modules():
            module.train()
        
        # Enable gradient for input tensors
        background_tensor.requires_grad = True
        test_tensor.requires_grad = True
        
        explainer = shap.GradientExplainer(
            wrapped_model,
            background_tensor
        )

        # Compute SHAP values
        print(f"Computing SHAP values...")
        shap_values_raw = explainer.shap_values(test_tensor)
        
        # Convert to numpy and handle different output formats
        if isinstance(shap_values_raw, list):
            # If list (multi-output), take first element
            shap_values_tensor = shap_values_raw[0]
        else:
            shap_values_tensor = shap_values_raw
            
        if isinstance(shap_values_tensor, torch.Tensor):
            shap_values = shap_values_tensor.detach().cpu().numpy()
        else:
            shap_values = shap_values_tensor
        
        # Ensure 2D shape (samples, features)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)

        print(f"SHAP values shape: {shap_values.shape}")
        
        # Restore model to eval mode
        self.model.eval()

        # Save plots
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving plots to {save_dir}...")

            # Summary plot
            self.plot_summary(shap_values, test_dynamic_flat,
                            save_dir / "shap_dynamic_summary.png")

            # Bar plot (aggregated by feature type)
            self.plot_feature_importance(shap_values,
                                        save_dir / "shap_dynamic_importance.png")

            # Temporal importance
            self.plot_temporal_importance(shap_values,
                                        save_dir / "shap_temporal_importance.png")

        return shap_values, self.dynamic_temporal_names

    def plot_summary(self, shap_values: np.ndarray, features: np.ndarray,
                    save_path: Path):
        """Plot SHAP summary"""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, features,
                         feature_names=self.dynamic_temporal_names,
                         show=False, max_display=20)
        plt.title("Dynamic Feature SHAP Summary")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")

    def plot_feature_importance(self, shap_values: np.ndarray, save_path: Path):
        """Plot aggregated feature importance (across time)"""
        # Aggregate SHAP values by feature type (sum across time steps)
        feature_importance = {}

        for feat_name in self.dynamic_feature_names:
            # Find indices for this feature across all time steps
            indices = [i for i, name in enumerate(self.dynamic_temporal_names)
                      if feat_name in name]

            # Sum absolute SHAP values
            importance = np.mean(np.abs(shap_values[:, indices]))
            feature_importance[feat_name] = importance

        # Sort by importance
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: x[1], reverse=True)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        names, values = zip(*sorted_features)

        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Dynamic Feature Importance (Aggregated Across Time)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")

    def plot_temporal_importance(self, shap_values: np.ndarray, save_path: Path):
        """Plot temporal patterns (importance by day offset)"""
        window_size = self.dataset.window_size

        # Aggregate by time step
        temporal_importance = np.zeros(window_size)

        for t in range(window_size):
            # Indices for this time step
            start_idx = t * self.dataset.dynamic_dim
            end_idx = (t + 1) * self.dataset.dynamic_dim

            # Mean absolute SHAP
            temporal_importance[t] = np.mean(np.abs(shap_values[:, start_idx:end_idx]))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        days = [f"Day {t - window_size + 1}" for t in range(window_size)]
        ax.plot(days, temporal_importance, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean |SHAP value|')
        ax.set_title('Temporal Importance Pattern')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")


# ============================================================
# Stage 2: Static Feature SHAP via Proxy Model
# ============================================================

class StaticFeatureSHAP:
    """
    Analyze static feature importance using proxy model

    Method:
    1. Train simple MLP on static features only
    2. Apply SHAP on proxy model (easier interpretation)
    3. Proxy model doesn't need high accuracy - only for ranking
    """

    def __init__(self, dataset: LandslideDataset, device: str = 'cuda'):
        self.dataset = dataset
        self.device = torch.device(device)

        # Feature names
        self.static_feature_names = self._get_static_feature_names()

    def _get_static_feature_names(self) -> List[str]:
        """Get static feature names"""
        return [
            'has_forestroad', 'dem_average', 'slope_average', 'aspect_average',
            'curv_plan_average', 'curv_prof_average', 'twi_average', 'lnspi_average',
            'tri_sd3_average', 'tpi90_average', 'dist_fault', 'dist_stream',
            'rock2_ratio', 'rock3_ratio', 'rock4_ratio'
        ]

    def train_proxy_model(self,
                         train_samples: List[int],
                         val_samples: List[int],
                         epochs: int = 50,
                         lr: float = 0.001) -> StaticFeatureProxy:
        """
        Train proxy model on static features

        Returns:
            proxy_model: Trained proxy model
        """
        print("\n" + "="*70)
        print("Training Proxy Model for Static Features")
        print("="*70)

        # Prepare data
        X_train, y_train = [], []
        X_val, y_val = [], []

        for idx in train_samples:
            sample = self.dataset[idx]
            node_idx = sample['node_idx'].item()
            static_feat = self.dataset.static_features[node_idx].numpy()
            X_train.append(static_feat)
            y_train.append(sample['label'].item())

        for idx in val_samples:
            sample = self.dataset[idx]
            node_idx = sample['node_idx'].item()
            static_feat = self.dataset.static_features[node_idx].numpy()
            X_val.append(static_feat)
            y_val.append(sample['label'].item())

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        print(f"\nTrain size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")

        # Build proxy model
        proxy_model = StaticFeatureProxy(
            input_dim=self.dataset.static_features.shape[1],
            hidden_dims=[64, 32],
            dropout=0.3
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(proxy_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_acc = 0.0

        for epoch in range(epochs):
            # Train
            proxy_model.train()
            optimizer.zero_grad()

            logits = proxy_model(X_train)
            loss = criterion(logits, y_train)

            loss.backward()
            optimizer.step()

            # Validate
            proxy_model.eval()
            with torch.no_grad():
                val_logits = proxy_model(X_val)
                val_loss = criterion(val_logits, y_val)

                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss.item():.4f} | "
                      f"Val Acc: {val_acc:.4f}")

        print(f"\nProxy Model Training Complete!")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"(Note: Low accuracy is OK - we only need feature ranking)")

        return proxy_model

    def analyze(self,
               proxy_model: StaticFeatureProxy,
               test_samples: List[int],
               background_samples: List[int],
               save_dir: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Perform SHAP analysis on static features using proxy model

        Returns:
            shap_values: SHAP values
            feature_names: Static feature names
        """
        if shap is None:
            raise ImportError("SHAP not installed")

        print("\n" + "="*70)
        print("Stage 2: Static Feature SHAP Analysis (Proxy Model)")
        print("="*70)

        # Prepare data
        X_test, X_background = [], []

        for idx in test_samples:
            sample = self.dataset[idx]
            node_idx = sample['node_idx'].item()
            X_test.append(self.dataset.static_features[node_idx].numpy())

        for idx in background_samples:
            sample = self.dataset[idx]
            node_idx = sample['node_idx'].item()
            X_background.append(self.dataset.static_features[node_idx].numpy())

        X_test = np.array(X_test)
        X_background = np.array(X_background)

        print(f"\nTest samples: {len(X_test)}")
        print(f"Background samples: {len(X_background)}")

        # SHAP explainer (GradientExplainer for PyTorch)
        print(f"\nInitializing SHAP explainer...")
        
        # Set proxy model to train mode for gradient computation
        proxy_model.train()
        
        # Convert to tensors and enable gradient
        background_tensor = torch.FloatTensor(X_background).to(self.device)
        test_tensor = torch.FloatTensor(X_test).to(self.device)
        background_tensor.requires_grad = True
        test_tensor.requires_grad = True
        
        explainer = shap.GradientExplainer(
            proxy_model,
            background_tensor
        )

        # Compute SHAP values
        print(f"Computing SHAP values...")
        shap_values_raw = explainer.shap_values(test_tensor)
        
        # Convert to numpy and handle different output formats
        if isinstance(shap_values_raw, list):
            # If list (multi-output), take first element
            shap_values_tensor = shap_values_raw[0]
        else:
            shap_values_tensor = shap_values_raw
            
        if isinstance(shap_values_tensor, torch.Tensor):
            shap_values = shap_values_tensor.detach().cpu().numpy()
        else:
            shap_values = shap_values_tensor
        
        # Ensure 2D shape (samples, features)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)

        print(f"SHAP values shape: {shap_values.shape}")
        
        # Restore proxy model to eval mode
        proxy_model.eval()

        # Save plots
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving plots to {save_dir}...")

            # Summary plot
            self.plot_summary(shap_values, X_test,
                            save_dir / "shap_static_summary.png")

            # Bar plot
            self.plot_feature_importance(shap_values,
                                        save_dir / "shap_static_importance.png")

        return shap_values, self.static_feature_names

    def plot_summary(self, shap_values: np.ndarray, features: np.ndarray,
                    save_path: Path):
        """Plot SHAP summary"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, features,
                         feature_names=self.static_feature_names,
                         show=False)
        plt.title("Static Feature SHAP Summary (Proxy Model)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")

    def plot_feature_importance(self, shap_values: np.ndarray, save_path: Path):
        """Plot feature importance bar chart"""
        # Mean absolute SHAP
        importance = np.mean(np.abs(shap_values), axis=0)

        # Sort
        sorted_idx = np.argsort(importance)[::-1]
        sorted_names = [self.static_feature_names[i] for i in sorted_idx]
        sorted_values = importance[sorted_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_names)), sorted_values, color='coral')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Static Feature Importance (Proxy Model)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")


# ============================================================
# Stage 3: Integrated Analysis
# ============================================================

class IntegratedSHAPAnalysis:
    """
    Combine Stage 1 and Stage 2 results
    """

    def __init__(self,
                 dynamic_shap_values: np.ndarray,
                 static_shap_values: np.ndarray,
                 dynamic_feature_names: List[str],
                 static_feature_names: List[str],
                 dataset: LandslideDataset):
        self.dynamic_shap = dynamic_shap_values
        self.static_shap = static_shap_values
        self.dynamic_names = dynamic_feature_names
        self.static_names = static_feature_names
        self.dataset = dataset

    def aggregate_dynamic_importance(self) -> Dict[str, float]:
        """Aggregate dynamic SHAP across time steps"""
        # Base feature names (without time suffix)
        base_features = [
            'acc3d_mean', 'acc3d_max',
            'acc7d_mean', 'acc7d_max',
            'peak1h_mean', 'peak1h_max'
        ]

        if self.dataset.use_insar:
            base_features.extend(['cumulative_displacement', 'displacement_delay_days'])

        if self.dataset.use_ndvi:
            base_features.append('ndvi_diff')

        importance = {}
        for feat in base_features:
            # Find all temporal instances
            indices = [i for i, name in enumerate(self.dynamic_names) if feat in name]
            # Mean absolute SHAP
            importance[feat] = np.mean(np.abs(self.dynamic_shap[:, indices]))

        return importance

    def aggregate_static_importance(self) -> Dict[str, float]:
        """Aggregate static SHAP"""
        importance = {}
        for i, feat in enumerate(self.static_names):
            importance[feat] = np.mean(np.abs(self.static_shap[:, i]))

        return importance

    def plot_integrated_comparison(self, save_path: Path, top_k: int = 5):
        """
        Combined bar plot: Top-K from each stream
        """
        dyn_imp = self.aggregate_dynamic_importance()
        stat_imp = self.aggregate_static_importance()

        # Top K
        top_dyn = sorted(dyn_imp.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_stat = sorted(stat_imp.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Dynamic
        names_dyn, values_dyn = zip(*top_dyn)
        ax1.barh(range(len(names_dyn)), values_dyn, color='steelblue')
        ax1.set_yticks(range(len(names_dyn)))
        ax1.set_yticklabels(names_dyn)
        ax1.set_xlabel('Mean |SHAP value|')
        ax1.set_title(f'Top {top_k} Dynamic Features')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Static
        names_stat, values_stat = zip(*top_stat)
        ax2.barh(range(len(names_stat)), values_stat, color='coral')
        ax2.set_yticks(range(len(names_stat)))
        ax2.set_yticklabels(names_stat)
        ax2.set_xlabel('Mean |SHAP value|')
        ax2.set_title(f'Top {top_k} Static Features')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved integrated plot: {save_path.name}")

    def generate_report(self, save_path: Path):
        """Generate text report"""
        dyn_imp = self.aggregate_dynamic_importance()
        stat_imp = self.aggregate_static_importance()

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SHAP Feature Importance Analysis Report\n")
            f.write("="*70 + "\n\n")

            # Dynamic features
            f.write("DYNAMIC FEATURES (Temporal Trigger Factors):\n")
            f.write("-"*70 + "\n")
            for feat, val in sorted(dyn_imp.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {feat:30s}: {val:.4f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("STATIC FEATURES (Inherent Susceptibility Factors):\n")
            f.write("-"*70 + "\n")
            for feat, val in sorted(stat_imp.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {feat:30s}: {val:.4f}\n")

            # Total importance
            total_dyn = sum(dyn_imp.values())
            total_stat = sum(stat_imp.values())
            total = total_dyn + total_stat

            f.write("\n" + "="*70 + "\n")
            f.write("OVERALL CONTRIBUTION:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Dynamic features: {total_dyn:.4f} ({total_dyn/total*100:.1f}%)\n")
            f.write(f"  Static features:  {total_stat:.4f} ({total_stat/total*100:.1f}%)\n")

        print(f"Saved report: {save_path.name}")


# ============================================================
# Main Analysis Function
# ============================================================

def run_full_shap_analysis(
    model: LandslideRiskModel,
    dataset: LandslideDataset,
    test_indices: List[int],
    background_indices: List[int],
    save_dir: Path,
    device: str = 'cuda'
) -> Dict:
    """
    Run complete two-stage SHAP analysis

    Args:
        model: Trained model
        dataset: Dataset
        test_indices: Test sample indices
        background_indices: Background sample indices
        save_dir: Output directory
        device: Device

    Returns:
        results: Dictionary with all SHAP results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Starting Full SHAP Analysis")
    print("="*70)

    # Stage 1: Dynamic features
    dynamic_analyzer = DynamicFeatureSHAP(model, dataset, device)
    dynamic_shap, dynamic_names = dynamic_analyzer.analyze(
        test_indices, background_indices, save_dir
    )

    # Stage 2: Static features
    static_analyzer = StaticFeatureSHAP(dataset, device)

    # Train proxy model
    train_size = int(len(dataset) * 0.8)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    proxy_model = static_analyzer.train_proxy_model(train_indices, val_indices)

    # SHAP analysis
    static_shap, static_names = static_analyzer.analyze(
        test_indices, background_indices, save_dir
    )

    # Stage 3: Integrated analysis
    integrated = IntegratedSHAPAnalysis(
        dynamic_shap, static_shap,
        dynamic_names, static_names,
        dataset
    )

    integrated.plot_integrated_comparison(save_dir / "shap_integrated_comparison.png")
    integrated.generate_report(save_dir / "shap_report.txt")

    print("\n" + "="*70)
    print("SHAP Analysis Complete!")
    print(f"Results saved to: {save_dir}")
    print("="*70)

    return {
        'dynamic_shap': dynamic_shap,
        'static_shap': static_shap,
        'dynamic_names': dynamic_names,
        'static_names': static_names
    }


if __name__ == "__main__":
    print("SHAP Analysis Module")
    print("Use run_full_shap_analysis() to perform complete analysis")
