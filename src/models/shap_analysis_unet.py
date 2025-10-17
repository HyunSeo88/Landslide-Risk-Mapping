"""
SHAP Analysis for Hierarchical GNN-U-Net Model

Two-stage SHAP analysis:
1. Stage 1 (Static Features): Tree-based proxy model for static feature importance
2. Stage 2 (Dynamic Channels): Deep learning explainer for channel-wise contributions

Author: Landslide Risk Analysis Project
Date: 2025-01-16
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_unet import HierarchicalGNNUNet


# ============================================================
# Stage 1: Static Feature Importance Analysis
# ============================================================

class StaticFeatureAnalyzer:
    """
    Analyzes static feature importance using tree-based proxy models
    
    This approach bypasses the complexity of GNN's neighborhood aggregation
    and directly identifies which static features correlate with landslide occurrence.
    
    Args:
        model_type: 'xgboost' or 'random_forest'
        random_state: Random seed
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        random_state: int = 42
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.proxy_model = None
        self.explainer = None
        self.feature_names = None
    
    def fit(
        self,
        static_features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        feature_names: Optional[List[str]] = None
    ):
        """
        Fit proxy model on static features and labels
        
        Args:
            static_features: (N, D) array of static features
            labels: (N,) array of binary labels
            feature_names: List of feature names (optional)
        """
        # Convert to numpy if needed
        if isinstance(static_features, torch.Tensor):
            static_features = static_features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(static_features.shape[1])]
        self.feature_names = feature_names
        
        print(f"Training {self.model_type} proxy model...")
        print(f"  Features: {static_features.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Positive samples: {labels.sum()}/{len(labels)}")
        
        # Build proxy model
        if self.model_type == 'xgboost':
            self.proxy_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.proxy_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.proxy_model.fit(static_features, labels)
        
        print(f"  Training accuracy: {self.proxy_model.score(static_features, labels):.4f}")
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.proxy_model)
        
        print("  SHAP explainer created")
    
    def explain(
        self,
        static_features: Union[np.ndarray, torch.Tensor],
        max_samples: Optional[int] = None
    ) -> shap.Explanation:
        """
        Compute SHAP values for static features
        
        Args:
            static_features: (N, D) array of features to explain
            max_samples: Maximum number of samples to explain (for speed)
        
        Returns:
            shap_values: SHAP explanation object
        """
        if self.explainer is None:
            raise RuntimeError("Must call fit() before explain()")
        
        # Convert to numpy
        if isinstance(static_features, torch.Tensor):
            static_features = static_features.detach().cpu().numpy()
        
        # Subsample if needed
        if max_samples is not None and len(static_features) > max_samples:
            indices = np.random.choice(len(static_features), max_samples, replace=False)
            static_features = static_features[indices]
        
        print(f"Computing SHAP values for {len(static_features)} samples...")
        shap_values = self.explainer(static_features)
        
        return shap_values
    
    def plot_summary(
        self,
        shap_values: shap.Explanation,
        output_path: Optional[str] = None,
        max_display: int = 20
    ):
        """Plot SHAP summary plot"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_bar(
        self,
        shap_values: shap.Explanation,
        output_path: Optional[str] = None,
        max_display: int = 20
    ):
        """Plot SHAP bar plot (feature importance)"""
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Bar plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(
        self,
        shap_values: shap.Explanation
    ) -> pd.DataFrame:
        """
        Get feature importance ranking
        
        Returns:
            df: DataFrame with columns ['feature', 'importance']
        """
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values.values).mean(axis=0)
        
        # Create dataframe
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df


# ============================================================
# Stage 2: Dynamic Channel Contribution Analysis
# ============================================================

class DynamicChannelAnalyzer:
    """
    Analyzes contribution of each input channel (GNN + dynamic features)
    to U-Net predictions using SHAP
    
    Args:
        unet_model: Trained U-Net model (Stage 2)
        channel_names: List of channel names
        device: Device to run on
    """
    
    def __init__(
        self,
        unet_model: nn.Module,
        channel_names: List[str],
        device: torch.device = torch.device('cpu')
    ):
        self.unet_model = unet_model
        self.channel_names = channel_names
        self.device = device
        self.explainer = None
        
        # Move model to device
        self.unet_model = self.unet_model.to(device)
        self.unet_model.eval()
    
    def create_explainer(
        self,
        background_samples: torch.Tensor,
        method: str = 'gradient'
    ):
        """
        Create SHAP explainer for U-Net
        
        Args:
            background_samples: (N, C, H, W) background samples for SHAP
            method: 'gradient' or 'deep'
        """
        print(f"Creating SHAP {method} explainer...")
        print(f"  Background samples: {background_samples.shape}")
        
        background_samples = background_samples.to(self.device)
        
        if method == 'gradient':
            self.explainer = shap.GradientExplainer(
                self.unet_model,
                background_samples
            )
        elif method == 'deep':
            self.explainer = shap.DeepExplainer(
                self.unet_model,
                background_samples
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print("  Explainer created")
    
    def explain_sample(
        self,
        sample: torch.Tensor,
        target_pixel: Optional[Tuple[int, int]] = None,
        slope_pixels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Explain a single sample's prediction
        
        Args:
            sample: (C, H, W) input image
            target_pixel: (row, col) target pixel to explain
                         If None, use max probability pixel in slope_pixels
            slope_pixels: (N, 2) array of slope pixel indices
        
        Returns:
            channel_contributions: (C,) array of SHAP values per channel
        """
        if self.explainer is None:
            raise RuntimeError("Must call create_explainer() first")
        
        # Ensure batch dimension
        if sample.dim() == 3:
            sample = sample.unsqueeze(0)  # (1, C, H, W)
        
        sample = sample.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            pred = self.unet_model(sample)  # (1, 1, H, W)
        
        # Determine target pixel
        if target_pixel is None:
            if slope_pixels is None:
                # Use max probability pixel in entire image
                pred_prob = torch.sigmoid(pred[0, 0])
                max_idx = torch.argmax(pred_prob)
                target_pixel = (max_idx // pred_prob.shape[1], max_idx % pred_prob.shape[1])
            else:
                # Use max probability pixel within slope
                rows, cols = slope_pixels[:, 0], slope_pixels[:, 1]
                pred_prob = torch.sigmoid(pred[0, 0])
                slope_probs = pred_prob[rows, cols]
                max_slope_idx = torch.argmax(slope_probs)
                target_pixel = (rows[max_slope_idx], cols[max_slope_idx])
        
        print(f"  Target pixel: {target_pixel}")
        print(f"  Prediction at target: {torch.sigmoid(pred[0, 0, target_pixel[0], target_pixel[1]]).item():.4f}")
        
        # Compute SHAP values
        print("  Computing SHAP values...")
        shap_values = self.explainer.shap_values(sample)
        
        # Extract SHAP values for target pixel
        # shap_values shape: (1, C, H, W)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For binary classification
        
        channel_contributions = shap_values[0, :, target_pixel[0], target_pixel[1]]
        
        return channel_contributions
    
    def plot_channel_contributions(
        self,
        channel_contributions: np.ndarray,
        output_path: Optional[str] = None
    ):
        """
        Plot channel-wise SHAP contributions
        
        Args:
            channel_contributions: (C,) array of SHAP values
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        colors = ['red' if x < 0 else 'blue' for x in channel_contributions]
        bars = ax.barh(range(len(channel_contributions)), channel_contributions, color=colors)
        
        # Labels
        ax.set_yticks(range(len(channel_contributions)))
        ax.set_yticklabels(self.channel_names)
        ax.set_xlabel('SHAP Value (Contribution to Prediction)')
        ax.set_title('Channel-wise Contributions to Landslide Risk Prediction')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Channel contribution plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_multiple_samples(
        self,
        samples: List[Tuple[torch.Tensor, Optional[np.ndarray]]],
        sample_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze multiple samples and aggregate results
        
        Args:
            samples: List of (image, slope_pixels) tuples
            sample_names: List of sample names
        
        Returns:
            df: DataFrame with channel contributions for each sample
        """
        if sample_names is None:
            sample_names = [f"Sample_{i}" for i in range(len(samples))]
        
        results = []
        
        for i, (image, slope_pixels) in enumerate(samples):
            print(f"\nAnalyzing {sample_names[i]}...")
            contributions = self.explain_sample(image, slope_pixels=slope_pixels)
            results.append(contributions)
        
        # Create dataframe
        df = pd.DataFrame(results, columns=self.channel_names, index=sample_names)
        
        return df


# ============================================================
# Integrated Analysis Pipeline
# ============================================================

def run_full_shap_analysis(
    hierarchical_model: HierarchicalGNNUNet,
    graph_data: torch.Tensor,
    samples_df: pd.DataFrame,
    dynamic_samples: List[torch.Tensor],
    slope_pixels_list: List[np.ndarray],
    feature_names: List[str],
    channel_names: List[str],
    output_dir: str,
    device: torch.device = torch.device('cpu')
):
    """
    Run complete SHAP analysis pipeline
    
    Args:
        hierarchical_model: Trained hierarchical model
        graph_data: Graph data with static features
        samples_df: DataFrame with sample info and labels
        dynamic_samples: List of dynamic raster tensors
        slope_pixels_list: List of slope pixel arrays
        feature_names: Static feature names
        channel_names: Dynamic channel names
        output_dir: Output directory
        device: Device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Running Full SHAP Analysis Pipeline")
    print("="*70)
    
    # ========================================
    # Stage 1: Static Feature Analysis
    # ========================================
    print("\n[Stage 1] Analyzing Static Features...")
    
    static_analyzer = StaticFeatureAnalyzer(model_type='xgboost')
    
    # Prepare data
    static_features = graph_data.x.cpu().numpy()
    labels = samples_df['label'].values
    
    # Fit proxy model
    static_analyzer.fit(static_features, labels, feature_names)
    
    # Compute SHAP values
    shap_values_static = static_analyzer.explain(static_features, max_samples=1000)
    
    # Plot and save
    static_analyzer.plot_summary(
        shap_values_static,
        output_path=str(output_dir / "static_shap_summary.png")
    )
    static_analyzer.plot_bar(
        shap_values_static,
        output_path=str(output_dir / "static_shap_bar.png")
    )
    
    # Save feature importance
    importance_df = static_analyzer.get_feature_importance(shap_values_static)
    importance_df.to_csv(output_dir / "static_feature_importance.csv", index=False)
    print(f"  Feature importance saved: {output_dir / 'static_feature_importance.csv'}")
    
    # ========================================
    # Stage 2: Dynamic Channel Analysis
    # ========================================
    print("\n[Stage 2] Analyzing Dynamic Channels...")
    
    # Extract U-Net model
    unet_model = hierarchical_model.unet
    
    dynamic_analyzer = DynamicChannelAnalyzer(
        unet_model=unet_model,
        channel_names=channel_names,
        device=device
    )
    
    # Create explainer with background samples
    num_background = min(10, len(dynamic_samples))
    background_samples = torch.stack(dynamic_samples[:num_background])
    dynamic_analyzer.create_explainer(background_samples, method='gradient')
    
    # Analyze samples
    num_analyze = min(5, len(dynamic_samples))
    sample_tuples = list(zip(dynamic_samples[:num_analyze], slope_pixels_list[:num_analyze]))
    sample_names = [f"Sample_{i}" for i in range(num_analyze)]
    
    contributions_df = dynamic_analyzer.analyze_multiple_samples(sample_tuples, sample_names)
    contributions_df.to_csv(output_dir / "dynamic_channel_contributions.csv")
    print(f"  Channel contributions saved: {output_dir / 'dynamic_channel_contributions.csv'}")
    
    # Plot average contributions
    avg_contributions = contributions_df.mean(axis=0).values
    dynamic_analyzer.plot_channel_contributions(
        avg_contributions,
        output_path=str(output_dir / "dynamic_channel_contributions.png")
    )
    
    print("\n" + "="*70)
    print("SHAP Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing SHAP Analysis Module")
    print("="*70)
    
    print("\nThis module requires trained models and data.")
    print("Use in inference pipeline with actual trained models.")
    
    print("\n" + "="*70)
    print("SHAP analysis module loaded successfully!")
    print("="*70)

