"""
Loss functions for Hierarchical Fusion Model

- HierarchicalMILLoss: Global Max Pooling-based MIL with Focal Loss
- HierarchicalCombinedLoss: MIL Loss + KFS Prior Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HierarchicalMILLoss(nn.Module):
    """
    Global Max Pooling-based Multiple Instance Learning Loss

    Strategy:
    1. Extract per-slope max probability via Global Max Pooling
    2. Compare with slope-level binary labels
    3. Use Focal Loss to handle hard examples

    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Balancing weight for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pixel_logits: torch.Tensor,
                slope_labels: torch.Tensor,
                slope_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_logits: (B, 1, H, W) - model prediction before sigmoid
            slope_labels: (B,) - binary labels (0/1) per slope
            slope_masks: (B, 1, H, W) - slope interior masks

        Returns:
            {
                'loss': scalar focal loss,
                'slope_probs': (B,) - aggregated probabilities per slope,
                'focal_weight_mean': scalar - mean focal weight for monitoring
            }
        """
        B = pixel_logits.size(0)

        # Check input for NaN
        if torch.isnan(pixel_logits).any():
            print(f"[NaN Detection] pixel_logits contains NaN before masking!")
            print(f"  NaN count: {torch.isnan(pixel_logits).sum().item()}/{pixel_logits.numel()}")
            if not torch.isnan(pixel_logits).all():
                valid_vals = pixel_logits[~torch.isnan(pixel_logits)]
                print(f"  Valid values: min={valid_vals.min():.4f}, max={valid_vals.max():.4f}")

        # 1. Apply mask to extract slope interior pixels
        # Mask invalid pixels with large negative value for max pooling
        # Use -1e4 instead of -1e9 to avoid float16 overflow in AMP
        masked_logits = pixel_logits.clone()
        masked_logits[slope_masks == 0] = -1e4  # Large negative value (compatible with float16)

        # 2. Global Max Pooling per slope
        # Extract maximum logit within each slope
        slope_logits = masked_logits.view(B, 1, -1).max(dim=2)[0]  # (B, 1)
        slope_logits = slope_logits.squeeze(1)  # (B,)

        # 3. Focal Loss with logits (AMP-safe)
        # FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        targets = slope_labels.float()  # (B,)

        # Convert logits to probabilities for p_t calculation
        slope_probs = torch.sigmoid(slope_logits)  # (B,)

        # p_t: predicted probability for true class
        p_t = slope_probs * targets + (1 - slope_probs) * (1 - targets)

        # α_t: balancing weight for positive class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t).pow(self.gamma)

        # Binary cross-entropy with logits (AMP-safe)
        bce = F.binary_cross_entropy_with_logits(
            slope_logits, targets, reduction='none'
        )

        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce
        loss = focal_loss.mean()

        # Check for NaN
        if torch.isnan(loss):
            print(f"[NaN Detection] MIL Loss is NaN!")
            print(f"  slope_logits: min={slope_logits.min():.4f}, max={slope_logits.max():.4f}")
            print(f"  slope_probs: min={slope_probs.min():.4f}, max={slope_probs.max():.4f}")
            print(f"  focal_weight: min={focal_weight.min():.4f}, max={focal_weight.max():.4f}")
            print(f"  bce: min={bce.min():.4f}, max={bce.max():.4f}")

        return {
            'loss': loss,
            'slope_probs': slope_probs.detach(),
            'focal_weight_mean': focal_weight.mean().item()
        }


class HierarchicalCombinedLoss(nn.Module):
    """
    Combined Loss = MIL Loss + Alpha Detail Loss

    Total Loss = L_MIL(final_output) + λ * L_alpha_detail(alpha_map, KFS_prior)

    Key Design:
    - MIL Loss: Applied to final_output (blended prediction)
    - Alpha Detail Loss: Gradient-based spatial pattern matching (Sobel)
    - No warm-up needed (static weight)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 alpha_loss_weight: float = 0.01, use_alpha_loss: bool = True):
        """
        Args:
            alpha: Focal loss balancing weight
            gamma: Focal loss focusing parameter
            alpha_loss_weight: Weight for alpha detail loss (λ)
            use_alpha_loss: Whether to use alpha regularization
        """
        super().__init__()
        self.mil_loss = HierarchicalMILLoss(alpha, gamma)
        self.alpha_loss_weight = alpha_loss_weight
        self.use_alpha_loss = use_alpha_loss

        # Pre-define Sobel filters as buffers (avoid recomputation)
        # Shape: (out_channels=1, in_channels=1, kernel_h=3, kernel_w=3)
        # Sobel X: horizontal edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ], dtype=torch.float32).unsqueeze(0))  # (1, 1, 3, 3)

        # Sobel Y: vertical edge detection
        self.register_buffer('sobel_y', torch.tensor([
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]]
        ], dtype=torch.float32).unsqueeze(0))  # (1, 1, 3, 3)

    def forward(self, outputs: Dict[str, torch.Tensor],
                slope_labels: torch.Tensor,
                slope_masks: torch.Tensor,
                kfs_prior: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict from model.forward()
                - 'final_output': (B, 1, H, W) - blended prediction logits
                - 'alpha_map': (B, 1, H, W) - dynamic blending gate [0, 1]
            slope_labels: (B,) - binary labels
            slope_masks: (B, 1, H, W) - slope interior masks
            kfs_prior: (B, 1, H, W) - KFS prior probabilities [0, 1]

        Returns:
            {
                'total_loss': scalar,
                'mil_loss': scalar,
                'alpha_loss': scalar,
                'slope_probs': (B,),
                'focal_weight_mean': scalar
            }
        """
        # 1. MIL Loss on final_output (blended prediction)
        mil_dict = self.mil_loss(
            outputs['final_output'],
            slope_labels,
            slope_masks
        )

        # 2. Alpha Detail Loss: Gradient-based spatial pattern matching
        if self.use_alpha_loss:
            alpha_loss = self._compute_alpha_detail_loss(
                outputs['alpha_map'],
                kfs_prior,
                slope_masks
            )
        else:
            alpha_loss = torch.tensor(0.0, device=slope_labels.device)

        # 3. Total Loss
        total_loss = mil_dict['loss'] + self.alpha_loss_weight * alpha_loss

        return {
            'total_loss': total_loss,
            'mil_loss': mil_dict['loss'],
            'alpha_loss': alpha_loss,
            'alpha_loss_weight': self.alpha_loss_weight,  # For monitoring
            'slope_probs': mil_dict['slope_probs'],
            'focal_weight_mean': mil_dict['focal_weight_mean']
        }

    def _compute_alpha_detail_loss(self, alpha_map: torch.Tensor,
                                   kfs_prior: torch.Tensor,
                                   slope_masks: torch.Tensor) -> torch.Tensor:
        """
        Encourage alpha_map to follow KFS prior's SPATIAL PATTERN (not absolute values)

        Method: Compare gradients (Sobel) between alpha_map and kfs_prior
        - Sobel extracts spatial structure (edges, transitions)
        - Loss = MSE between gradient magnitudes
        - Ignores absolute value differences, focuses on spatial detail

        Args:
            alpha_map: (B, 1, H, W) - predicted blending gate [0, 1]
            kfs_prior: (B, 1, H, W) - KFS prior probabilities [0, 1]
            slope_masks: (B, 1, H, W) - slope interior masks

        Returns:
            gradient_loss: scalar - spatial pattern matching loss
        """
        # Defensive: Sanitize KFS prior to [0, 1] range
        # Handles NoData, negative values, and extreme values from data loading
        kfs_prior = torch.clamp(kfs_prior, 0.0, 1.0)
        kfs_prior = torch.nan_to_num(kfs_prior, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure Sobel filters match input dtype
        sobel_x = self.sobel_x.to(dtype=alpha_map.dtype, device=alpha_map.device)
        sobel_y = self.sobel_y.to(dtype=alpha_map.dtype, device=alpha_map.device)

        # Compute gradients for alpha_map
        # Padding=1 with kernel_size=3 maintains spatial dimensions
        alpha_grad_x = F.conv2d(alpha_map, sobel_x, padding=1)  # (B, 1, H, W)
        alpha_grad_y = F.conv2d(alpha_map, sobel_y, padding=1)  # (B, 1, H, W)
        alpha_grad_mag = torch.sqrt(alpha_grad_x**2 + alpha_grad_y**2 + 1e-8)  # Gradient magnitude

        # Compute gradients for kfs_prior
        kfs_grad_x = F.conv2d(kfs_prior, sobel_x, padding=1)
        kfs_grad_y = F.conv2d(kfs_prior, sobel_y, padding=1)
        kfs_grad_mag = torch.sqrt(kfs_grad_x**2 + kfs_grad_y**2 + 1e-8)

        # Additional safety: handle any remaining NaN from gradient computation
        kfs_grad_mag = torch.nan_to_num(kfs_grad_mag, nan=0.0)

        # Gradient magnitude difference (spatial structure loss)
        gradient_diff = (alpha_grad_mag - kfs_grad_mag) ** 2  # (B, 1, H, W)

        # Apply slope mask (only compute inside slope interior)
        gradient_diff_masked = gradient_diff * slope_masks

        # Average over valid pixels
        num_valid = slope_masks.sum().clamp(min=1)
        gradient_loss = gradient_diff_masked.sum() / num_valid

        # Check for NaN
        if torch.isnan(gradient_loss):
            print(f"[NaN Detection] Alpha Detail Loss is NaN!")
            print(f"  alpha_map: min={alpha_map.min():.4f}, max={alpha_map.max():.4f}")
            print(f"  kfs_prior: min={kfs_prior.min():.4f}, max={kfs_prior.max():.4f}")
            print(f"  alpha_grad_mag: min={alpha_grad_mag.min():.4f}, max={alpha_grad_mag.max():.4f}")
            print(f"  kfs_grad_mag: min={kfs_grad_mag.min():.4f}, max={kfs_grad_mag.max():.4f}")
            print(f"  num_valid: {num_valid.item()}")
            return torch.tensor(0.0, device=gradient_loss.device, dtype=gradient_loss.dtype)

        return gradient_loss


class FocalLoss(nn.Module):
    """
    Standalone Focal Loss for binary classification

    Can be used as alternative to HierarchicalMILLoss
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Balancing weight for positive class
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B,) or (B, 1) - logits or probabilities
            targets: (B,) or (B, 1) - binary labels (0/1)

        Returns:
            loss: scalar or (B,) depending on reduction
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Convert to probabilities if needed
        if inputs.min() < 0 or inputs.max() > 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        targets = targets.float()

        # p_t: predicted probability for true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # α_t: balancing weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term
        focal_weight = (1 - p_t).pow(self.gamma)

        # BCE
        bce = F.binary_cross_entropy(probs, targets, reduction='none')

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
