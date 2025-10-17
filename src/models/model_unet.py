"""
Hierarchical GNN-U-Net Model for Pixel-Level Landslide Risk Prediction

This module implements a two-stage model:
- Stage 1 (Static Susceptibility): GNN predicts slope-level susceptibility
- Stage 2 (Dynamic Risk): U-Net predicts pixel-level risk using GNN output + dynamic features

Architecture:
- StaticSusceptibilityGNN: GraphSAGE/GAT for slope-level susceptibility
- UNetRiskModel: U-Net for pixel-level risk prediction
- HierarchicalGNNUNet: End-to-end integration of both stages

Author: Landslide Risk Analysis Project
Date: 2025-01-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from typing import Dict, Optional, Tuple


# ============================================================
# Stage 1: Static Susceptibility GNN
# ============================================================

class StaticSusceptibilityGNN(nn.Module):
    """
    GNN-based static susceptibility model
    
    Predicts slope-level susceptibility based on static features only.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        num_layers: Number of GNN layers
        gnn_type: 'sage' or 'gat'
        dropout: Dropout rate
        gat_heads: Number of attention heads (GAT only)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        gnn_type: str = 'sage',
        dropout: float = 0.3,
        gat_heads: int = 4
    ):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'sage':
            # GraphSAGE layers
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
        
        elif gnn_type == 'gat':
            # GAT layers
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=gat_heads, dropout=dropout, concat=True)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * gat_heads, hidden_channels, heads=gat_heads,
                           dropout=dropout, concat=True)
                )
            # Last layer: single head
            self.convs.append(
                GATConv(hidden_channels * gat_heads if num_layers > 1 else in_channels,
                       hidden_channels, heads=1, dropout=dropout, concat=False)
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # MLP classifier for susceptibility prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (num_nodes, in_channels) - node features
            edge_index: (2, num_edges) - graph connectivity
            edge_attr: (num_edges, 1) - edge weights (optional)
        
        Returns:
            logits: (num_nodes,) - susceptibility logits
        """
        # GNN encoding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x) if self.gnn_type == 'sage' else F.elu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        logits = self.classifier(x).squeeze(-1)  # (num_nodes,)
        
        return logits


# ============================================================
# Stage 2: U-Net for Dynamic Risk Prediction
# ============================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block with bilinear upsampling (no checkerboard artifacts)
    
    Uses bilinear interpolation instead of transposed convolution
    to avoid checkerboard patterns in the output.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Bilinear upsampling (no checkerboard!)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Additional conv to adjust channels
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Upsampled feature map from decoder
            x2: Skip connection from encoder
        """
        # Upsample
        x1 = self.up(x1)
        x1 = self.conv_reduce(x1)
        
        # Handle size mismatch with padding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetRiskModel(nn.Module):
    """
    U-Net model for pixel-level risk prediction
    
    Takes multi-channel input (GNN susceptibility map + dynamic features)
    and outputs pixel-level risk probability map.
    
    Args:
        in_channels: Number of input channels (1 GNN + N dynamic features)
        base_channels: Base number of channels (doubled at each level)
        depth: Depth of U-Net (number of down/up blocks)
        out_channels: Number of output channels (default: 1 for binary)
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        out_channels: int = 1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(ch, ch * 2))
            ch *= 2
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(Up(ch, ch // 2))
            ch //= 2
        
        # Output convolution
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) - multi-channel input image
        
        Returns:
            logits: (B, 1, H, W) - pixel-level risk logits
        """
        # Encoder with skip connections
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)
        
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        
        # Remove last skip connection (bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder with skip connections
        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[-(i+1)])
        
        # Output
        logits = self.outc(x)
        
        return logits


# ============================================================
# Hierarchical GNN-U-Net Integration
# ============================================================

class HierarchicalGNNUNet(nn.Module):
    """
    End-to-end hierarchical model combining GNN and U-Net
    
    Two-stage architecture:
    1. GNN predicts slope-level static susceptibility
    2. U-Net predicts pixel-level dynamic risk using GNN output + dynamic features
    
    Args:
        static_dim: Static feature dimension
        gnn_hidden: GNN hidden dimension
        gnn_layers: Number of GNN layers
        gnn_type: 'sage' or 'gat'
        gnn_dropout: GNN dropout rate
        gat_heads: Number of GAT heads
        dynamic_channels: Number of dynamic feature channels
        unet_base_channels: U-Net base channels
        unet_depth: U-Net depth
    """
    
    def __init__(
        self,
        static_dim: int,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        gnn_type: str = 'sage',
        gnn_dropout: float = 0.3,
        gat_heads: int = 4,
        dynamic_channels: int = 6,
        unet_base_channels: int = 64,
        unet_depth: int = 4
    ):
        super().__init__()
        
        self.static_dim = static_dim
        self.dynamic_channels = dynamic_channels
        self.gnn_type = gnn_type
        
        # Stage 1: Static Susceptibility GNN
        self.gnn = StaticSusceptibilityGNN(
            in_channels=static_dim,
            hidden_channels=gnn_hidden,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
            gat_heads=gat_heads
        )
        
        # Stage 2: Dynamic Risk U-Net
        # Input channels: 1 (GNN output) + dynamic_channels
        unet_in_channels = 1 + dynamic_channels
        
        self.unet = UNetRiskModel(
            in_channels=unet_in_channels,
            base_channels=unet_base_channels,
            depth=unet_depth,
            out_channels=1
        )
    
    def forward_stage1(
        self,
        static_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stage 1: GNN forward pass only
        
        Args:
            static_x: (num_nodes, static_dim) - static features
            edge_index: (2, num_edges) - graph connectivity
            edge_attr: (num_edges, 1) - edge weights
        
        Returns:
            susceptibility_logits: (num_nodes,) - slope-level susceptibility logits
        """
        return self.gnn(static_x, edge_index, edge_attr)
    
    def forward_stage2(
        self,
        multi_channel_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Stage 2: U-Net forward pass only
        
        Args:
            multi_channel_image: (B, C, H, W) - multi-channel input
                                 (GNN susceptibility map + dynamic features)
        
        Returns:
            risk_logits: (B, 1, H, W) - pixel-level risk logits
        """
        return self.unet(multi_channel_image)
    
    def freeze_gnn(self):
        """
        Freeze GNN parameters for Stage 2 training
        
        Sets requires_grad=False for all GNN parameters so they won't be updated
        during backpropagation.
        """
        for param in self.gnn.parameters():
            param.requires_grad = False
        print("  ✓ GNN parameters frozen (requires_grad=False)")
    
    def unfreeze_gnn(self):
        """
        Unfreeze GNN parameters (for fine-tuning or end-to-end training)
        
        Sets requires_grad=True for all GNN parameters.
        """
        for param in self.gnn.parameters():
            param.requires_grad = True
        print("  ✓ GNN parameters unfrozen (requires_grad=True)")
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        gnn_raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        End-to-end forward pass
        
        Args:
            batch: Dictionary containing:
                - 'static_x': (num_nodes, static_dim) - static features
                - 'edge_index': (2, num_edges) - graph connectivity
                - 'edge_attr': (num_edges, 1) - edge weights
                - 'dynamic_raster': (B, dynamic_channels, H, W) - dynamic features
                - 'raster_converter': RasterConverter object (for GNN -> raster)
                - 'cat_values': (num_nodes,) - slope IDs
                - 'shape': (H, W) - raster shape
                - 'transform': Affine transform
            gnn_raster: (B, 1, H, W) - Pre-computed GNN susceptibility raster (optional)
                        If provided, skip Stage 1
        
        Returns:
            risk_logits: (B, 1, H, W) - pixel-level risk logits
        """
        # Stage 1: GNN (if not pre-computed)
        if gnn_raster is None:
            # Get GNN predictions
            susceptibility_logits = self.forward_stage1(
                batch['static_x'],
                batch['edge_index'],
                batch.get('edge_attr')
            )
            
            # Convert to probabilities
            susceptibility_probs = torch.sigmoid(susceptibility_logits)
            
            # Convert to raster format
            # Note: This is a placeholder - actual implementation needs RasterConverter
            # For now, assume batch provides pre-converted raster
            if 'gnn_susceptibility_raster' in batch:
                gnn_raster = batch['gnn_susceptibility_raster']
            else:
                raise ValueError("Must provide 'gnn_susceptibility_raster' in batch or gnn_raster argument")
        
        # Stage 2: U-Net
        # Concatenate GNN raster with dynamic features
        dynamic_raster = batch['dynamic_raster']  # (B, dynamic_channels, H, W)
        
        multi_channel_input = torch.cat([gnn_raster, dynamic_raster], dim=1)  # (B, 1+C, H, W)
        
        risk_logits = self.forward_stage2(multi_channel_input)
        
        return risk_logits


# ============================================================
# Loss Functions for MIL
# ============================================================

class MILLoss(nn.Module):
    """
    Multiple Instance Learning Loss
    
    Aggregates pixel-level predictions within each slope region
    and computes loss at slope level.
    
    Args:
        aggregation: 'max', 'mean', or 'lse' (log-sum-exp)
        base_loss: Base loss function ('bce' or 'focal')
        pos_weight: Positive class weight for BCE
    """
    
    def __init__(
        self,
        aggregation: str = 'max',
        base_loss: str = 'bce',
        pos_weight: Optional[float] = None
    ):
        super().__init__()
        
        self.aggregation = aggregation
        self.base_loss = base_loss
        
        if base_loss == 'bce':
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight])
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        elif base_loss == 'focal':
            from src.models.model import FocalLoss
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def aggregate_bag(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate logits within a bag (slope region)
        
        Args:
            logits: (num_pixels,) - pixel-level logits in the bag
        
        Returns:
            aggregated: scalar - bag-level representative score
        """
        if self.aggregation == 'max':
            return logits.max()
        elif self.aggregation == 'mean':
            return logits.mean()
        elif self.aggregation == 'lse':
            # Log-sum-exp for smooth max
            return torch.logsumexp(logits, dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def forward(
        self,
        pixel_logits: torch.Tensor,
        zone_patches: torch.Tensor,
        cats: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MIL loss (patch-based) with NoData masking
        
        Args:
            pixel_logits: (B, 1, H, W) - pixel-level predictions
            zone_patches: (B, H, W) - slope ID raster patches
            cats: (B,) - slope IDs for each sample
            labels: (B,) - slope-level labels
        
        Returns:
            loss: scalar
        """
        batch_size = pixel_logits.shape[0]
        representative_scores = []
        
        for i in range(batch_size):
            cat = cats[i].item()
            zone_patch = zone_patches[i]  # (H, W)
            
            # Create valid mask (exclude NoData = 0)
            valid_mask = (zone_patch > 0)
            
            # Create mask for this slope (only valid pixels)
            slope_mask = (zone_patch == cat) & valid_mask
            
            if slope_mask.sum() == 0:
                # No valid pixels in slope - use 0
                representative_scores.append(torch.tensor(0.0, device=pixel_logits.device))
                continue
            
            # Extract logits for slope pixels (only valid, no NoData)
            slope_logits = pixel_logits[i, 0, slope_mask]  # (num_pixels,)
            
            # Aggregate
            representative_score = self.aggregate_bag(slope_logits)
            representative_scores.append(representative_score)
        
        # Stack and compute loss
        representative_scores = torch.stack(representative_scores)  # (B,)
        loss = self.criterion(representative_scores, labels)
        
        return loss


# ============================================================
# Model Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing Hierarchical GNN-U-Net Model")
    print("="*70)
    
    # Dummy parameters
    batch_size = 4
    num_nodes = 100
    num_edges = 500
    static_dim = 21
    dynamic_channels = 6
    H, W = 256, 256
    
    print("\n[1] Testing StaticSusceptibilityGNN")
    gnn_model = StaticSusceptibilityGNN(
        in_channels=static_dim,
        hidden_channels=64,
        num_layers=2,
        gnn_type='sage'
    )
    
    static_x = torch.randn(num_nodes, static_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.rand(num_edges, 1)
    
    susceptibility_logits = gnn_model(static_x, edge_index, edge_attr)
    print(f"  Output shape: {susceptibility_logits.shape}")
    print(f"  Output range: [{susceptibility_logits.min():.3f}, {susceptibility_logits.max():.3f}]")
    
    print("\n[2] Testing UNetRiskModel")
    unet_model = UNetRiskModel(
        in_channels=1 + dynamic_channels,  # 1 GNN + 6 dynamic
        base_channels=64,
        depth=4
    )
    
    multi_channel_input = torch.randn(batch_size, 1 + dynamic_channels, H, W)
    risk_logits = unet_model(multi_channel_input)
    print(f"  Input shape: {multi_channel_input.shape}")
    print(f"  Output shape: {risk_logits.shape}")
    
    print("\n[3] Testing HierarchicalGNNUNet")
    hierarchical_model = HierarchicalGNNUNet(
        static_dim=static_dim,
        gnn_hidden=64,
        gnn_layers=2,
        gnn_type='sage',
        dynamic_channels=dynamic_channels,
        unet_base_channels=64,
        unet_depth=4
    )
    
    # Test Stage 1 only
    print("  Testing Stage 1 (GNN)...")
    susc_logits = hierarchical_model.forward_stage1(static_x, edge_index, edge_attr)
    print(f"    Output shape: {susc_logits.shape}")
    
    # Test Stage 2 only
    print("  Testing Stage 2 (U-Net)...")
    gnn_raster = torch.sigmoid(susc_logits[:10]).reshape(1, 1, 5, 2).expand(batch_size, 1, H, W)
    dynamic_raster = torch.randn(batch_size, dynamic_channels, H, W)
    multi_ch = torch.cat([gnn_raster, dynamic_raster], dim=1)
    risk_out = hierarchical_model.forward_stage2(multi_ch)
    print(f"    Output shape: {risk_out.shape}")
    
    print("\n[4] Testing MILLoss")
    mil_loss_fn = MILLoss(aggregation='max', base_loss='bce', pos_weight=1.0)
    
    # Create dummy slope pixel indices
    slope_pixel_indices = []
    for _ in range(batch_size):
        num_pixels = torch.randint(10, 100, (1,)).item()
        pixels = torch.randint(0, min(H, W), (num_pixels, 2))
        slope_pixel_indices.append(pixels)
    
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    loss = mil_loss_fn(risk_out, slope_pixel_indices, labels)
    print(f"  MIL Loss: {loss.item():.4f}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

