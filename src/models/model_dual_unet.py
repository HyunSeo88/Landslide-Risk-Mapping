"""
Dual-Stream Hierarchical GNN-U-Net for Dynamic Landslide Risk Prediction

This module implements a hierarchical two-stream architecture:
- State Stream (CNN): Static terrain features (texture/geometry)
- Trigger Stream (ConvLSTM): Dynamic temporal features (rainfall, InSAR, NDVI)
- Cross-Attention Fusion: State-Trigger interaction modeling
- Attention-MIL: Interpretable weakly supervised learning

Architecture:
1. GNN provides slope-level context (from Stage 1)
2. State Encoder (CNN) extracts spatial terrain patterns
3. Trigger Encoder (ConvLSTM) models temporal trigger patterns
4. Cross-Attention Fusion combines State + Trigger + GNN context
5. U-Net Decoder generates pixel-level risk map
6. Attention-MIL aggregates pixels to slope-level for weak supervision

Author: Landslide Risk Analysis Project
Date: 2025-01-17
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# ============================================================
# ConvLSTM Components
# ============================================================

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell
    
    Combines CNN spatial feature extraction with LSTM temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        
        # Gates: input, forget, output, cell
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_tensor: (B, C, H, W)
            cur_state: (h_cur, c_cur) each (B, hidden_dim, H, W)
        
        Returns:
            h_next, c_next: Next hidden and cell states
        """
        h_cur, c_cur = cur_state
        
        # Concatenate along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Convolution
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device):
        """Initialize hidden and cell states"""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM
    
    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels (can be list for multiple layers)
        kernel_size: Kernel size
        num_layers: Number of ConvLSTM layers
        batch_first: If True, input is (B, T, C, H, W)
        bias: Whether to use bias
        return_all_layers: Whether to return outputs from all layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        num_layers: int = 2,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # Create ConvLSTM cells
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            input_tensor: (B, T, C, H, W) if batch_first=True
            hidden_state: Initial hidden states (optional)
        
        Returns:
            layer_output_list: List of outputs for each layer
            last_state_list: List of last (h, c) for each layer
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=(h, c)
                )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden_dim, H, W)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int]):
        """Initialize hidden states for all layers"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size, self.cell_list[0].conv.weight.device)
            )
        return init_states


# ============================================================
# Dual-Stream Encoders
# ============================================================

class DoubleConv(nn.Module):
    """Double convolution block"""
    
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
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class StateEncoder(nn.Module):
    """
    CNN-based encoder for static terrain features
    
    Extracts spatial texture and geometry patterns from high-resolution
    static rasters (slope, curvature, TWI, TRI, etc.)
    
    Args:
        in_channels: Number of static feature channels
        base_channels: Base number of channels
        depth: Depth of encoder (number of down blocks)
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4
    ):
        super().__init__()
        
        self.depth = depth
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(ch, ch * 2))
            ch *= 2
        
        self.out_channels = ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: (B, C_static, H, W)
        
        Returns:
            features: (B, out_channels, H', W') - Bottleneck features
            skip_connections: List of skip connection features
        """
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)
        
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        
        # Remove last (bottleneck is not a skip connection)
        features = skip_connections.pop()
        
        return features, skip_connections


class TriggerEncoder(nn.Module):
    """
    ConvLSTM-based encoder for dynamic temporal features
    
    Models temporal patterns in trigger factors (rainfall accumulation,
    InSAR displacement trends, NDVI changes, etc.)
    
    Args:
        in_channels: Number of dynamic feature channels (per timestep)
        hidden_channels: ConvLSTM hidden channels
        num_layers: Number of ConvLSTM layers
        output_channels: Output feature channels (for fusion)
        downsample_factor: Downsampling factor to match State encoder (2^depth)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        output_channels: int = 256,
        downsample_factor: int = 16
    ):
        super().__init__()
        
        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # Downsampling to match State encoder resolution
        # Calculate number of down blocks needed
        num_downs = int(np.log2(downsample_factor))
        
        down_blocks = []
        ch = hidden_channels
        for i in range(num_downs):
            down_blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch * 2),
                    nn.ReLU(inplace=True)
                )
            )
            ch *= 2
        
        self.down_blocks = nn.Sequential(*down_blocks)
        
        # Project to output channels
        self.projection = nn.Sequential(
            nn.Conv2d(ch, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = output_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, T, C_dynamic, H, W) - Temporal sequence
        
        Returns:
            features: (B, out_channels, H', W') - Compressed temporal features (downsampled)
        """
        # ConvLSTM forward
        layer_output_list, last_state_list = self.convlstm(x)
        
        # Use last hidden state (contains compressed temporal information)
        h_n, c_n = last_state_list[-1]  # (B, hidden_channels, H, W)
        
        # Downsample to match State encoder resolution
        h_n = self.down_blocks(h_n)  # (B, hidden_channels * 2^num_downs, H/16, W/16)
        
        # Project to output dimensions
        features = self.projection(h_n)
        
        return features


# ============================================================
# Cross-Attention Fusion
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module
    
    Learns the interaction between State (terrain) and Trigger (dynamics):
    - State attends to Trigger: "Which trigger patterns matter for this terrain?"
    - Trigger attends to State: "Which terrain features amplify this trigger?"
    
    GNN context is injected to provide slope-level susceptibility information.
    
    Args:
        state_channels: State feature channels
        trigger_channels: Trigger feature channels
        num_heads: Number of attention heads
        fusion_type: 'concat' or 'add'
    """
    
    def __init__(
        self,
        state_channels: int,
        trigger_channels: int,
        num_heads: int = 4,
        fusion_type: str = 'concat'
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.fusion_type = fusion_type
        
        # Cross-attention: State -> Trigger
        self.cross_attn_state_to_trigger = nn.MultiheadAttention(
            embed_dim=state_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention: Trigger -> State
        self.cross_attn_trigger_to_state = nn.MultiheadAttention(
            embed_dim=trigger_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Fusion layer
        if fusion_type == 'concat':
            # state_channels + trigger_channels + 1 (GNN)
            fused_channels = state_channels + trigger_channels + 1
        elif fusion_type == 'add':
            # Project to same dimension
            assert state_channels == trigger_channels, "Channels must match for 'add' fusion"
            fused_channels = state_channels + 1  # +1 for GNN
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels // 2, fused_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = fused_channels // 2
    
    def forward(
        self,
        state_features: torch.Tensor,
        trigger_features: torch.Tensor,
        gnn_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state_features: (B, C_state, H, W)
            trigger_features: (B, C_trigger, H, W)
            gnn_context: (B, 1, H, W)
        
        Returns:
            fused_features: (B, out_channels, H, W)
        """
        B, C_state, H, W = state_features.shape
        C_trigger = trigger_features.shape[1]
        
        # Reshape for attention: (B, H*W, C)
        state_flat = state_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C_state)
        trigger_flat = trigger_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C_trigger)
        
        # Cross-attention: State queries Trigger
        state_to_trigger, _ = self.cross_attn_state_to_trigger(
            query=state_flat,
            key=trigger_flat,
            value=trigger_flat
        )  # (B, H*W, C_state)
        
        # Cross-attention: Trigger queries State
        trigger_to_state, _ = self.cross_attn_trigger_to_state(
            query=trigger_flat,
            key=state_flat,
            value=state_flat
        )  # (B, H*W, C_trigger)
        
        # Reshape back to spatial: (B, C, H, W)
        state_to_trigger = state_to_trigger.permute(0, 2, 1).reshape(B, C_state, H, W)
        trigger_to_state = trigger_to_state.permute(0, 2, 1).reshape(B, C_trigger, H, W)
        
        # Fuse with GNN context
        if self.fusion_type == 'concat':
            fused = torch.cat([state_to_trigger, trigger_to_state, gnn_context], dim=1)
        elif self.fusion_type == 'add':
            fused = torch.cat([state_to_trigger + trigger_to_state, gnn_context], dim=1)
        
        # Final fusion convolution
        fused_features = self.fusion_conv(fused)
        
        return fused_features


# ============================================================
# U-Net Decoder
# ============================================================

class Up(nn.Module):
    """Upscaling block with bilinear upsampling"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # After concat: (skip_channels + in_channels // 2)
        self.conv = DoubleConv(skip_channels + in_channels // 2, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature from decoder (B, in_channels, H, W)
            x2: Skip connection from encoder (B, skip_channels, 2*H, 2*W)
        """
        x1 = self.up(x1)
        x1 = self.conv_reduce(x1)
        
        # Handle size mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DualStreamDecoder(nn.Module):
    """
    U-Net style decoder with skip connections from State encoder
    
    Args:
        fused_channels: Channels from fusion module
        state_skip_channels: List of skip connection channels from State encoder
        depth: Decoder depth
    """
    
    def __init__(
        self,
        fused_channels: int,
        state_skip_channels: List[int],
        depth: int = 4
    ):
        super().__init__()
        
        self.depth = depth
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        ch = fused_channels
        
        for i in range(depth):
            skip_ch = state_skip_channels[-(i+1)]
            out_ch = ch // 2
            self.up_blocks.append(Up(in_channels=ch, skip_channels=skip_ch, out_channels=out_ch))
            ch = out_ch
        
        # Output convolution
        self.outc = nn.Conv2d(ch, 1, kernel_size=1)
        
        self.intermediate_channels = ch
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (B, fused_channels, H', W') - Fused features
            skip_connections: List of skip connection features from State encoder
        
        Returns:
            logits: (B, 1, H, W) - Pixel-level risk logits
            intermediate_features: (B, C, H, W) - Features before final conv (for Attention-MIL)
        """
        # Decoder with skip connections
        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[-(i+1)])
        
        intermediate_features = x
        
        # Output
        logits = self.outc(x)
        
        return logits, intermediate_features


# ============================================================
# Attention-based MIL Loss
# ============================================================

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning Loss
    
    Unlike max/mean pooling, this module learns to weight pixels based on
    intermediate features, providing interpretability and better performance.
    
    Args:
        feature_dim: Channel dimension of intermediate features
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
    
    def forward(
        self,
        pixel_logits: torch.Tensor,
        intermediate_features: torch.Tensor,
        zone_patches: torch.Tensor,
        cats: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Attention-MIL loss
        
        Args:
            pixel_logits: (B, 1, H, W) - Pixel-level predictions
            intermediate_features: (B, C, H, W) - Features before final conv
            zone_patches: (B, H, W) - Slope ID raster patches
            cats: (B,) - Slope IDs
            labels: (B,) - Slope-level labels
        
        Returns:
            loss: Scalar loss
            attention_maps: (B, H, W) - Attention weights for interpretability
        """
        batch_size = pixel_logits.shape[0]
        device = pixel_logits.device
        
        # Compute attention scores
        attention_scores = self.attention_net(intermediate_features)  # (B, 1, H, W)
        
        representative_logits = []
        attention_maps_list = []
        
        for i in range(batch_size):
            cat = cats[i].item()
            zone_patch = zone_patches[i]  # (H, W)
            
            # Create valid mask (exclude NoData = 0)
            valid_mask = (zone_patch > 0).float()
            
            # Create mask for this slope
            slope_mask = ((zone_patch == cat) & (zone_patch > 0)).float()  # (H, W)
            
            if slope_mask.sum() == 0:
                # No valid pixels
                representative_logits.append(torch.tensor(0.0, device=device))
                attention_maps_list.append(torch.zeros_like(slope_mask))
                continue
            
            # Extract slope pixels
            slope_attention = attention_scores[i, 0] * slope_mask  # (H, W)
            slope_logits = pixel_logits[i, 0] * slope_mask  # (H, W)
            
            # Softmax attention weights (only over slope pixels)
            # Add large negative value to non-slope pixels
            slope_attention_masked = slope_attention.masked_fill(slope_mask == 0, -1e9)
            attention_weights = F.softmax(slope_attention_masked.flatten(), dim=0).reshape_as(slope_mask)
            
            # Weighted aggregation
            bag_logit = (slope_logits * attention_weights * slope_mask).sum()
            
            representative_logits.append(bag_logit)
            attention_maps_list.append(attention_weights)
        
        # Stack
        representative_logits = torch.stack(representative_logits)  # (B,)
        attention_maps = torch.stack(attention_maps_list)  # (B, H, W)
        
        # BCE loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(representative_logits, labels)
        
        return loss, attention_maps


# ============================================================
# Complete Dual-Stream Model
# ============================================================

class DualHierarchicalGNNUNet(nn.Module):
    """
    Complete Dual-Stream Hierarchical GNN-U-Net Model
    
    Architecture:
    1. GNN provides slope-level context (from Stage 1, frozen)
    2. State Encoder (CNN) extracts spatial patterns from static rasters
    3. Trigger Encoder (ConvLSTM) models temporal patterns from dynamic sequences
    4. Cross-Attention Fusion combines State + Trigger + GNN context
    5. U-Net Decoder generates pixel-level risk map
    6. Attention-MIL provides weak supervision with interpretability
    
    Args:
        static_dim: Static feature dimension (for GNN, from Stage 1)
        gnn_hidden: GNN hidden channels
        gnn_layers: GNN layers
        gnn_type: GNN type ('sage' or 'gat')
        gnn_dropout: GNN dropout
        gat_heads: GAT heads
        state_channels: Number of state feature channels (static rasters)
        trigger_channels: Number of trigger feature channels per timestep
        state_base_channels: Base channels for State encoder
        trigger_hidden_channels: ConvLSTM hidden channels
        trigger_num_layers: ConvLSTM layers
        fusion_num_heads: Cross-attention heads
        decoder_depth: Decoder depth
    """
    
    def __init__(
        self,
        # GNN parameters (from Stage 1)
        static_dim: int,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        gnn_type: str = 'sage',
        gnn_dropout: float = 0.3,
        gat_heads: int = 4,
        # Dual-stream parameters
        state_channels: int = 5,
        trigger_channels: int = 3,
        state_base_channels: int = 32,
        trigger_hidden_channels: int = 64,
        trigger_num_layers: int = 2,
        fusion_num_heads: int = 4,
        decoder_depth: int = 4
    ):
        super().__init__()
        
        # Import GNN from Stage 1
        from src.models.model_unet import StaticSusceptibilityGNN
        
        self.gnn = StaticSusceptibilityGNN(
            in_channels=static_dim,
            hidden_channels=gnn_hidden,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
            gat_heads=gat_heads
        )
        
        # Dual-stream encoders
        self.state_encoder = StateEncoder(
            in_channels=state_channels,
            base_channels=state_base_channels,
            depth=decoder_depth
        )
        
        # Calculate downsample factor (2^depth)
        downsample_factor = 2 ** decoder_depth
        
        self.trigger_encoder = TriggerEncoder(
            in_channels=trigger_channels,
            hidden_channels=trigger_hidden_channels,
            num_layers=trigger_num_layers,
            output_channels=self.state_encoder.out_channels,  # Match State output
            downsample_factor=downsample_factor
        )
        
        # Fusion module
        self.fusion = CrossAttentionFusion(
            state_channels=self.state_encoder.out_channels,
            trigger_channels=self.trigger_encoder.out_channels,
            num_heads=fusion_num_heads,
            fusion_type='concat'
        )
        
        # Decoder
        # Calculate skip connection channels
        ch = state_base_channels
        skip_channels = [ch]
        for i in range(decoder_depth):
            ch *= 2
            skip_channels.append(ch)
        skip_channels = skip_channels[:-1]  # Remove bottleneck
        
        self.decoder = DualStreamDecoder(
            fused_channels=self.fusion.out_channels,
            state_skip_channels=skip_channels,
            depth=decoder_depth
        )
        
        # Attention-MIL
        self.attention_mil = AttentionMIL(
            feature_dim=self.decoder.intermediate_channels
        )
    
    def freeze_gnn(self):
        """Freeze GNN parameters (should be done after loading Stage 1 weights)"""
        for param in self.gnn.parameters():
            param.requires_grad = False
        print("  ✓ GNN parameters frozen (requires_grad=False)")
    
    def unfreeze_gnn(self):
        """Unfreeze GNN parameters"""
        for param in self.gnn.parameters():
            param.requires_grad = True
        print("  ✓ GNN parameters unfrozen (requires_grad=True)")
    
    def forward(
        self,
        gnn_patch: torch.Tensor,
        state_patch: torch.Tensor,
        trigger_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            gnn_patch: (B, 1, H, W) - Pre-computed GNN susceptibility
            state_patch: (B, C_state, H, W) - Static terrain features
            trigger_sequence: (B, T, C_trigger, H, W) - Temporal dynamic features
        
        Returns:
            pixel_logits: (B, 1, H, W) - Pixel-level risk logits
            intermediate_features: (B, C, H, W) - Intermediate features for Attention-MIL
        """
        # State encoding
        state_features, skip_connections = self.state_encoder(state_patch)
        
        # Trigger encoding (temporal)
        trigger_features = self.trigger_encoder(trigger_sequence)
        
        # Downsample GNN context to match encoded feature resolution
        # state_features and trigger_features are at H/16, W/16
        # gnn_patch is at H, W - need to downsample by factor of 16
        gnn_context_downsampled = F.adaptive_avg_pool2d(
            gnn_patch, 
            (state_features.size(2), state_features.size(3))
        )
        
        # Fusion with GNN context
        fused_features = self.fusion(state_features, trigger_features, gnn_context_downsampled)
        
        # Decoding
        pixel_logits, intermediate_features = self.decoder(fused_features, skip_connections)
        
        return pixel_logits, intermediate_features
    
    def compute_loss(
        self,
        pixel_logits: torch.Tensor,
        intermediate_features: torch.Tensor,
        zone_patches: torch.Tensor,
        cats: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Attention-MIL loss
        
        Returns:
            loss: Scalar loss
            attention_maps: (B, H, W) - Attention weights
        """
        return self.attention_mil(
            pixel_logits,
            intermediate_features,
            zone_patches,
            cats,
            labels
        )


# ============================================================
# Model Testing
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("="*70)
    print("Testing Dual-Stream Hierarchical GNN-U-Net")
    print("="*70)
    
    # Dummy parameters
    batch_size = 2
    time_steps = 5
    height, width = 128, 128
    
    # Model
    model = DualHierarchicalGNNUNet(
        static_dim=21,
        state_channels=5,
        trigger_channels=3,
        state_base_channels=32,
        trigger_hidden_channels=64,
        decoder_depth=4
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy inputs
    gnn_patch = torch.randn(batch_size, 1, height, width)
    state_patch = torch.randn(batch_size, 5, height, width)
    trigger_sequence = torch.randn(batch_size, time_steps, 3, height, width)
    
    # Forward
    pixel_logits, intermediate_features = model(gnn_patch, state_patch, trigger_sequence)
    
    print(f"\nForward pass:")
    print(f"  Input shapes:")
    print(f"    GNN patch: {gnn_patch.shape}")
    print(f"    State patch: {state_patch.shape}")
    print(f"    Trigger sequence: {trigger_sequence.shape}")
    print(f"  Output shapes:")
    print(f"    Pixel logits: {pixel_logits.shape}")
    print(f"    Intermediate features: {intermediate_features.shape}")
    
    # Test Attention-MIL
    zone_patches = torch.randint(0, 100, (batch_size, height, width))
    cats = torch.tensor([50, 75])
    labels = torch.tensor([0.0, 1.0])
    
    loss, attention_maps = model.compute_loss(
        pixel_logits,
        intermediate_features,
        zone_patches,
        cats,
        labels
    )
    
    print(f"\nAttention-MIL loss:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Attention maps shape: {attention_maps.shape}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

