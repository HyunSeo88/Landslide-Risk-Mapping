"""
Hierarchical Fusion Model for Landslide Risk Prediction

3-Encoder Architecture:
- Encoder-S: Static terrain features (ResNet-style with skip connections)
- Encoder-D: Dynamic temporal features (ConvLSTM)
- Encoder-G: GNN 128d embeddings (context refinement)

2-Stage Fusion:
- SD Fusion: Cross-Attention between Static and Dynamic
- Total Fusion: SD + GNN → Total Bottleneck

Decoder:
- U-Net with skip connections from Encoder-S only

Final Calibration:
- KFS Prior integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ChannelAttentionBlock(nn.Module):
    """
    Channel Attention Block (CAB)

    Learns channel-wise importance weights and applies them to input features.

    Architecture:
        1. Global Average Pooling → (B, C, H, W) → (B, C, 1, 1)
        2. FC → ReLU → FC → Sigmoid → (B, C, 1, 1)
        3. Multiply attention weights with input

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for bottleneck (default: 16)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        reduced_channels = max(in_channels // reduction_ratio, 4)  # At least 4 channels

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) → (B, C, 1, 1)
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),  # Expansion
            nn.Sigmoid()  # Attention weights [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W) - channel-wise weighted features
        """
        attention_weights = self.attention(x)  # (B, C, 1, 1)
        return x * attention_weights  # Broadcast and multiply


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm"""

    def __init__(self, in_channels: int, out_channels: int,
                 num_groups: int = 8, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = F.relu(out, inplace=True)

        return out


class StaticEncoder(nn.Module):
    """
    Static terrain feature encoder

    Input: (B, 23, H, W) - 8 terrain (w/ aspect split) + 12 binary + 1 accm + 1 NDVI + 1 mask
    Output:
        - bottleneck: (B, 256, H/16, W/16)
        - skip_connections: [(B, 128, H/8, W/8), (B, 64, H/4, W/4), (B, 32, H/2, W/2)]
    """

    def __init__(self, config: Dict):
        super().__init__()

        in_channels = config['in_channels']
        hidden_channels = config['hidden_channels']  # [32, 64, 128, 256]
        use_residual = config.get('use_residual', True)
        num_groups = config.get('num_groups', 8)
        dropout = config.get('dropout', 0.1)

        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups, hidden_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Downsampling blocks with skip connections
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for i in range(len(hidden_channels)):
            in_ch = hidden_channels[i-1] if i > 0 else hidden_channels[0]
            out_ch = hidden_channels[i]

            if use_residual:
                block = ResidualBlock(in_ch, out_ch, num_groups, dropout)
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect'),
                    nn.GroupNorm(num_groups, out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout)
                )

            self.down_blocks.append(block)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 23, H, W)

        Returns:
            bottleneck: (B, 256, H/16, W/16)
            skips: [(B, 128, H/8, W/8), (B, 64, H/4, W/4), (B, 32, H/2, W/2)]
        """
        x = self.init_conv(x)  # (B, 32, H, W)

        skip_connections = []

        for i, block in enumerate(self.down_blocks):
            x = block(x)

            # Store skip connections (except bottleneck)
            if i < len(self.down_blocks) - 1:
                skip_connections.append(x)
                x = self.pool(x)

        # Reverse skip order: [256→128, 128→64, 64→32]
        skip_connections.reverse()

        bottleneck = x

        return bottleneck, skip_connections


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for temporal modeling"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        # Gates: input, forget, output, cell
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            padding_mode='reflect'
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, input_dim, H, W)
            h: (B, hidden_dim, H, W)
            c: (B, hidden_dim, H, W)

        Returns:
            h_next: (B, hidden_dim, H, W)
            c_next: (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class DynamicEncoder(nn.Module):
    """
    Dynamic temporal feature encoder with ConvLSTM

    Input: (B, T=5, 3, H, W) - temporal rainfall sequences
    Output: (B, 256, H/16, W/16) - temporal bottleneck
    """

    def __init__(self, config: Dict):
        super().__init__()

        in_channels = config['in_channels']  # 3 per timestep
        hidden_channels = config['hidden_channels']  # [32, 64, 128, 256]
        convlstm_layers = config.get('convlstm_layers', 2)
        use_residual = config.get('use_residual', False)  # Spatial residual connection option
        use_temporal_residual = config.get('use_temporal_residual', False)  # Temporal residual connection
        num_groups = config.get('num_groups', 8)
        dropout = config.get('dropout', 0.1)

        self.use_temporal_residual = use_temporal_residual

        # Initial spatial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups, hidden_channels[0]),
            nn.ReLU(inplace=True)
        )

        # ConvLSTM layers
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(hidden_channels[0], hidden_channels[0])
            for _ in range(convlstm_layers)
        ])

        # Spatial downsampling (after temporal processing)
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for i in range(len(hidden_channels)):
            in_ch = hidden_channels[i-1] if i > 0 else hidden_channels[0]
            out_ch = hidden_channels[i]

            if use_residual:
                block = ResidualBlock(in_ch, out_ch, num_groups, dropout)
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect'),
                    nn.GroupNorm(num_groups, out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout)
                )

            self.down_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T=5, 3, H, W)

        Returns:
            bottleneck: (B, 256, H/16, W/16)
        """
        B, T, C, H, W = x.shape

        # Process each timestep through ConvLSTM
        h_states = []
        c_states = []

        # Initialize hidden states
        for _ in self.convlstm_cells:
            h_states.append(torch.zeros(B, self.init_conv[0].out_channels, H, W,
                                       device=x.device, dtype=x.dtype))
            c_states.append(torch.zeros(B, self.init_conv[0].out_channels, H, W,
                                       device=x.device, dtype=x.dtype))

        # Process temporal sequence
        last_xt = None  # Store last timestep input
        for t in range(T):
            xt = x[:, t]  # (B, 3, H, W)
            xt = self.init_conv(xt)  # (B, 32, H, W)

            # Pass through ConvLSTM layers
            for i, cell in enumerate(self.convlstm_cells):
                h_states[i], c_states[i] = cell(xt if i == 0 else h_states[i-1],
                                                h_states[i], c_states[i])

            # Keep track of last timestep
            if t == T - 1:
                last_xt = xt

        # Use final hidden state (temporal summary)
        h_final = h_states[-1]  # (B, 32, H, W)

        # Temporal residual connection: h_final + last_xt
        # h_final: accumulated temporal information (5-day summary)
        # last_xt: most recent timestep information (current rainfall intensity)
        if self.use_temporal_residual:
            out = h_final + last_xt  # Element-wise addition
        else:
            out = h_final

        # Spatial downsampling
        for i, block in enumerate(self.down_blocks):
            out = block(out)
            if i < len(self.down_blocks) - 1:
                out = self.pool(out)

        return out


class GNNContextEncoder(nn.Module):
    """
    GNN context encoder

    Refines 128d GNN embeddings to spatial context map

    Input: (B, 128, H, W) - rasterized GNN embeddings
    Output: (B, 256, H/16, W/16) - spatial context
    """

    def __init__(self, config: Dict):
        super().__init__()

        in_channels = config['in_channels']  # 128
        hidden_channels = config['hidden_channels']  # [64, 128, 256]
        output_channels = config['output_channels']  # 256
        num_groups = config.get('num_groups', 8)
        dropout = config.get('dropout', 0.1)

        layers = []
        channels = [in_channels] + hidden_channels

        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2,
                         padding=1, padding_mode='reflect'),
                nn.GroupNorm(num_groups, channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ])

        # Final conv to match output channels
        if hidden_channels[-1] != output_channels:
            layers.append(nn.Conv2d(hidden_channels[-1], output_channels, 1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 128, H, W)

        Returns:
            context: (B, 256, H/16, W/16)
        """
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention fusion between Static and Dynamic encoders

    Q: Static bottleneck
    K, V: Dynamic bottleneck
    """

    def __init__(self, config: Dict):
        super().__init__()

        dim = config['dim']  # 256
        num_heads = config['num_heads']  # 8
        dropout = config.get('dropout', 0.1)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, 256, H, W) - Static bottleneck
            key_value: (B, 256, H, W) - Dynamic bottleneck

        Returns:
            fused: (B, 256, H, W)
            attention_weights: (B, num_heads, HW, HW)
        """
        B, C, H, W = query.shape

        # Reshape to sequence format
        q = query.flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = key_value.flatten(2).transpose(1, 2)  # (B, HW, C)

        # Project
        q = self.q_proj(q)  # (B, HW, C)
        k = self.k_proj(kv)  # (B, HW, C)
        v = self.v_proj(kv)  # (B, HW, C)

        # Multi-head split
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, head_dim)
        k = k.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.clone()
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v  # (B, heads, HW, head_dim)
        out = out.transpose(1, 2).reshape(B, H*W, C)  # (B, HW, C)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual + LayerNorm
        out = self.norm(out + q.transpose(1, 2).reshape(B, H*W, C))

        # Reshape back to spatial
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out, attn_weights


class TotalFusion(nn.Module):
    """
    Total fusion: SD + GNN → Total Bottleneck
    """

    def __init__(self, config: Dict):
        super().__init__()

        sd_dim = config['sd_dim']  # 256
        gnn_dim = config['gnn_dim']  # 256
        output_dim = config['output_dim']  # 256
        fusion_method = config.get('fusion_method', 'concat_conv')

        self.fusion_method = fusion_method

        if fusion_method == 'concat_conv':
            self.fusion = nn.Sequential(
                nn.Conv2d(sd_dim + gnn_dim, output_dim, 3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(8, output_dim),
                nn.ReLU(inplace=True)
            )
        elif fusion_method == 'add':
            assert sd_dim == gnn_dim == output_dim
            self.fusion = nn.Identity()
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Conv2d(sd_dim + gnn_dim, output_dim, 1),
                nn.Sigmoid()
            )
            self.fusion = nn.Conv2d(sd_dim + gnn_dim, output_dim, 3,
                                   padding=1, padding_mode='reflect')

    def forward(self, sd_fused: torch.Tensor, gnn_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sd_fused: (B, 256, H, W)
            gnn_context: (B, 256, H, W)

        Returns:
            total_bottleneck: (B, 256, H, W)
        """
        if self.fusion_method == 'add':
            return sd_fused + gnn_context
        elif self.fusion_method == 'concat_conv':
            concatenated = torch.cat([sd_fused, gnn_context], dim=1)
            return self.fusion(concatenated)
        elif self.fusion_method == 'gated':
            concatenated = torch.cat([sd_fused, gnn_context], dim=1)
            gate = self.gate(concatenated)
            features = self.fusion(concatenated)
            return gate * features
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class HierarchicalDecoder(nn.Module):
    """
    U-Net decoder with skip connections from Encoder-S only

    Input: Total bottleneck (B, 256, H/16, W/16)
    Skip connections: [(B, 128, H/8, W/8), (B, 64, H/4, W/4), (B, 32, H/2, W/2)]
    Output: Dual heads
        - p_model_logits: (B, 1, H, W) - model's dynamic risk prediction
        - alpha_logits: (B, 1, H, W) - dynamic blending gate
    """

    def __init__(self, config: Dict):
        super().__init__()

        bottleneck_dim = config['bottleneck_dim']  # 256
        skip_channels = config['skip_channels']  # [128, 64, 32]
        output_channels = config['output_channels']  # 1
        num_groups = config.get('num_groups', 8)
        dropout = config.get('dropout', 0.1)
        use_cab = config.get('use_cab', True)  # Channel Attention Block
        cab_reduction = config.get('cab_reduction', 16)  # CAB reduction ratio

        # Upsampling blocks with CAB
        self.up_blocks = nn.ModuleList()
        self.cab_blocks = nn.ModuleList() if use_cab else None

        in_ch = bottleneck_dim
        for skip_ch in skip_channels:
            # Convolution block
            block = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, skip_ch, 3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(num_groups, skip_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(skip_ch, skip_ch, 3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(num_groups, skip_ch),
                nn.ReLU(inplace=True)
            )
            self.up_blocks.append(block)

            # Channel Attention Block (applied after concat, before conv)
            if use_cab:
                cab = ChannelAttentionBlock(in_ch + skip_ch, reduction_ratio=cab_reduction)
                self.cab_blocks.append(cab)

            in_ch = skip_ch

        # Dual output heads (both derive from same decoder features)
        self.head_p_model = nn.Conv2d(skip_channels[-1], output_channels, 1)  # Risk prediction
        self.head_alpha = nn.Conv2d(skip_channels[-1], output_channels, 1)    # Blending gate

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, bottleneck: torch.Tensor,
                skip_connections: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bottleneck: (B, 256, H/16, W/16)
            skip_connections: [(B, 128, H/8, W/8), (B, 64, H/4, W/4), (B, 32, H/2, W/2)]

        Returns:
            p_model_logits: (B, 1, H, W) - model's dynamic risk prediction
            alpha_logits: (B, 1, H, W) - dynamic blending gate
        """
        x = bottleneck

        for i, (block, skip) in enumerate(zip(self.up_blocks, skip_connections)):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)  # Concat: upsampled + skip

            # Apply Channel Attention Block AFTER concat, BEFORE conv
            if self.cab_blocks is not None:
                x = self.cab_blocks[i](x)

            x = block(x)  # Convolution block

        # Dual output heads (both from same high-res features)
        p_model_logits = self.head_p_model(x)  # (B, 1, H, W)
        alpha_logits = self.head_alpha(x)      # (B, 1, H, W)

        return p_model_logits, alpha_logits




class HierarchicalFusionModel(nn.Module):
    """
    Hierarchical Fusion Model for Landslide Risk Prediction

    3-Encoder + 2-Stage Fusion + U-Net Decoder + Dynamic Blending
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_config = config['model']

        # GNN Encoder 사용 여부
        self.use_gnn_encoder = model_config.get('use_gnn_encoder', True)

        # Encoders
        self.encoder_s = StaticEncoder(model_config['static_encoder'])
        self.encoder_d = DynamicEncoder(model_config['dynamic_encoder'])

        if self.use_gnn_encoder:
            self.encoder_g = GNNContextEncoder(model_config['gnn_encoder'])
            self.total_fusion = TotalFusion(model_config['total_fusion'])
        else:
            self.encoder_g = None
            self.total_fusion = None

        # Fusion modules
        self.sd_fusion = CrossAttentionFusion(model_config['sd_fusion'])

        # Decoder (outputs dual heads: p_model and alpha)
        self.decoder = HierarchicalDecoder(model_config['decoder'])

        # Dynamic blending enabled (no separate calibration module needed)
        self.use_dynamic_blending = model_config['kfs_calibration'].get('use_kfs_calibration', False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: {
                'static': (B, 21, H, W),
                'dynamic': (B, T=5, 3, H, W),
                'gnn_embedding': (B, 128, H, W) or zeros,
                'kfs_prior': (B, 1, H, W),
                'slope_mask': (B, 1, H, W)
            }

        Returns:
            {
                'final_output': (B, 1, H, W) - blended risk logits (for MIL loss)
                'alpha_map': (B, 1, H, W) - dynamic blending gate (for alpha loss)
                'p_model_map': (B, 1, H, W) - model's pure prediction (for monitoring)
                'attention_weights': (B, num_heads, HW, HW)
            }
        """
        # 1. Encoder-S: Static terrain
        s_bottleneck, s_skips = self.encoder_s(batch['static'])

        # 2. Encoder-D: Dynamic temporal
        d_bottleneck = self.encoder_d(batch['dynamic'])

        # 3. SD Fusion: Cross-Attention
        sd_fused, attention_weights = self.sd_fusion(s_bottleneck, d_bottleneck)

        # 4. Total Fusion (조건부)
        if self.use_gnn_encoder:
            # GNN 사용: Encoder-G + Total Fusion
            g_context = self.encoder_g(batch['gnn_embedding'])
            total_bottleneck = self.total_fusion(sd_fused, g_context)
        else:
            # GNN 비활성화: SD Fusion만 사용
            total_bottleneck = sd_fused

        # 5. Decoder: Dual output heads
        p_model_logits, alpha_logits = self.decoder(total_bottleneck, s_skips)

        # 6. Dynamic Blending (if enabled)
        if self.use_dynamic_blending:
            # Convert logits to probabilities
            p_model_map = torch.sigmoid(p_model_logits)  # (B, 1, H, W) - model prediction [0, 1]
            alpha_map = torch.sigmoid(alpha_logits)      # (B, 1, H, W) - blending gate [0, 1]

            # Dynamic blending: α(x,y) * KFS(x,y) + (1-α(x,y)) * P_model(x,y)
            kfs_prior = batch['kfs_prior']  # (B, 1, H, W) - KFS prior [0, 1]
            blended_prob = alpha_map * kfs_prior + (1 - alpha_map) * p_model_map

            # Convert back to logits for loss computation (stability clamping)
            blended_prob_clamped = torch.clamp(blended_prob, 1e-7, 1 - 1e-7)
            final_output = torch.log(blended_prob_clamped / (1 - blended_prob_clamped))
        else:
            # No blending - use model prediction directly
            final_output = p_model_logits
            alpha_map = torch.zeros_like(p_model_logits)  # Dummy alpha
            p_model_map = torch.sigmoid(p_model_logits)

        return {
            'final_output': final_output,        # Blended logits (for MIL loss)
            'alpha_map': alpha_map,              # Dynamic gate (for alpha loss)
            'p_model_map': p_model_map,          # Pure model prediction (monitoring)
            'attention_weights': attention_weights
        }
