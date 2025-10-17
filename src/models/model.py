"""
GNN-RNN Hybrid Model for Landslide Risk Prediction

Architecture:
- GNN Encoder: GraphSAGE or GAT (spatial features)
- RNN Encoder: Bi-LSTM (temporal features)
- Cross-Attention Fusion: Attention between GNN and RNN embeddings with residual connection
- MLP Classifier: Binary classification with Batch Normalization

Author: Landslide Risk Analysis Project
Date: 2025-01-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


# ============================================================
# GNN Encoders
# ============================================================

class GraphSAGE_Encoder(nn.Module):
    """
    GraphSAGE encoder for spatial feature learning

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate (applied between layers, not used initially)
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, in_channels) - node features
            edge_index: (2, num_edges) - graph connectivity
            edge_attr: (num_edges, 1) - edge weights (cosine similarity)

        Returns:
            x: (num_nodes, hidden_channels) - node embeddings
        """
        # Note: GraphSAGE uses edge_attr as weights in aggregation
        # We pass edge_attr but SAGEConv may not use it directly
        # For weighted aggregation, need custom implementation or use edge_weight parameter

        for i, conv in enumerate(self.convs):
            # Extract edge weights if available
            edge_weight = edge_attr.squeeze(-1) if edge_attr is not None else None

            x = conv(x, edge_index)  # SAGEConv doesn't have edge_weight in basic version

            if i < self.num_layers - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GAT_Encoder(nn.Module):
    """
    Graph Attention Network encoder for spatial feature learning

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension (per head)
        num_layers: Number of GNN layers
        heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4, dropout=0.0):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer: multi-head attention with concatenation
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads,
                   dropout=dropout, concat=True)
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                       dropout=dropout, concat=True)
            )

        # Last layer: single head, no concatenation
        self.convs.append(
            GATConv(hidden_channels * heads if num_layers > 1 else in_channels,
                   hidden_channels, heads=1, dropout=dropout, concat=False)
        )

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, in_channels) - node features
            edge_index: (2, num_edges) - graph connectivity
            edge_attr: (num_edges, 1) - NOT used (GAT uses self-attention)

        Returns:
            x: (num_nodes, hidden_channels) - node embeddings
        """
        # GAT uses self-attention, so edge_attr is ignored

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < self.num_layers - 1:
                x = F.elu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# ============================================================
# RNN Encoder
# ============================================================

class BiLSTM_Encoder(nn.Module):
    """
    Bidirectional LSTM encoder for temporal feature learning

    Args:
        input_dim: Input feature dimension (dynamic features)
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate (applied between LSTM layers)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2  # Bidirectional

    def forward(self, x):
        """
        Args:
            x: (batch_size, timesteps, input_dim) - time series

        Returns:
            h: (batch_size, hidden_dim*2) - last timestep hidden state
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, timesteps, hidden*2)

        # Use last timestep output
        h = lstm_out[:, -1, :]  # (batch, hidden_dim*2)

        return h


# ============================================================
# Cross-Attention Fusion with Residual Connection
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between GNN and RNN embeddings with residual connection

    Design:
    1. Compute attention matrix between h_gnn and h_rnn
    2. Apply attention weights to both embeddings
    3. Add residual connection (Attention output + Original embeddings)

    Args:
        gnn_dim: GNN embedding dimension
        rnn_dim: RNN embedding dimension (usually hidden_dim * 2)
    """
    def __init__(self, gnn_dim, rnn_dim):
        super().__init__()

        self.gnn_dim = gnn_dim
        self.rnn_dim = rnn_dim

        # Query, Key, Value projections
        # GNN queries RNN
        self.q_gnn = nn.Linear(gnn_dim, gnn_dim)
        self.k_rnn = nn.Linear(rnn_dim, gnn_dim)  # Project to same dim

        # RNN queries GNN
        self.q_rnn = nn.Linear(rnn_dim, rnn_dim)
        self.k_gnn = nn.Linear(gnn_dim, rnn_dim)  # Project to same dim

        # Value projections
        self.v_gnn = nn.Linear(gnn_dim, gnn_dim)
        self.v_rnn = nn.Linear(rnn_dim, rnn_dim)

        # Cross-projection layers for residual connection
        self.proj_rnn_to_gnn = nn.Linear(rnn_dim, gnn_dim, bias=False)
        self.proj_gnn_to_rnn = nn.Linear(gnn_dim, rnn_dim, bias=False)

        # Scaling factor for dot-product attention
        self.scale_gnn = gnn_dim ** 0.5
        self.scale_rnn = rnn_dim ** 0.5

    def forward(self, h_gnn, h_rnn):
        """
        Args:
            h_gnn: (batch, gnn_dim) - GNN embeddings
            h_rnn: (batch, rnn_dim) - RNN embeddings

        Returns:
            h_fused: (batch, gnn_dim + rnn_dim) - fused embeddings
            attn_weights: (batch, 2) - attention importance scores
        """
        batch_size = h_gnn.shape[0]

        # === GNN attends to RNN ===
        q_gnn = self.q_gnn(h_gnn)  # (batch, gnn_dim)
        k_rnn = self.k_rnn(h_rnn)  # (batch, gnn_dim)
        v_rnn_for_gnn = self.v_rnn(h_rnn)  # (batch, rnn_dim)

        # Attention scores: (batch, 1)
        attn_score_gnn = (q_gnn * k_rnn).sum(dim=1, keepdim=True) / self.scale_gnn
        attn_weight_gnn = torch.sigmoid(attn_score_gnn)  # (batch, 1)

        # Attended GNN: incorporate RNN information with residual
        # Project v_rnn to gnn_dim for addition
        v_rnn_projected = self.proj_rnn_to_gnn(v_rnn_for_gnn)  # (batch, gnn_dim)
        h_gnn_attended = h_gnn + attn_weight_gnn * v_rnn_projected


        # === RNN attends to GNN ===
        q_rnn = self.q_rnn(h_rnn)  # (batch, rnn_dim)
        k_gnn = self.k_gnn(h_gnn)  # (batch, rnn_dim)
        v_gnn_for_rnn = self.v_gnn(h_gnn)  # (batch, gnn_dim)

        # Attention scores: (batch, 1)
        attn_score_rnn = (q_rnn * k_gnn).sum(dim=1, keepdim=True) / self.scale_rnn
        attn_weight_rnn = torch.sigmoid(attn_score_rnn)  # (batch, 1)

        # Attended RNN: incorporate GNN information with residual
        # Project v_gnn to rnn_dim for addition
        v_gnn_projected = self.proj_gnn_to_rnn(v_gnn_for_rnn)  # (batch, rnn_dim)
        h_rnn_attended = h_rnn + attn_weight_rnn * v_gnn_projected

        # === Concatenate attended embeddings ===
        h_fused = torch.cat([h_gnn_attended, h_rnn_attended], dim=1)  # (batch, gnn_dim + rnn_dim)

        # === Aggregate attention weights for interpretability ===
        # Average of both attention weights
        attn_importance = torch.cat([attn_weight_gnn, attn_weight_rnn], dim=1)  # (batch, 2)
        attn_importance = F.softmax(attn_importance, dim=1)  # Normalize

        return h_fused, attn_importance


# ============================================================
# MLP Classifier with Batch Normalization
# ============================================================

class MLPClassifier(nn.Module):
    """
    MLP classifier with Batch Normalization and Dropout

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        dropout: Dropout rate (applied after each hidden layer)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.4):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Batch Normalization
                nn.ReLU(),
                nn.Dropout(dropout)  # Dropout after activation
            ])
            in_dim = hidden_dim

        # Output layer (no BN, no dropout)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch,) - binary classification logits
        """
        logits = self.mlp(x).squeeze(-1)  # (batch,)
        return logits


# ============================================================
# Main Landslide Risk Model
# ============================================================

class LandslideRiskModel(nn.Module):
    """
    GNN-RNN Hybrid Model for Landslide Risk Prediction

    Components:
    1. GNN Encoder (GraphSAGE or GAT): Learn spatial relationships
    2. RNN Encoder (Bi-LSTM): Learn temporal patterns
    3. Cross-Attention Fusion: Integrate spatial and temporal information
    4. MLP Classifier: Binary classification

    Args:
        static_dim: Static feature dimension (automatically inferred from data)
        dynamic_dim: Dynamic feature dimension (automatically inferred from data)
        gnn_type: 'sage' or 'gat'
        gnn_hidden: GNN hidden dimension
        gnn_layers: Number of GNN layers
        rnn_hidden: RNN hidden dimension (output will be rnn_hidden*2)
        rnn_layers: Number of RNN layers
        dropout: Dropout rate (not used in encoders initially, only in MLP)
        gat_heads: Number of attention heads for GAT (default: 4)
    """
    def __init__(self,
                 static_dim,
                 dynamic_dim,
                 gnn_type='sage',
                 gnn_hidden=64,
                 gnn_layers=2,
                 rnn_hidden=64,
                 rnn_layers=2,
                 dropout=0.3,
                 gat_heads=4):
        super().__init__()

        # Store dimensions for SHAP analysis
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.gnn_type = gnn_type

        # 1. GNN Encoder
        if gnn_type == 'sage':
            self.gnn = GraphSAGE_Encoder(static_dim, gnn_hidden, gnn_layers, dropout=0.0)
        elif gnn_type == 'gat':
            self.gnn = GAT_Encoder(static_dim, gnn_hidden, gnn_layers, heads=gat_heads, dropout=0.0)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Choose 'sage' or 'gat'.")

        # 2. RNN Encoder
        self.rnn = BiLSTM_Encoder(dynamic_dim, rnn_hidden, rnn_layers, dropout=0.0)

        # 3. Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(gnn_hidden, rnn_hidden * 2)

        # 4. MLP Classifier
        fused_dim = gnn_hidden + rnn_hidden * 2
        self.classifier = MLPClassifier(fused_dim, hidden_dims=[128, 64], dropout=dropout)

    def forward(self, batch, return_attention=False, return_embeddings=False):
        """
        Forward pass

        Args:
            batch: Dictionary containing:
                - static_x: (num_nodes, static_dim)
                - edge_index: (2, num_edges)
                - edge_attr: (num_edges, 1) - used only for GraphSAGE
                - node_indices: (batch_size,)
                - dynamic_x: (batch_size, window_size, dynamic_dim)
            return_attention: Return attention weights for interpretability
            return_embeddings: Return intermediate embeddings for SHAP analysis

        Returns:
            logits: (batch_size,) - classification logits
            OR
            dict with logits, attention, embeddings (if return_* flags are True)
        """
        # Unpack batch
        static_x = batch['static_x']
        edge_index = batch['edge_index']
        edge_attr = batch['edge_attr'] if self.gnn_type == 'sage' else None
        node_indices = batch['node_indices']
        dynamic_x = batch['dynamic_x']

        # 1. GNN: Encode all nodes
        h_gnn_all = self.gnn(static_x, edge_index, edge_attr)  # (num_nodes, gnn_hidden)

        # 2. Select batch nodes
        h_gnn = h_gnn_all[node_indices]  # (batch_size, gnn_hidden)

        # 3. RNN: Encode time series
        h_rnn = self.rnn(dynamic_x)  # (batch_size, rnn_hidden*2)

        # 4. Cross-Attention Fusion with Residual
        h_fused, attn_weights = self.fusion(h_gnn, h_rnn)  # (batch, gnn_hidden + rnn_hidden*2)

        # 5. Classification
        logits = self.classifier(h_fused)  # (batch,)

        # Return options
        if not return_attention and not return_embeddings:
            return logits

        outputs = {'logits': logits}
        if return_attention:
            outputs['attention'] = attn_weights
        if return_embeddings:
            outputs['h_gnn'] = h_gnn
            outputs['h_rnn'] = h_rnn
            outputs['h_fused'] = h_fused

        return outputs

    def forward_from_embeddings(self, h_gnn, dynamic_x):
        """
        Forward pass from pre-computed GNN embeddings
        Used for SHAP analysis of dynamic features

        Args:
            h_gnn: (batch, gnn_hidden) - fixed GNN embeddings
            dynamic_x: (batch, window_size, dynamic_dim) - variable input

        Returns:
            logits: (batch,)
        """
        # RNN encoding
        h_rnn = self.rnn(dynamic_x)

        # Fusion
        h_fused, _ = self.fusion(h_gnn, h_rnn)

        # Classification
        logits = self.classifier(h_fused)

        return logits


# ============================================================
# Proxy Model for Static Feature SHAP Analysis
# ============================================================

class StaticFeatureProxy(nn.Module):
    """
    Simple MLP for static feature importance analysis via SHAP

    Purpose: NOT for high prediction accuracy, but for feature ranking

    Args:
        input_dim: Static feature dimension
        hidden_dims: List of hidden dimensions
        dropout: Dropout rate
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # Binary output

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - static features

        Returns:
            logits: (batch, 1) or (batch,) - binary classification logits
        """
        logits = self.mlp(x)
        # Keep 2D for SHAP compatibility, squeeze for training
        if not self.training and logits.shape[-1] == 1:
            return logits  # (batch, 1) for SHAP
        return logits.squeeze(-1)  # (batch,) for training


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Formula: FL = -alpha * (1-pt)^gamma * log(pt)

    Args:
        alpha: Weighting factor for positive class (0.25 default)
        gamma: Focusing parameter (2.0 default)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch,) - model predictions (before sigmoid)
            targets: (batch,) - binary labels (0 or 1)

        Returns:
            loss: scalar
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss

        return loss.mean()


def get_loss_fn(loss_type='weighted_bce', pos_weight=None, device=None):
    """
    Get loss function by type

    Args:
        loss_type: 'weighted_bce' or 'focal'
        pos_weight: Positive class weight for BCE (ignored for focal)
        device: Device to place tensors on

    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'weighted_bce':
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
            if device is not None:
                pos_weight = pos_weight.to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif loss_type == 'focal':
        return FocalLoss(alpha=0.25, gamma=2.0)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================
# Model Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing Landslide Risk Model")
    print("="*70)

    # Dummy data
    batch_size = 16
    num_nodes = 100
    num_edges = 500
    static_dim = 15
    dynamic_dim = 6
    window_size = 5

    # Create dummy batch
    batch = {
        'static_x': torch.randn(num_nodes, static_dim),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'edge_attr': torch.rand(num_edges, 1),
        'node_indices': torch.randint(0, num_nodes, (batch_size,)),
        'dynamic_x': torch.randn(batch_size, window_size, dynamic_dim),
        'labels': torch.randint(0, 2, (batch_size,)).float()
    }

    # Test GraphSAGE model
    print("\n[1] Testing GraphSAGE Model")
    model_sage = LandslideRiskModel(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        gnn_type='sage',
        gnn_hidden=64,
        rnn_hidden=64
    )

    outputs = model_sage(batch, return_attention=True, return_embeddings=True)
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Attention shape: {outputs['attention'].shape}")
    print(f"  h_gnn shape: {outputs['h_gnn'].shape}")
    print(f"  h_rnn shape: {outputs['h_rnn'].shape}")
    print(f"  h_fused shape: {outputs['h_fused'].shape}")

    # Test GAT model
    print("\n[2] Testing GAT Model")
    model_gat = LandslideRiskModel(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        gnn_type='gat',
        gnn_hidden=64,
        rnn_hidden=64,
        gat_heads=4
    )

    logits = model_gat(batch)
    print(f"  Logits shape: {logits.shape}")

    # Test forward_from_embeddings (for SHAP)
    print("\n[3] Testing forward_from_embeddings")
    with torch.no_grad():
        h_gnn_fixed = outputs['h_gnn']
        logits_shap = model_sage.forward_from_embeddings(h_gnn_fixed, batch['dynamic_x'])
    print(f"  SHAP-compatible logits shape: {logits_shap.shape}")

    # Test Proxy Model
    print("\n[4] Testing Static Feature Proxy Model")
    proxy_model = StaticFeatureProxy(input_dim=static_dim)
    static_features = batch['static_x'][batch['node_indices']]
    proxy_logits = proxy_model(static_features)
    print(f"  Proxy logits shape: {proxy_logits.shape}")

    # Test loss functions
    print("\n[5] Testing Loss Functions")
    targets = batch['labels']

    loss_bce = get_loss_fn('weighted_bce', pos_weight=1.0)
    loss_val_bce = loss_bce(outputs['logits'], targets)
    print(f"  Weighted BCE Loss: {loss_val_bce.item():.4f}")

    loss_focal = get_loss_fn('focal')
    loss_val_focal = loss_focal(outputs['logits'], targets)
    print(f"  Focal Loss: {loss_val_focal.item():.4f}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
