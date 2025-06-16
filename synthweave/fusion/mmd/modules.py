from typing import List
import torch
import torch.nn as nn
from ...utils.modules import LinearXavier

"""
Core modules for the Multi-Modal Joint-Decoder (MMD) architecture.

This module implements the fundamental building blocks used in the MMD architecture:
- SelfAttention: Self-attention mechanism for intra-modality feature refinement
- FeedForward: Position-wise feed-forward network for feature transformation
- BiCroAttention: Bi-directional cross-attention for inter-modality feature fusion
- MMDBlock: Complete block combining all above components with residual connections
"""


class SelfAttention(nn.Module):
    """Self-Attention (SelfAtt) sub-layer.

    Implements scaled dot-product self-attention mechanism that allows each position
    in the input sequence to attend to all positions in the same sequence.

    Attributes:
        Wq: Query transformation matrix
        Wk: Key transformation matrix
        Wv: Value transformation matrix
        scale: Scaling factor for dot products (1/sqrt(d_k))
        softmax: Softmax layer for attention weights
    """

    def __init__(self, modal_dim, n_heads=1, dropout_p=0.1):
        """Initialize the SelfAttention module.

        Args:
            modal_dim: Dimension of the input features
        """
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            modal_dim, n_heads, dropout=dropout_p, batch_first=True
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Z):
        """Forward pass of the self-attention mechanism.

        Args:
            Z: Input tensor of shape (batch_size, seq_len, modal_dim)

        Returns:
            torch.Tensor: Attended features of shape (batch_size, seq_len, modal_dim)

        Process:
            1. Compute query, key, and value projections
            2. Calculate scaled dot-product attention
            3. Apply softmax to get attention weights
            4. Compute weighted sum of values
        """
        Z_hat, _ = self.mha(Z, Z, Z)
        Z_hat = self.dropout(Z_hat)

        return Z_hat


class FeedForward(nn.Module):
    """Feed-Forward (FF) sub-layer.

    Implements a position-wise feed-forward network consisting of two linear
    transformations with a ReLU activation in between.

    Attributes:
        linear1: First linear transformation
        linear2: Second linear transformation
        relu: ReLU activation function
    """

    def __init__(self, modal_dim, d_ff, dropout_p=0.1):
        """Initialize the FeedForward module.

        Args:
            modal_dim: Dimension of input/output features
            d_ff: Dimension of the inner layer
        """
        super(FeedForward, self).__init__()
        self.linear1 = LinearXavier(modal_dim, d_ff)
        self.linear2 = LinearXavier(d_ff, modal_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, modal_dim)

        Returns:
            torch.Tensor: Transformed features of shape (batch_size, seq_len, modal_dim)
        """
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


class BiCroAttention(nn.Module):
    """Bi-Directional Cross-Attention (BiCroAtt) block.

    Implements bi-directional cross-attention mechanism that allows each modality
    to attend to features from all other modalities.

    Attributes:
        Wq: Query transformation matrix
        Wk: Key transformation matrix
        Wv: Value transformation matrix
        scale: Scaling factor for dot products (1/sqrt(d_k))
        softmax: Softmax layer for attention weights
    """

    def __init__(self, modal_dim, n_heads=1, dropout_p=0.1):
        """Initialize the BiCroAttention module.

        Args:
            modal_dim: Dimension of the input features
        """
        super(BiCroAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            modal_dim, n_heads, dropout=dropout_p, batch_first=True
        )
        self.out = nn.Linear(modal_dim, modal_dim, bias=True)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, P: list[torch.Tensor]) -> torch.Tensor:
        """Compute bi-directional cross-attention for the i-th modality.

        Args:
            P_i: Features from the i-th modality
            *P_others: Features from all other modalities

        Returns:
            torch.Tensor: Cross-attended features concatenated across modalities

        Process:
            1. Compute query from current modality
            2. For each other modality:
                - Compute key and value projections
                - Calculate cross-attention
                - Apply attention to values
            3. Concatenate attended features
        """
        outs: List[torch.Tensor] = []

        for i, Pi in enumerate(P):
            cross_sum = 0.0
            for j, Pj in enumerate(P):
                if i == j:  # skip self
                    continue
                # Q comes from j, K/V from i
                att, _ = self.mha(query=Pj, key=Pi, value=Pi)
                cross_sum = cross_sum + att
            cross_avg = cross_sum / (len(P) - 1)  # mean over others
            outs.append(self.drop(self.out(cross_avg)))  # (B, T, EMB)
        return outs


class MMDBlock(nn.Module):
    """Single block of the Multi-Modal Joint-Decoder (MMD).

    Implements a complete MMD block containing BiCroAtt, SelfAtt, and FeedForward
    sub-layers with residual connections and layer normalization.

    Attributes:
        bi_cro_att: Bi-directional cross-attention layer
        self_att: Self-attention layer
        ff: Feed-forward layer
        ln1, ln2, ln3: Layer normalization layers
    """

    def __init__(self, modality_dim, num_att_heads=1, dropout_p=0.1) -> None:
        """Initialize the MMDBlock.

        Args:
            modality_dim: Dimension of the input features
        """
        super(MMDBlock, self).__init__()

        self.bi_cro_att = BiCroAttention(modality_dim, num_att_heads, dropout_p)
        self.ln1 = nn.LayerNorm(modality_dim)

        self.self_att = SelfAttention(modality_dim, num_att_heads, dropout_p)
        self.ln2 = nn.LayerNorm(modality_dim)

        self.ff = FeedForward(modality_dim, 4 * modality_dim, dropout_p)
        self.ln3 = nn.LayerNorm(modality_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass for a single MMDBlock.

        Args:
            tensors: List of embeddings from each modality

        Returns:
            list[torch.Tensor]: List of refined embeddings for each modality

        Process:
            1. Apply BiCroAtt with residual connection and layer norm
            2. Apply SelfAtt with residual connection and layer norm
            3. Apply FeedForward with residual connection and layer norm
        """
        # 1) cross-modal
        cross_outs = self.bi_cro_att(tensors)  # list[(B,T,d)]
        x1 = [self.ln1(t + c) for t, c in zip(tensors, cross_outs)]

        # 2) self-att per modality
        self_outs = [self.self_att(t) for t in x1]  # list[(B,T,d)]
        x2 = [self.ln2(t + self.dropout(sa)) for t, sa in zip(x1, self_outs)]

        # 3) feed-forward
        ff_outs = [self.ff(t) for t in x2]
        x3 = [self.ln3(t + self.dropout(ff)) for t, ff in zip(x2, ff_outs)]

        return x3  # same list length
