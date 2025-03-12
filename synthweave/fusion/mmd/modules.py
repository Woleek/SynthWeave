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

    def __init__(self, modal_dim):
        """Initialize the SelfAttention module.

        Args:
            modal_dim: Dimension of the input features
        """
        super(SelfAttention, self).__init__()
        self.Wq = LinearXavier(modal_dim, modal_dim)
        self.Wk = LinearXavier(modal_dim, modal_dim)
        self.Wv = LinearXavier(modal_dim, modal_dim)
        self.scale = 1 / (modal_dim**0.5)

        self.softmax = nn.Softmax(dim=-1)

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
        Q = self.Wq(Z)
        K = self.Wk(Z)
        V = self.Wv(Z)
        Z_hat = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale) @ V
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

    def __init__(self, modal_dim, d_ff):
        """Initialize the FeedForward module.

        Args:
            modal_dim: Dimension of input/output features
            d_ff: Dimension of the inner layer
        """
        super(FeedForward, self).__init__()
        self.linear1 = LinearXavier(modal_dim, d_ff)
        self.linear2 = LinearXavier(d_ff, modal_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, modal_dim)

        Returns:
            torch.Tensor: Transformed features of shape (batch_size, seq_len, modal_dim)
        """
        return self.linear2(self.relu(self.linear1(x)))


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

    def __init__(self, modal_dim):
        """Initialize the BiCroAttention module.

        Args:
            modal_dim: Dimension of the input features
        """
        super(BiCroAttention, self).__init__()
        self.Wq = LinearXavier(modal_dim, modal_dim)
        self.Wk = LinearXavier(modal_dim, modal_dim)
        self.Wv = LinearXavier(modal_dim, modal_dim)
        self.scale = 1 / (modal_dim**0.5)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, P_i: torch.Tensor, *P_others: list[torch.Tensor]) -> torch.Tensor:
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
        Qi = self.Wq(P_i)
        Ki = self.Wk(P_i)
        Vi = self.Wv(P_i)

        attended_inputs = []
        for P_j in P_others:  # Iterate over all other modalities
            Qj = self.Wq(P_j)
            Kj = self.Wk(P_j)
            Vj = self.Wv(P_j)

            attended_inputs.append(
                self.softmax(torch.matmul(Qi, Kj.transpose(-2, -1)) * self.scale) @ Vj
            )

        return torch.cat(attended_inputs, dim=-1)


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

    def __init__(self, modality_dim) -> None:
        """Initialize the MMDBlock.

        Args:
            modality_dim: Dimension of the input features
        """
        super(MMDBlock, self).__init__()

        self.bi_cro_att = BiCroAttention(modality_dim)
        self.self_att = SelfAttention(modality_dim)
        self.ff = FeedForward(modality_dim, modality_dim)

        self.ln1 = nn.LayerNorm(modality_dim)
        self.ln2 = nn.LayerNorm(modality_dim)
        self.ln3 = nn.LayerNorm(modality_dim)

    def forward(self, proj_embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass for a single MMDBlock.

        Args:
            proj_embeds: List of projected embeddings from each modality

        Returns:
            list[torch.Tensor]: List of refined embeddings for each modality

        Process:
            1. Apply BiCroAtt with residual connection and layer norm
            2. Apply SelfAtt with residual connection and layer norm
            3. Apply FeedForward with residual connection and layer norm
        """
        attended_embeds = [
            self.ln1(
                proj_embeds[i]
                + self.bi_cro_att(
                    proj_embeds[i],
                    *[proj_embeds[j] for j in range(len(proj_embeds)) if j != i],
                )
            )
            for i in range(len(proj_embeds))
        ]

        refined_embeds = [
            self.ln2(attended_embeds[i] + self.self_att(attended_embeds[i]))
            for i in range(len(attended_embeds))
        ]

        refined_embeds = [
            self.ln3(refined_embeds[i] + self.ff(refined_embeds[i]))
            for i in range(len(refined_embeds))
        ]

        return refined_embeds
