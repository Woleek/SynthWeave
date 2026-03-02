"""
Temporal aggregation modules for clip-level deepfake detection.

All modules aggregate per-window embeddings (W, D) → (D,) for a single clip.
Used in Approach A (inference-only) via granular_eval.py --temporal_agg.

Modules:
    MeanAggregator      - Simple mean pooling (baseline, 0 extra params)
    AttentionAggregator - Learned scalar attention over windows (~D²+D params)
    Conv1dAggregator    - 1D temporal convolution (~2D² params)
    GRUAggregator       - Bidirectional GRU (~4D² params)
    TransformerAggregator - Single-layer self-attention with CLS token (~4D² params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


TEMPORAL_AGG_TYPES = Literal["mean", "attention", "conv1d", "gru", "transformer"]


class MeanAggregator(nn.Module):
    """Baseline: simple mean over window dimension. No learnable parameters."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (W, D) per-window embeddings
        Returns:
            (D,) aggregated representation
        """
        return x.mean(dim=0)

    def num_parameters(self) -> int:
        return 0


class AttentionAggregator(nn.Module):
    """
    Scalar attention aggregator.
    Learns a query vector q; score_i = q · x_i → softmax → weighted sum.
    ~D params (query vector only) + projection head.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.Linear(embed_dim, 1, bias=False)  # D → 1 score per window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (W, D) per-window embeddings
        Returns:
            (D,) attention-weighted representation
        """
        scores = self.attn(x)           # (W, 1)
        weights = F.softmax(scores, dim=0)  # (W, 1)
        return (weights * x).sum(dim=0)  # (D,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Conv1dAggregator(nn.Module):
    """
    1D temporal convolution over the window sequence.
    Conv1d(D, D, k=3, padding=1) → BN → ReLU → mean pool.
    ~2D² params.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (W, D) per-window embeddings
        Returns:
            (D,) aggregated representation
        """
        # Conv1d expects (batch, channels, length); here batch=1, channels=D, length=W
        h = x.unsqueeze(0).permute(0, 2, 1)  # (1, D, W)
        h = self.act(self.bn(self.conv(h)))    # (1, D, W)
        return h.mean(dim=-1).squeeze(0)       # (D,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUAggregator(nn.Module):
    """
    Bidirectional GRU over window sequence; mean-pool all hidden states.
    BiGRU(D → hidden_dim), output_dim = 2*hidden_dim → projected back to D.
    ~4D² params.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        if hidden_dim is None:
            hidden_dim = embed_dim // 2  # keep output at embed_dim after concat
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (W, D) per-window embeddings
        Returns:
            (D,) aggregated representation
        """
        h, _ = self.gru(x.unsqueeze(0))  # (1, W, 2*hidden)
        pooled = h.squeeze(0).mean(dim=0)  # (2*hidden,)
        return self.proj(pooled)           # (D,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerAggregator(nn.Module):
    """
    Single-layer transformer with a learnable CLS token.
    Appends CLS to window sequence, runs multi-head self-attention,
    returns CLS output as the clip representation.
    ~4D² params.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (W, D) per-window embeddings
        Returns:
            (D,) aggregated representation (CLS output)
        """
        # Prepend CLS token: shape (1, W+1, D)
        seq = torch.cat([self.cls_token.unsqueeze(0), x.unsqueeze(0)], dim=1)
        out = self.transformer(seq)   # (1, W+1, D)
        return out[0, 0, :]           # CLS token output: (D,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_temporal_aggregator(agg_type: str, embed_dim: int) -> nn.Module:
    """Factory: return the requested temporal aggregator module.

    Args:
        agg_type: one of "mean", "attention", "conv1d", "gru", "transformer"
        embed_dim: embedding dimension of per-window representations
    Returns:
        Instantiated (untrained) aggregator module
    """
    _map = {
        "mean":        MeanAggregator,
        "attention":   AttentionAggregator,
        "conv1d":      Conv1dAggregator,
        "gru":         GRUAggregator,
        "transformer": TransformerAggregator,
    }
    if agg_type not in _map:
        raise ValueError(f"Unknown temporal aggregator: {agg_type!r}. "
                         f"Choose from {list(_map.keys())}")
    return _map[agg_type](embed_dim)
