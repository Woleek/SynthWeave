import torch
import torch.nn as nn

from typing import Dict, List, Optional
from .base import BaseFusion

"""
Gating-based fusion modules for multimodal feature fusion.

This module implements gating-based fusion methods, specifically the
Gating Feature Fusion (GFF) module, which uses learned gates to control
the flow of information from different modalities.
"""


class GFF(BaseFusion):
    """Gating Feature Fusion (GFF) module.

    Uses a gating mechanism to control the flow of information from multiple modalities.
    Each modality contributes to the fused representation based on a learned gate vector.
    The gated fusion approach enables selective integration of modalities based on their
    importance or quality.

    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845

    Attributes:
        gate_proj: Linear projection layer for computing gating vectors
        dropout: Dropout layer for regularization
        softmax: Softmax layer for normalizing gate values
        tanh: Tanh activation function

    Note:
        Expects embeddings of shape (batch_size, embed_dim)
    """

    def __init__(
        self,
        output_dim: int,
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the GFF module.

        Args:
            output_dim: Dimension of the output features
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in linear layers
            dropout: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments passed to parent class
        """
        super(GFF, self).__init__(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        self.output_dim = output_dim
        
        # Per-modality FC+Dropout
        self.mod_proj = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(self.proj_dim, self.output_dim, bias=bias),
                nn.Dropout(dropout)
            )
            for k in self.modalities
        })

        # Gate MLP: maps concat(raw features) -> N modality gates
        self.gate_proj = nn.Sequential(
            nn.Linear(self.proj_dim * len(self.modalities), self.output_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, len(self.modalities), bias=bias)
        )

        # Tanh activation and softmax for gating
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # over modalities

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the GFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, output_dim)

        Process:
            1. Concatenates embeddings from all modalities
            2. Applies tanh activation to individual embeddings
            3. Computes gating vectors through projection and softmax
            4. Applies gates to modality features
            5. Sums the gated features to produce final fusion
        """
        # Project & activate each modality feature
        mod_feats = [self.tanh(self.mod_proj[k](embeddings[k])) for k in self.modalities]  # list of (B, H)
        mod_feats = torch.stack(mod_feats, dim=1)  # (B, N, H)

        # Compute gates from concatenated raw embeddings
        concat = torch.cat([embeddings[k] for k in self.modalities], dim=1)  # (B, sum(Di))
        gate_logits = self.gate_proj(concat)  # (B, N)
        gates = self.softmax(gate_logits).unsqueeze(-1)  # (B, N, 1)

        # Weighted sum of modality features
        fused = (gates * mod_feats).sum(dim=1)  # (B, H)
        return fused
