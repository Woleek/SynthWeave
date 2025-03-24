import torch
import torch.nn as nn

from typing import Dict, List, Optional
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier, LinearXavier

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

        # Gate projection to learn modality importance
        if self.proj_dim is None:
            self.gate_proj = LazyLinearXavier(output_dim * len(modality_keys), bias)
        else:
            self.gate_proj = LinearXavier(
                self.proj_dim * len(modality_keys),
                output_dim * len(modality_keys),
                bias,
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Softmax
        self.softmax = nn.Softmax(dim=-2)  # Act on the modality dimension

        # Tanh
        self.tanh = nn.Tanh()

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
        # Concatenate the embeddings
        concat_embeds = torch.cat(
            list(embeddings.values()), dim=-1
        )  # (batch_size, n_modals * embed_dim)

        # Apply tanh activation
        embeddings = {
            key: self.tanh(embedding) for key, embedding in embeddings.items()
        }

        # Compute gating vectors
        gate_logits = self.gate_proj(
            concat_embeds
        )  # (batch_size, n_modals * output_dim)
        gate_logits = self.dropout(gate_logits)

        # Apply sigmoid activation to each gate
        gate_logits = gate_logits.view(-1, len(self.modalities), self.output_dim)
        gate = self.softmax(gate_logits)  # (batch_size, n_modals, output_dim)

        # Apply gating mechanism to each modality
        stack_embeds = torch.stack(
            list(embeddings.values()), dim=1
        )  # (batch_size, n_modals, embed_dim)
        gated_embeds = gate * stack_embeds  # element-wise multiplication

        # Sum the gated embeddings
        fusion_vector = gated_embeds.sum(dim=1)  # (batch_size, output_dim)

        return fusion_vector
