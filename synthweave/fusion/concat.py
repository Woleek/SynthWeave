import torch
import torch.nn as nn

from typing import Dict, List, Optional
from .base import BaseFusion

"""
Concatenation-based fusion modules for multimodal feature fusion.

This module implements concatenation-based fusion methods, specifically the
Concatenation Feature Fusion (CFF) module, which serves as a baseline method
for multimodal fusion tasks.
"""


class CFF(BaseFusion):
    """Concatenation Feature Fusion (CFF) module.

    A baseline fusion method that concatenates uni-modal feature vectors and processes
    the concatenated features through a fully connected layer with dropout and ReLU
    activation.

    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845

    Attributes:
        fc_layer: Fully connected layer to process concatenated features

    Note:
        Expects embeddings of shape (batch_size, embed_dim)
    """

    def __init__(
        self,
        output_dim: int,
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the CFF module.

        Args:
            output_dim: Dimension of the output features
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in linear layers
            dropout_p: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments passed to parent class
        """
        super(CFF, self).__init__(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout_p,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        # Fully connected layer to process concatenated features
        self.fc = nn.Linear(self.proj_dim * len(self.modalities), self.output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the CFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, output_dim)
        """
        # Concatenate embeddings from all modalities
        concat_embeds = torch.cat(
            [embeddings[k] for k in self.modalities], dim=-1
        )  # (B, sum(Di))

        # 2. FC + Dropout + ReLU
        out = self.fc(concat_embeds)
        out = self.dropout(out)
        out = self.relu(out)
        return out
