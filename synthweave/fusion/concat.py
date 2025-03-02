import torch
import torch.nn as nn

from typing import Dict, List, Optional
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier, LinearXavier

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
        dropout: float = 0.5,
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
            dropout: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments passed to parent class
        """
        super(CFF, self).__init__(
            modality_keys,
            input_dims,
            bias,
            dropout,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        # Fully connected layer to process concatenated features
        if self.proj_dim is None:
            self.fc_layer = LazyLinearXavier(output_dim, bias)
        else:
            self.fc_layer = LinearXavier(
                self.proj_dim * len(modality_keys), output_dim, bias
            )

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the CFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, output_dim)
        """
        # Concatenate embeddings
        concat_embeds = torch.cat(
            list(embeddings.values()), dim=-1
        )  # (batch_size, n_modals * embed_dim)

        # Process concatenated features
        fusion_vector = self.fc_layer(concat_embeds)  # (batch_size, output_dim)

        return fusion_vector
