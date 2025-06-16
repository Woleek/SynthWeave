import torch
import torch.nn as nn

from typing import Dict, List, Optional
from .modules import MMDBlock
from ..base import BaseFusion
from ...utils.modules import LazyLinearXavier, LinearXavier
from ... import logger

"""
Multi-Modal Joint-Decoder (MMD) module for multimodal feature fusion.

This module implements the MMD fusion method which uses multiple layers of
bi-directional cross-attention to compute attention weights between input modalities
and combines them using self-attention and feed-forward networks.
"""


class MMD(BaseFusion):
    """Multi-Modal Joint-Decoder (MMD) fusion module.

    Uses L layers of MMDBlock with bi-directional cross-attention to compute attention
    weights between input modalities. The attended features are then combined with
    self-attention and feed-forward sub-layers to generate discriminative and
    modality-enhanced representations.

    Based on: "AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake"
    Source: https://ieeexplore.ieee.org/abstract/document/10081373

    Attributes:
        unify_layers: ModuleList of projection layers for each modality
        blocks: ModuleList of MMDBlock layers
        dropout: Dropout layer for regularization

    Note:
        - Expects embeddings of shape (batch_size, embed_dim)
        - Number of layers can be configured via num_layers parameter
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
        """Initialize the MMD module.

        Args:
            output_dim: Dimension of the output features
            n_modals: Number of input modalities
            dropout: Whether to use dropout for regularization
            unify_embeds: Whether to project all modalities to same dimension
            num_layers: Number of MMDBlock layers to use

        Note:
            Logs a warning message with the number of layers initialized
        """
        super(MMD, self).__init__(
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
        self._num_layers = kwargs.get("num_layers", 3)
        logger.info(f"MMD initialized with {self._num_layers} layers.")

        num_att_heads: int = kwargs.get("num_att_heads", 1)

        # Stack L layers of MMDBlock
        self.blocks = nn.ModuleList(
            [
                MMDBlock(self.proj_dim, num_att_heads, dropout_p)
                for _ in range(self._num_layers)
            ]
        )

        # Final projection layer
        self.fc_layer = LinearXavier(
            self.proj_dim * len(modality_keys), output_dim, bias
        )

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the MMD module.

        Args:
            embeddings: List of tensors from each modality
                       Shape: [(batch_size, embed_dim), ...]

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, n_modals * embed_dim)

        Process:
            1. Passes through L MMDBlock layers
            2. Concatenates refined modality features
        """
        emb_list = [
            embeddings[k].unsqueeze(1) for k in self.modalities
        ]  # list[(B, 1, PROJ)]

        # Pass through L MMDBlocks
        for block in self.blocks:
            emb_list = block(emb_list)  # list length preserved

        # Concatenate the refined modality features
        cls_tokens = [emb[:, 0] for emb in emb_list]  # list[(B, PROJ)]
        fusion_vector = torch.cat(cls_tokens, dim=1)  # (B, M * PROJ)

        # Final projection layer
        fusion_vector = self.fc_layer(fusion_vector)

        return fusion_vector
