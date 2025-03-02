import torch
import torch.nn as nn

from typing import List
from .modules import MMDBlock
from ..base import BaseFusion
from ...utils.modules import LazyLinearXavier
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

    Based on: "Multi-Modal Joint-Decoder for Robust Visual Question Answering"
    Source: https://arxiv.org/abs/2310.13103

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
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
        num_layers: int = 3,
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
        super(MMD, self).__init__(dropout, unify_embeds)
        self._num_layers = num_layers
        logger.warning(f"MMD initialized with {num_layers} layers.")

        # Projection layers to project each modality into a common space
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList(
                [LazyLinearXavier(output_dim) for _ in range(n_modals)]
            )
        else:
            self.unify_layers = nn.ModuleList([nn.Identity() for _ in range(n_modals)])

        # Stack L layers of MMDBlock
        self.blocks = nn.ModuleList(
            [MMDBlock(output_dim) for _ in range(self._num_layers)]
        )

        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the MMD module.

        Args:
            embeddings: List of tensors from each modality
                       Shape: [(batch_size, embed_dim), ...]

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, n_modals * embed_dim)

        Process:
            1. Projects each modality into common space
            2. Applies dropout to projected embeddings
            3. Passes through L MMDBlock layers
            4. Concatenates refined modality features
        """
        # Project inputs into a common space
        proj_embeds = [
            unify_layer(embed)
            for unify_layer, embed in zip(self.unify_layers, embeddings)
        ]

        proj_embeds = [self.dropout(embed) for embed in proj_embeds]

        # Pass through L MMDBlocks
        for block in self.blocks:
            proj_embeds = block(proj_embeds)

        # Concatenate the refined modality features
        fusion_vector = torch.cat(proj_embeds, dim=-1)

        return fusion_vector
