import torch
import torch.nn as nn

from typing import List
from .modules import MMDBlock
from ..base import BaseFusion
from ...utils.modules import LazyLinearXavier
from ... import logger

class MMD(BaseFusion):
    """
    Multi-Modal Joint-Decoder (MMD) module.
    
    Uses L layers of MMDBlock with bi-directional cross-attention to compute attention weights between input modalities.
    The attended features are then combined with self-attention and feed-forward sub-layers to generate discriminative and modality-enhanced representations.
    The final fused feature is obtained by concatenating the refined modality features.
    """
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
        num_layers: int = 3
    ):
        """
        Initialize the MMD.
        """
        super(MMD, self).__init__(dropout, unify_embeds)
        self._num_layers = num_layers
        logger.warning(f"MMD initialized with {num_layers} layers.")
        
        # Projection layers to project each modality into a common space
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList([
                LazyLinearXavier(output_dim)
                for _ in range(n_modals)
            ])
        else:
            self.unify_layers = nn.ModuleList([
                nn.Identity()
                for _ in range(n_modals)
            ])

        # Stack L layers of MMDBlock
        self.blocks = nn.ModuleList([MMDBlock(output_dim) for _ in range(self._num_layers)])
        
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the MMD.
        """
        # Project inputs into a common space
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        proj_embeds = [self.dropout(embed) for embed in proj_embeds]

        # Pass through L MMDBlocks
        for block in self.blocks:
            proj_embeds = block(proj_embeds)

        # Concatenate the refined modality features
        fusion_vector = torch.cat(proj_embeds, dim=-1)

        return fusion_vector