import torch
import torch.nn as nn

from typing import List
from .modules import MMDBlock
from ..base import BaseFusion
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
        dropout: bool = True,
        input_dims: List[int] | None = None,
        num_layers: int = 3
    ):
        """
        Initialize the MMD.
        """
        super(MMD, self).__init__(output_dim, dropout, input_dims)
        self._num_layers = num_layers
        logger.warning(f"MMD initialized with {num_layers} layers.")
        
    def _lazy_init(self) -> None:
        # Projection layers to project each modality into a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim // len(self._input_dims)) for input_dim in self._input_dims
        ])

        # Stack L layers of MMDBlock
        self.blocks = nn.ModuleList([MMDBlock(self._output_dim // len(self._input_dims)) for _ in range(self._num_layers)])
        
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
        proj_inputs = [self.proj_layers[i](embeddings[i]) for i in range(len(embeddings))]
        proj_inputs = [self.dropout(proj) for proj in proj_inputs]

        # Pass through L MMDBlocks
        for block in self.blocks:
            proj_inputs = block(proj_inputs)

        # Concatenate the refined modality features
        fusion_vector = torch.cat(proj_inputs, dim=-1)

        return fusion_vector