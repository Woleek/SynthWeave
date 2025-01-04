import torch
import torch.nn as nn

from typing import List
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier

class CFF(BaseFusion):
    """
    Concatenation Feature Fusion (CFF) module.
    
    Baseline method, concatenates uni-modal feature vectors and processes the fused features through a fully connected layer with dropout and ReLU activation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True
    ) -> None:
        """
        Initializes the CFF module.
        """
        super(CFF, self).__init__(dropout, unify_embeds)
        
        # Unify representations into same dimension
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
        
        # Fully connected layer to process concatenated features
        self.fc_layer = LazyLinearXavier(output_dim)
            
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CFF module.
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]

        # Concatenate embeddings
        concat_embeds = torch.cat(proj_embeds, dim=-1)
        
        # Process concatenated features
        fusion_vector = self.fc_layer(concat_embeds)
        
        # Apply dropout
        # fusion_vector = self.dropout(fusion_vector)
        
        return fusion_vector