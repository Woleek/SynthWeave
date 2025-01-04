import torch
import torch.nn as nn

from typing import List
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier

class GFF(BaseFusion):
    """
    Gating Feature Fusion (GFF) module.
    
    Uses a gating mechanism to control the flow of information from multiple modalities. Each modality contributes to the fused representation based on a learned gate vector. The gated fusion approach enables selective integration of modalities based on their importance or quality.
    
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
        Initializes the GFF module.
        """
        if n_modals != 2:
            raise ValueError("GFF module requires exactly two input modalities.")
        
        super(GFF, self).__init__(dropout, unify_embeds)

        # Fully connected layers for each modality to project to a common space
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
        
        # Fully connected layer to project the concatenated features to a common space
        self.gate_proj = LazyLinearXavier(output_dim)
        
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
        # Tanh
        self.tanh = nn.Tanh()
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the GFF module.
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        # Concatenate the embeddings
        concat_embeds = torch.cat(proj_embeds, dim=-1)
        
        # Apply dropout
        proj_embeds = [self.dropout(embed) for embed in proj_embeds]
        
        # Apply tanh activation
        proj_embeds = [self.tanh(embed) for embed in proj_embeds]
        
        # Project the concatenated features into the common space
        gate_proj_embeds = self.gate_proj(concat_embeds)
        
        # Apply dropout
        gate_proj_embeds = self.dropout(gate_proj_embeds)
        
        # Calculate the gate vector
        gate = self.sigmoid(gate_proj_embeds)
        
        # Apply gating mechanism
        gated_embeds = [
            torch.mul(gate, proj_embeds[0]),
            torch.mul(1 - gate, proj_embeds[1])
        ]
        
        # Sum the gated embeddings
        fusion_vector = torch.stack(gated_embeds, dim=0).sum(dim=0)
        
        return fusion_vector