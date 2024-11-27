import torch
import torch.nn as nn

from typing import List
from .base import BaseFusion

class GFF(BaseFusion):
    """
    Gating Feature Fusion (GFF) module.
    
    Uses a gating mechanism to control the flow of information from multiple modalities. Each modality contributes to the fused representation based on a learned gate vector. The gated fusion approach enables selective integration of modalities based on their importance or quality.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def _lazy_init(self) -> None:
        """
        Initializes the GFF module.
        """
        
        if len(self._input_dims) != 2:
            raise ValueError("GFF module requires exactly two input modalities.")

        # Fully connected layers for each modality to project to a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim) 
            for input_dim 
            in self._input_dims
        ])
        
        # Fully connected layer to project the concatenated features to a common space
        self.gate_proj = nn.Linear(sum(self._input_dims), self._output_dim)
        
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
        
        # Concatenate the embeddings
        concat_embeds = torch.cat(embeddings, dim=-1)
        
        # Project each modality into the common space
        proj_embeds = [
            layer(embed)
            for layer, embed 
            in zip(self.proj_layers, embeddings)
        ]
        
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