import torch
import torch.nn as nn
from typing import List

class GFF(nn.Module):
    """
    Gating Feature Fusion (GFF) module.
    
    Uses a gating mechanism to control the flow of information from multiple modalities. Each modality contributes to the fused representation based on a learned gate vector. The gated fusion approach enables selective integration of modalities based on their importance or quality.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int, dropout=True) -> None:
        """
        Initializes the GFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
            dropout (bool): Whether to apply dropout
        """
        super(GFF, self).__init__()
        
        if len(input_dims) != 2:
            raise ValueError("GFF module requires exactly two input modalities.")

        # Fully connected layers for each modality to project to a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) 
            for input_dim 
            in input_dims
        ])
        
        # Fully connected layer to project the concatenated features to a common space
        self.gate_proj = nn.Linear(sum(input_dims), hidden_dim)
        
        # Dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
        # Tanh
        self.tanh = nn.Tanh()
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the GFF module.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
            
        Returns:
            torch.Tensor: Fused features
        """
        
        # Concatenate the embeddings
        concat_embeds = torch.cat(proj_embeds, dim=-1)
        
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
            torch.mul(gate, proj_embeds[:, 0]),
            torch.mul(1 - gate, proj_embeds[:, 1])
        ]
        
        # Sum the gated embeddings
        fusion_vector = torch.sum(gated_embeds)
        
        return fusion_vector