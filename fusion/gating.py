import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class GFF(nn.Module):
    """
    Gating Feature Fusion (GFF) module.
    
    Uses a gating mechanism to control the flow of information from multiple modalities. Each modality contributes to the fused representation based on a learned gate vector. The gated fusion approach enables selective integration of modalities based on their importance or quality.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int) -> None:
        """
        Initializes the GFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
        """
        super(GFF, self).__init__()
        
        n_mods = len(input_dims)

        # Fully connected layers for each modality to project to a common space
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(0.5) # prevent overfitting
            ) for input_dim in input_dims
        ])

        # Gate calculation layer
        self.gate_layer = nn.Linear(hidden_dim * n_mods, n_mods)
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the GFF module.
        
        Steps:
        - Project each modality into the common space.
        - Compute gate values.
        - Apply gating to each modality.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
            
        Returns:
            torch.Tensor: Fused features
        """
        
        # Project each modality into the common space
        projected_features = [
            F.tanh(layer(emb)) 
            for layer, emb 
            in zip(self.projection_layers, embeddings)
        ]
        projected_concat = torch.cat(projected_features, dim=-1)  # Concatenate along the feature dimension
        
        # Compute gate values
        gate_logits = self.gate_layer(projected_concat) 
        gate_values = F.softmax(gate_logits, dim=-1)
        
        # Apply gating to each modality
        fused_features = sum(gate_values[:, i:i+1] * feature for i, feature in enumerate(projected_features))
        
        return fused_features