import torch
import torch.nn as nn

from typing import List

class CFF(nn.Module):
    """
    Concatenation Feature Fusion (CFF) module.
    
    Baseline method, concatenates uni-modal feature vectors and processes the fused features through a fully connected layer with dropout and ReLU activation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int) -> None:
        """
        Initializes the CFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
        """
        super(CFF, self).__init__()
        
        # Fully connected layer to process concatenated features
        self.fc_drop = nn.Sequential(
            nn.Linear(sum(input_dims), hidden_dim),
            nn.Dropout(0.7) # prevent overfitting
        )
        
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CFF module.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
        
        Returns:
            torch.Tensor: Fused features
        """
        
        # Concatenate embeddings
        concat_embeds = torch.cat(embeddings, dim=-1)
        
        # Process concatenated features
        fusion_vector = self.fc_drop(concat_embeds)
        
        # Apply ReLU activation
        fusion_vector = self.relu(fusion_vector)
        
        return fusion_vector