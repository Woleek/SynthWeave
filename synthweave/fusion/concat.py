import torch
import torch.nn as nn

from typing import List
from .base import BaseFusion

class CFF(BaseFusion):
    """
    Concatenation Feature Fusion (CFF) module.
    
    Baseline method, concatenates uni-modal feature vectors and processes the fused features through a fully connected layer with dropout and ReLU activation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def _lazy_init(self) -> None:
        """
        Initializes the CFF module.
        """
        
        # Fully connected layer to process concatenated features
        self.fc_layer = nn.Linear(sum(self._input_dims), self._output_dim)
        
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
        # ReLU activation
        self.relu = nn.ReLU()

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CFF module.
        """
        
        # Concatenate embeddings
        concat_embeds = torch.cat(embeddings, dim=-1)
        
        # Process concatenated features
        fusion_vector = self.fc_layer(concat_embeds)
        
        # Apply dropout
        fusion_vector = self.dropout(fusion_vector)
        
        # Apply ReLU activation
        fusion_vector = self.relu(fusion_vector)
        
        return fusion_vector