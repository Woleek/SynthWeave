import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class AFF(nn.Module):
    """
    Attention Feature Fusion (AFF) module.
    
    Uses an attention mechanism to calculate attention weights for each modality based on their quality and significance. It dynamically adjusts the weights and fuses the modalities into a robust representation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int) -> None:
        """
        Initializes the AFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
        """
        super(AFF, self).__init__()
        # Linear layers to map uni-modal embeddings to a common hidden dimension
        self.projection_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) 
            for input_dim 
            in input_dims
        ])
        
        # Attention layer for computing weights
        n_mods = len(input_dims)
        self.attention_layer = nn.Linear(hidden_dim * n_mods, n_mods)


    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the AFF module.
        
        Steps:
        - Maps the uni-modal embeddings to a common hidden dimension.
        - Computes attention weights for each modality.
        - Fuses the modalities based on the attention weights.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
        
        Returns:
            torch.Tensor: Fused features
        """
        # Project uni-modal embeddings to a common hidden dimension
        proj_embeds = [
            uni_modal_attention(embed) 
            for uni_modal_attention, embed 
            in zip(self.projection_layers, embeddings)
        ]
        
        # Concatenate hidden embeddings
        concatenated = torch.cat(proj_embeds, dim=-1)
        
        # Compute attention weights
        attention_logits = self.attention_layer(concatenated)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Fuse modalities based on attention weights
        fused_features = torch.sum(attention_weights.unsqueeze(-1) * torch.stack(embeddings, dim=-1), dim=-1)
        
        return fused_features 
    
    
class IAFF(nn.Module):
    """
    Inter-Attention Feature Fusion (IAFF) module.
    
    This model applies an inter-attention mechanism to efficiently extract and fuse features from multiple modalities. It computes attention scores within each modality and between modalities interactively, ensuring that the most critical information is retained. The unimodal information is added back to prevent loss of essential features.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int) -> None:
        """
        Initializes the IAFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
        """
        super(IAFF, self).__init__()
        n_mods = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Fully connected layers to project each modality into a common space
        self.projection_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=False) for input_dim in input_dims
        ])
        
        # Attention layers to compute interactions between modalities
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_mods)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the IAFF module.
        
        Steps:
        - Maps the uni-modal embeddings to a common hidden dimension.
        - Computes attention weights for each modality.
        - Fuses the modalities based on the attention weights.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
            
        Returns:
            torch.Tensor: Fused features
        """
        # Project each modality into the common space
        projected_features = [
            layer(input_) for layer, input_ in zip(self.projection_layers, embeddings)
        ]

        # Compute inter-attention for each modality
        attended_features = []
        for i, feature in enumerate(projected_features):
            # Compute attention weights using the projected feature
            attention_scores = F.softmax(
                torch.matmul(feature, self.attention_layers[i](feature).T) / (self.hidden_dim ** 0.5),
                dim=-1
            )
            # Apply attention and add unimodal information back
            attended_feature = feature + torch.matmul(attention_scores, feature)
            attended_features.append(self.dropout(attended_feature))

        # Fuse all attended features by summation
        fused_features = torch.stack(attended_features, dim=0).sum(dim=0)
        return fused_features