import torch
import torch.nn as nn
from typing import List

class AFF(nn.Module):
    """
    Attention Feature Fusion (AFF) module.
    
    Uses an attention mechanism to calculate attention weights for each modality based on their quality and significance. It dynamically adjusts the weights and fuses the modalities into a robust representation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int, dropout=True) -> None:
        """
        Initializes the AFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
            dropout (bool): Whether to apply dropout
        """
        super(AFF, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Linear layers to map uni-modal embeddings to a common hidden dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) 
            for input_dim 
            in input_dims
        ])
        
        # Attention layer for computing weights
        self.att_layer = nn.Linear(sum(input_dims), hidden_dim)
        
        # Dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
            
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the AFF module.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
        
        Returns:
            torch.Tensor: Fused features
        """
        # Concatenate embeddings
        concat_embeds = torch.cat(embeddings, dim=-1)
        
        # Project uni-modal embeddings to a common hidden dimension
        proj_embeds = [
            layer(embed) 
            for layer, embed 
            in zip(self.proj_layers, embeddings)
        ]
        
        # Apply dropout
        proj_embeds = [self.dropout(embed) for embed in proj_embeds]
        
        # Calculate attention weights
        att_logits = self.att_layer(concat_embeds)
        att_weights = self.softmax(att_logits)
        
        # Apply attention weights to each modality
        att_embeds = []
        for i, embed in enumerate(proj_embeds):
            # Multiply attention weights with the embeddings
            att_embed = torch.mul(embed, att_weights[:, i])
            
            att_embeds.append(att_embed)
        
        # Fuse all attended features by summation
        fusion_vector = torch.sum(att_embeds)
        
        return fusion_vector 
    
    
class IAFF(nn.Module):
    """
    Inter-Attention Feature Fusion (IAFF) module.
    
    This model applies an inter-attention mechanism to efficiently extract and fuse features from multiple modalities. It computes attention scores within each modality and between modalities interactively, ensuring that the most critical information is retained. The unimodal information is added back to prevent loss of essential features.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int, dropout=True) -> None:
        """
        Initializes the IAFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
            dropout (bool): Whether to apply dropout
        """
        super(IAFF, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Fully connected layers to project each modality into a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for input_dim in input_dims
        ])
        
        # Attention layer for computing weights
        self.att_layer = nn.Linear(sum(input_dims), hidden_dim)

        # Dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the IAFF module.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
            
        Returns:
            torch.Tensor: Fused features
        """
        # Concatenate embeddings
        concat_embeds = torch.cat(embeddings, dim=-1)
        
        # Project each modality into the common space
        proj_embeds = [
            layer(embed) for layer, embed in zip(self.proj_layers, embeddings)
        ]
        
        # Compute attention weights
        att_logits = self.att_layer(concat_embeds)
        att_weights = self.softmax(att_logits)

        # Compute inter-attention for each modality
        att_embeds = []
        for i, embed in enumerate(proj_embeds):
            # Multiply attention weights with the embeddings
            att_embed = torch.mul(embed, att_weights[:, i])
            
            # Apply softmax
            att_embed = self.softmax(att_embed)
            
            # Apply dropout
            att_embed = self.dropout(att_embed)
            
            # Multiply attention weights with normalized attention weights
            att_embed = torch.mul(embed, att_embed)
            
            # Add attended embedding to projected embedding
            att_embed = embed + att_embed
            
            # Apply dropout
            att_embed = self.dropout(att_embed)
            
            att_embeds.append(att_embed)

        # Fuse all attended features by summation
        fusion_vector = torch.sum(att_embed)
        
        # Apply dropout
        fusion_vector = self.dropout(fusion_vector)

        return fusion_vector
    
class CAFF(nn.Module):
    """
    Cross-Attention Feature Fusion (CAFF) module.
    
    This model uses cross-correlation to compute attention weights for uni-modal features. These weights are then applied to modify the relevance of each element in the feature vectors, generating discriminative and modality-enhanced representations. The final fused feature is obtained by concatenating the attended uni-modal features.
    
    Based on: "Active Speaker Recognition using Cross Attention Audio-Video Fusion"
    Source: https://ieeexplore.ieee.org/document/9922810
    """
    
    def __init__(self, input_dims: List[torch.Size], hidden_dim: int, dropout=True) -> None:
        """
        Initializes the CAFF module.
        
        Args:
            input_dims (List[torch.Size]): List of input dimensions of the uni-modal models.
            hidden_dim (int): Hidden dimension of the fully connected
            dropout (bool): Whether to apply dropout
        """
        super(CAFF, self).__init__()
        
        if len(input_dims) != 2:
            raise ValueError("CAFF requires exactly two input modalities.")
        
        # Fully connected layers to project each modality into a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for input_dim in input_dims
        ])
        
        # Learnable cross-correlation weights metrix
        self.cross_corr_matrix = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
        # Dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
            
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CAFF module.
        
        Args:
            embeddings (List[torch.Tensor]): List of embeddings from each uni-modal model.
            
        Returns:
            torch.Tensor: Fused features
        """
        # Project uni-modal features into a common space
        proj_embeds = [
            layer(embed) for layer, embed in zip(self.proj_layers, embeddings)
        ]
        
        # Apply dropout
        proj_embeds = [self.dropout(embed) for embed in proj_embeds]

        # Compute cross-correlation matrix
        cross_corr = torch.matmul(proj_embeds[0], self.cross_corr_matrix) @ proj_embeds[1].T

        # Derive cross-attention weights via column-wise softmax
        cross_att_weights = [
            self.softmax(cross_corr, dim=-1),
            self.softmax(cross_corr.T, dim=-1)
        ]

        # Compute attention-weighted features
        att_embeds = [torch.matmul(catt_weight, embed) for catt_weight, embed in zip(cross_att_weights, proj_embeds)]
        
        # Apply skip connections and nonlinearity via tanh
        refined_embeds = [torch.tanh(embed + att_embed) for embed, att_embed in zip(proj_embeds, att_embeds)]

        # Concatenate refined features to obtain final fused representation
        fusion_vector = torch.cat(refined_embeds, dim=-1)

        return fusion_vector