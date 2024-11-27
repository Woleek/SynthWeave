import torch
import torch.nn as nn
from typing import List

from .base import BaseFusion

class AFF(BaseFusion):
    """
    Attention Feature Fusion (AFF) module.
    
    Uses an attention mechanism to calculate attention weights for each modality based on their quality and significance. It dynamically adjusts the weights and fuses the modalities into a robust representation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def _lazy_init(self) -> None:
        """
        Initializes the AFF module.
        """
        
        # Linear layers to map uni-modal embeddings to a common hidden dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim) 
            for input_dim 
            in self._input_dims
        ])
        
        # Attention layer for computing weights
        self.att_layer = nn.Linear(sum(self._input_dims), self._output_dim)
        
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
            
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the AFF module.
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
            att_weight = att_weights[:, i].detach().unsqueeze(dim=1).repeat(1, self._output_dim)
            att_embed = torch.mul(embed, att_weight)
            
            att_embeds.append(att_embed)
        
        # Fuse all attended features by summation
        fusion_vector = torch.stack(att_embeds, dim=0).sum(dim=0)
        
        return fusion_vector 
    
    
class IAFF(BaseFusion):
    """
    Inter-Attention Feature Fusion (IAFF) module.
    
    This model applies an inter-attention mechanism to efficiently extract and fuse features from multiple modalities. It computes attention scores within each modality and between modalities interactively, ensuring that the most critical information is retained. The unimodal information is added back to prevent loss of essential features.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def _lazy_init(self) -> None:
        """
        Initializes the IAFF module.
        """
        # Fully connected layers to project each modality into a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim) for input_dim in self._input_dims
        ])
        
        # Attention layer for computing weights
        self.att_layer = nn.Linear(sum(self._input_dims), self._output_dim)

        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the IAFF module.
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
            att_weight = att_weights[:, i].detach().unsqueeze(dim=1).repeat(1, self._output_dim)
            att_embed = torch.mul(embed, att_weight)
            
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
        fusion_vector = torch.stack(att_embeds, dim=0).sum(dim=0)
        
        # Apply dropout
        fusion_vector = self.dropout(fusion_vector)

        return fusion_vector
    
class CAFF(BaseFusion):
    """
    Cross-Attention Feature Fusion (CAFF) module.
    
    This model uses cross-correlation to compute attention weights for uni-modal features. These weights are then applied to modify the relevance of each element in the feature vectors, generating discriminative and modality-enhanced representations. The final fused feature is obtained by concatenating the attended uni-modal features.
    
    Based on: "Active Speaker Recognition using Cross Attention Audio-Video Fusion"
    Source: https://ieeexplore.ieee.org/document/9922810
    """
    
    def _lazy_init(self) -> None:
        """
        Initializes the CAFF module.
        """
        if len(self._input_dims) != 2:
            raise ValueError("CAFF requires exactly two input modalities.")
        
        # Fully connected layers to project each modality into a common space
        self.proj_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim // 2) for input_dim in self._input_dims
        ])
        
        # Learnable cross-correlation weights metrix
        self.cross_corr_matrix = nn.Parameter(torch.randn(self._output_dim // 2, self._output_dim // 2))
        
        # Dropout
        if self._dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
            
        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CAFF module.
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
            self.softmax(cross_corr),
            self.softmax(cross_corr.T)
        ]

        # Compute attention-weighted features
        att_embeds = [torch.matmul(catt_weight, embed) for catt_weight, embed in zip(cross_att_weights, proj_embeds)]
        
        # Apply skip connections and nonlinearity via tanh
        refined_embeds = [torch.tanh(embed + att_embed) for embed, att_embed in zip(proj_embeds, att_embeds)]

        # Concatenate refined features to obtain final fused representation
        fusion_vector = torch.cat(refined_embeds, dim=-1)

        return fusion_vector