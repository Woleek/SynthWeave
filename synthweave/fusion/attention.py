import torch
import torch.nn as nn
from typing import List

from .base import BaseFusion
from ..utils.modules import LazyLinearXavier

class AFF(BaseFusion):
    """
    Attention Feature Fusion (AFF) module.
    
    Uses an attention mechanism to calculate attention weights for each modality based on their quality and significance. It dynamically adjusts the weights and fuses the modalities into a robust representation.
    
    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845
    """
    
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """
        Initializes the AFF module.
        """
        super(AFF, self).__init__(dropout, unify_embeds)
        
        self._output_dim = output_dim
        
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
        
        # Attention layer for computing weights
        self.att_layer = LazyLinearXavier(output_dim)
        
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
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        # Concatenate embeddings
        concat_embeds = torch.cat(proj_embeds, dim=-1)
        
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
    
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """
        Initializes the IAFF module.
        """
        super(IAFF, self).__init__(dropout, unify_embeds)
        
        self._output_dim = output_dim
        
        # Fully connected layers to project each modality into a common space
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
        
        # Attention layer for computing weights
        self.att_layer = LazyLinearXavier(output_dim)

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
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        # Concatenate embeddings
        concat_embeds = torch.cat(proj_embeds, dim=-1)
        
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
        # fusion_vector = self.dropout(fusion_vector)

        return fusion_vector
    
class CAFF(BaseFusion):
    """
    Cross-Attention Feature Fusion (CAFF) module.
    
    This model uses cross-correlation to compute attention weights for uni-modal features. These weights are then applied to modify the relevance of each element in the feature vectors, generating discriminative and modality-enhanced representations. The final fused feature is obtained by concatenating the attended uni-modal features.
    
    Based on: "Active Speaker Recognition using Cross Attention Audio-Video Fusion"
    Source: https://ieeexplore.ieee.org/document/9922810
    """
    
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """
        Initializes the CAFF module.
        """
        super(CAFF, self).__init__(dropout, unify_embeds)
        
        # Fully connected layers to project each modality into a common space
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
        
        # Learnable cross-correlation weights metrix
        self.cross_corr_matrix = nn.Parameter(torch.randn(output_dim, output_dim))
        
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
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
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