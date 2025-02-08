import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .base import BaseFusion
from ..utils.modules import LazyLinearXavier, LinearXavier

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
        modality_keys: List[str],
        
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        
        **kwargs
    ) -> None:
        """
        Initializes the AFF module.
        """
        super(AFF, self).__init__(modality_keys, input_dims, bias, dropout, unify_embeds, hidden_proj_dim, out_proj_dim, normalize_proj)
        
        num_att_heads: int = kwargs.get("num_att_heads", 1)
        
        # Attention layer for computing weights
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_att_heads,
            dropout=dropout,
            batch_first=True
        )
        
        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the AFF module.
        """        
        # Stack embeddings
        mod_embeds = torch.stack(list(embeddings.values()), dim=1) # (batch_size, num_modals, embed_dim)
        
        # Apply attention (self-attention over modalities)
        att_emb, att_weights = self.attention_layer(mod_embeds, mod_embeds, mod_embeds)
        
        # Fuse all attended features by summing over modalities
        fusion_vector = att_emb.sum(dim=1) # (batch_size, embed_dim)
        
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
        modality_keys: List[str],
        
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        
        **kwargs
    ) -> None:
        """
        Initializes the IAFF module.
        """
        super(IAFF, self).__init__(modality_keys, input_dims, bias, dropout, unify_embeds, hidden_proj_dim, out_proj_dim, normalize_proj)
        
        num_att_heads: int = kwargs.get("num_att_heads", 1)
        
        # Multihead Attention for inter-attention between modalities
        self.inter_attention_layer = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_att_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the IAFF module.
        """
        # Stack modality embeddings for inter-attention
        mod_embeds = torch.stack(list(embeddings.values()), dim=1) # (batch_size, num_modals, embed_dim)
        
        # Inter-attention mechanism (Self-Attention across modalities)
        att_emb, attn_weights = self.inter_attention_layer(mod_embeds, mod_embeds, mod_embeds)
        
        # Apply softmax normalization
        att_emb_norm = self.softmax(att_emb)

        # Compute modality-enhanced embeddings with skip connections
        enhanced_embeds = [
            torch.tanh(emb + att_emb_norm[:, i, :])  # Add attention-enhanced embedding
            for i, emb in enumerate(list(embeddings.values()))
        ]

        # Fuse all attended features by summation
        fusion_vector = torch.stack(enhanced_embeds, dim=0).sum(dim=0) # (batch_size, embed_dim)

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
        modality_keys: List[str],
        
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        
        **kwargs
    ) -> None:
        """
        Initializes the CAFF module.
        """
        super(CAFF, self).__init__(modality_keys, input_dims, bias, dropout, unify_embeds, hidden_proj_dim, out_proj_dim, normalize_proj)
        
        num_att_heads: int = kwargs.get("num_att_heads", 1)
        
        # Multihead Attention Layers (one for each modality pair)
        self.cross_attention_layers = nn.ModuleDict([
            (f"{key1}_{key2}", nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_att_heads,
                dropout=dropout,
                batch_first=True
            ))
            for i, key1 in enumerate(modality_keys)
            for j, key2 in enumerate(modality_keys)
            if i != j
        ])
        
        # Aggregation layer
        self.aggregation_layer = LinearXavier(output_dim * len(modality_keys), output_dim, bias)
        
        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the CAFF module.
        """
        # Compute cross-attention for each modality pair
        attended_embeds = []
        for key1, embed1 in embeddings.items():
            for key2, embed2 in embeddings.items():
                if key1 == key2:
                    continue
                
                # Apply cross-attention
                att_emb, att_weight = self.cross_attention_layers[f"{key1}_{key2}"](embed1.unsqueeze(1), embed2.unsqueeze(1), embed2.unsqueeze(1))
                attended_embeds.append(att_emb.squeeze(1))
        
        # Combine attended features with original embeddings via skip connection and nonlinearity via tanh
        refined_embeds = [
            torch.tanh(embed + attended)
            for embed, attended in zip(list(embeddings.values()), attended_embeds)
        ]

        # Concatenate refined features to obtain fused representation
        concat_refined_embeds = torch.cat(refined_embeds, dim=-1) # (batch_size, num_modals * embed_dim)
        
        # Apply aggregation layer
        fusion_vector = self.aggregation_layer(concat_refined_embeds) # (batch_size, embed_dim)

        return fusion_vector