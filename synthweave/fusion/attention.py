import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .base import BaseFusion
from ..utils.modules import LazyLinearXavier, LinearXavier

"""
Attention-based fusion modules for multimodal feature fusion.

This module implements various attention mechanisms for fusing features from multiple
modalities, including AFF (Attention Feature Fusion), IAFF (Inter-Attention Feature Fusion),
and CAFF (Cross-Attention Feature Fusion).
"""


class AFF(BaseFusion):
    """Attention Feature Fusion (AFF) module.

    Uses an attention mechanism to calculate attention weights for each modality based on
    their quality and significance. It dynamically adjusts the weights and fuses the
    modalities into a robust representation.

    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845

    Attributes:
        attention_layer: MultiheadAttention layer for computing attention weights

    Note:
        Expects embeddings of shape (batch_size, embed_dim)
    """

    def __init__(
        self,
        output_dim: int,
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the AFF module.

        Args:
            output_dim: Dimension of the output features
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in linear layers
            dropout: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments including num_att_heads
        """
        super(AFF, self).__init__(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout_p,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        num_att_heads: int = kwargs.get("num_att_heads", 1)

        attn_dim = self.proj_dim
        assert attn_dim is not None, (
            "AFF needs equal-sized modality embeddings. "
            "Either pass unify_embeds=True or supply out_proj_dim."
        )

        # Attention layer for computing weights
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_att_heads,
            dropout=dropout_p,
            batch_first=True,
        )

        # Project to desired output dimension if needed
        self.post_proj = (
            LinearXavier(attn_dim, output_dim, bias=False)
            if attn_dim != output_dim
            else nn.Identity()
        )

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the AFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        # Stack embeddings
        mod_embeds = torch.stack(
            [embeddings[k] for k in self.modalities], dim=1
        )  # (B, M, PROJ)

        # Apply attention (self-attention over modalities)
        att_emb, att_weights = self.attention_layer(
            mod_embeds, mod_embeds, mod_embeds
        )  # (B, M, PROJ)

        # Fuse all attended features by summing over modalities
        fusion_vector = att_emb.sum(dim=1)  # (B, PROJ)

        # Apply post-attention projection if needed
        fusion_vector = self.post_proj(fusion_vector)  # (B, EMB)

        return fusion_vector


class IAFF(BaseFusion):
    """Inter-Attention Feature Fusion (IAFF) module.

    Applies an inter-attention mechanism to efficiently extract and fuse features from
    multiple modalities. Computes attention scores within each modality and between
    modalities interactively, ensuring critical information retention.

    Based on: "Audio-Visual Fusion Based on Interactive Attention for Person Verification"
    Source: https://www.mdpi.com/1424-8220/23/24/9845

    Attributes:
        inter_attention_layer: MultiheadAttention layer for inter-modality attention
        softmax: Softmax layer for attention normalization

    Note:
        Expects embeddings of shape (batch_size, embed_dim)
    """

    def __init__(
        self,
        output_dim: int,
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the IAFF module.

        Args:
            output_dim: Dimension of the output features
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in linear layers
            dropout: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments including num_att_heads
        """
        super(IAFF, self).__init__(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout_p,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        num_att_heads: int = kwargs.get("num_att_heads", 1)

        attn_dim = self.proj_dim
        assert attn_dim is not None, (
            "IAFF needs equal-size modality embeddings. "
            "Use unify_embeds=True or set out_proj_dim."
        )

        # Multihead Attention for inter-attention between modalities
        self.inter_attention_layer = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_att_heads,
            dropout=dropout_p,
            batch_first=True,
        )

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)

        # Project to desired output dimension if needed
        self.post_proj = (
            LinearXavier(attn_dim, output_dim, bias=False)
            if attn_dim != output_dim
            else nn.Identity()
        )

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the IAFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        # Stack modality embeddings for inter-attention
        mod_embeds = torch.stack(
            [embeddings[k] for k in self.modalities], dim=1
        )  # (B, M, PROJ)

        # Inter-attention mechanism (Self-Attention across modalities)
        att_emb, attn_weights = self.inter_attention_layer(
            mod_embeds, mod_embeds, mod_embeds
        )  # (B, M, PROJ)

        # Scalar gates (softmax over feature dimension)
        sim = (mod_embeds * att_emb).sum(dim=-1)  # (B, M)
        gates = self.softmax(sim / math.sqrt(self.proj_dim))  # (B, M)
        gates = gates.unsqueeze(-1)  # (B, M, 1)

        # Enhance embeddings with residuals and gates
        enhanced_embeds = mod_embeds + gates * att_emb  # (B, M, PROJ)

        # Apply dropout
        enhanced_embeds = self.dropout(enhanced_embeds)  # (B, M, PROJ)

        # Sum and apply dropout again
        fusion_vector = self.dropout(enhanced_embeds.sum(dim=1))  # (B, PROJ)

        # Project to output dimension if needed
        fusion_vector = self.post_proj(fusion_vector)  # (B, EMB)

        return fusion_vector


class CAFF(BaseFusion):
    """Cross-Attention Feature Fusion (CAFF) module.

    Uses cross-correlation to compute attention weights for uni-modal features.
    These weights modify the relevance of each feature vector element, generating
    discriminative and modality-enhanced representations.

    Based on: "Active Speaker Recognition using Cross Attention Audio-Video Fusion"
    Source: https://ieeexplore.ieee.org/document/9922810

    Attributes:
        cross_attention_layers: ModuleDict of MultiheadAttention layers for each modality pair
        aggregation_layer: Linear layer for final feature aggregation

    Note:
        Expects embeddings of shape (batch_size, embed_dim)
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
        **kwargs,
    ) -> None:
        """Initialize the CAFF module.

        Args:
            output_dim: Dimension of the output features
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in linear layers
            dropout: Dropout probability
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection
            **kwargs: Additional arguments including num_att_heads
        """
        super(CAFF, self).__init__(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
        )

        num_att_heads: int = kwargs.get("num_att_heads", 1)

        attn_dim = self.proj_dim
        assert attn_dim is not None, (
            "IAFF needs equal-size modality embeddings. "
            "Use unify_embeds=True or set out_proj_dim."
        )

        # Multihead Attention Layers (one for each modality pair)
        self.cross_attention_layers = nn.ModuleDict(
            [
                (
                    f"{key1}_{key2}",
                    nn.MultiheadAttention(
                        embed_dim=attn_dim,
                        num_heads=num_att_heads,
                        dropout=dropout,
                        batch_first=True,
                    ),
                )
                for i, key1 in enumerate(modality_keys)
                for j, key2 in enumerate(modality_keys)
                if i != j
            ]
        )

        # Aggregation layer
        self.aggregation_layer = LinearXavier(
            self.proj_dim * len(modality_keys), output_dim, bias
        )

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the CAFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        # Compute cross-attention for each modality pair
        sum_att = {k: torch.zeros_like(v) for k, v in embeddings.items()}
        for i, m1 in enumerate(self.modalities):
            for j, m2 in enumerate(self.modalities):
                if i == j:
                    continue
                att, _ = self.cross_attention_layers[f"{m1}_{m2}"](
                    embeddings[m1].unsqueeze(1),  # query  (B,1,PROJ)
                    embeddings[m2].unsqueeze(1),  # key    (B,1,PROJ)
                    embeddings[m2].unsqueeze(1),  # value  (B,1,PROJ)
                )  # att → (B,1,PROJ)
                sum_att[m1] += att.squeeze(1)  # accumulate (B,PROJ)

        # Combine attended features with original embeddings via skip connection and nonlinearity via tanh
        refined_embeds = [
            torch.tanh(embeddings[m] + sum_att[m]) for m in self.modalities
        ]  # [(B, PROJ)]

        # Apply aggregation layer
        fusion_vector = self.aggregation_layer(
            torch.cat(refined_embeds, dim=-1)
        )  # (B, EMB)

        return fusion_vector
