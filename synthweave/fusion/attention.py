import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .base import BaseFusion

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
        
        # Modality-specific FC layers
        self.proj = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(attn_dim, self.output_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            )
            for k in self.modalities
        })

        # Attention layer: simple MLP to produce attention scores
        self.attention_layer = nn.Sequential(
            nn.Linear(attn_dim * len(self.modalities), self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, len(self.modalities))
        )
        
        self.softmax = nn.Softmax(dim=1)

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the AFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        # Project each modality
        projected = [self.proj[k](embeddings[k]) for k in self.modalities]  # list of (B, PROJ)
        
        # Concatenate raw embeddings for attention
        concat = torch.cat([embeddings[k] for k in self.modalities], dim=1)  # (B, PROJ * M)
        att_logits = self.attention_layer(concat)  # (B, M)
        att_weights = self.softmax(att_logits)  # (B, M)
        
        # Multiply attention weights by projected features
        weighted = [
            att_weights[:, i].unsqueeze(1) * projected[i]
            for i in range(len(self.modalities))
        ]
        
        # Sum for fusion
        fusion_vector = sum(weighted)  # (B, PROJ)

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
        
        # FC for each modality
        self.proj = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(attn_dim, self.output_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
            for k in self.modalities
        })
        
        # Attention layer: takes concatenated raw embeddings, outputs 2 attention scores (one for each modality)
        self.attention_layer = nn.Sequential(
            nn.Linear(attn_dim * len(self.modalities), self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, len(self.modalities))
        )

        # Dropout & softmax
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=1)  # softmax over modalities

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the IAFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        projected = [self.proj[k](embeddings[k]) for k in self.modalities]
        concat = torch.cat([embeddings[k] for k in self.modalities], dim=1)
        gate_logits = self.attention_layer(concat)

        proj_stack = torch.stack(projected, dim=1)
        proj_norm = torch.nn.functional.normalize(proj_stack, dim=-1)
        sim = torch.matmul(proj_norm, proj_norm.transpose(1, 2))
        inter_term = sim.mean(dim=-1)

        attn_weights = self.softmax(gate_logits + inter_term)
        fusion = sum(
            attn_weights[:, i].unsqueeze(1) * projected[i]
            for i in range(len(self.modalities))
        )
        return self.dropout(fusion)


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

        # Learnable weight matrices for each modality pair (i ≠ j)
        self.cross_weights = nn.ParameterDict()
        self.align_layers = nn.ModuleDict()
        for mi in self.modalities:
            for mj in self.modalities:
                if mi != mj:
                    self.cross_weights[f"{mi}_{mj}"] = nn.Parameter(
                        torch.randn(self.proj_dim, self.proj_dim)
                    )
                    self.align_layers[f"{mi}_{mj}"] = nn.Identity()

        # Output FC layer after flatten+concat
        self.fc = nn.Linear(self.proj_dim * len(self.modalities), self.output_dim, bias=bias)
        
        # Sotfmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        print("[INFO] This fusion expects embeddings of shape (batch_size, embed_dim).")

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the CAFF module.

        Args:
            embeddings: Dictionary mapping modality names to their feature tensors
                       Shape: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Fused representation with shape (batch_size, embed_dim)
        """
        if all(v.dim() == 2 for v in embeddings.values()):
            processed = {}

            for mi in self.modalities:
                xi = embeddings[mi]  # (B, D)
                attended = []
                for mj in self.modalities:
                    if mi == mj:
                        continue
                    xj = embeddings[mj]  # (B, D)
                    W = self.cross_weights[f"{mi}_{mj}"]  # (D, D)

                    # Per-feature cross-gating attention
                    xiW = torch.matmul(xi, W)  # (B, D)
                    corr = xiW * xj  # (B, D)
                    gate = torch.sigmoid(corr)
                    xj_att = gate * xj

                    attended.append(torch.tanh(xi + xj_att))

                processed[mi] = torch.stack(attended, dim=0).mean(dim=0)  # (B, D)

            fusion = torch.cat([processed[k] for k in self.modalities], dim=1)
            fusion = self.dropout(fusion)
            return self.fc(fusion)

        # embeddings: dict {modality: (B, D) or (B, K, D)}
        processed = {}

        # Convert all embeddings to (B, K, D), where K=1 if not present
        reshaped = {}
        for k, v in embeddings.items():
            if v.dim() == 2:
                v = v.unsqueeze(1)  # (B, 1, D)
            reshaped[k] = v

        for mi in self.modalities:
            # Gather all cross-attended features from other modalities
            attended = []
            xi = reshaped[mi]  # (B, K, Di)
            for mj in self.modalities:
                if mi == mj:
                    continue
                xj = reshaped[mj]  # (B, K, Dj)
                W = self.cross_weights[f"{mi}_{mj}"]  # (Di, Dj)
                # Cross-correlation
                xiW = torch.matmul(xi, W)  # (B, K, Dj)
                # Cross-corr matrix: (B, K, K)
                C = torch.matmul(xiW, xj.transpose(1, 2))
                # Attention: softmax over last dim
                attn = self.softmax(C)  # (B, K, K)
                # Attend to xj: (B, K, K) x (B, K, Dj) --> (B, K, Dj)
                xj_att = torch.matmul(attn, xj)  # (B, K, Dj)
                if xj_att.shape[-1] != xi.shape[-1]:
                    xj_att = self.align_layers[f"{mi}_{mj}"](xj_att)
                # Skip connection + tanh
                attended.append(torch.tanh(xi + xj_att))
            # Aggregate attended versions (e.g., average)
            fused = torch.stack(attended, dim=0).mean(dim=0)  # (B, K, Di)
            processed[mi] = fused

        # Flatten and concatenate all modalities
        outs = []
        for k in self.modalities:
            out = processed[k].flatten(1)  # (B, K*D)
            outs.append(out)
        fusion = torch.cat(outs, dim=1)  # (B, sum(K*D))
        fusion = self.dropout(fusion)
        fusion = self.fc(fusion)
        return fusion
