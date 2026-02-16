from typing import Dict, List, Optional

import torch.nn as nn


def get_fusion(
    fusion_name: str,
    output_dim: int,
    modality_keys: List[str],
    input_dims: Optional[Dict[str, int]] = None,
    bias: bool = True,
    dropout: float = 0.5,
    unify_embeds: bool = True,
    hidden_proj_dim: Optional[int] = None,
    out_proj_dim: Optional[int] = None,
    normalize_proj: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create fusion modules based on specified type.

    This function creates and returns a fusion module instance based on the provided
    fusion type. Available fusion types include CFF (Concatenation Feature Fusion),
    GFF (Gating Feature Fusion), AFF (Attention Feature Fusion), and others.

    Args:
        fusion_name: The type of fusion module to create (e.g., 'CFF', 'GFF', 'AFF')
        output_dim: The dimension of the output features
        modality_keys: List of modality names to be fused
        input_dims: Dictionary mapping modality names to their input dimensions
        bias: Whether to include bias in linear layers
        dropout: Dropout probability for regularization
        unify_embeds: Whether to project all modalities to same dimension
        hidden_proj_dim: Hidden dimension for projection layers
        out_proj_dim: Output dimension for projection layers
        normalize_proj: Whether to apply L2 normalization after projection
        **kwargs: Additional arguments passed to specific fusion modules

    Returns:
        BaseFusion: An instance of the specified fusion module

    Raises:
        ValueError: If the specified fusion_name is not found in FUSION_MAP

    Available Fusion Types:
        - CFF: Concatenation Feature Fusion
        - GFF: Gating Feature Fusion
        - AFF: Attention Feature Fusion
        - IAFF: Inter-Attention Feature Fusion
        - CAFF: Cross-Attention Feature Fusion
        - MV: Majority Voting
        - ASF: Average Score Fusion
        - SF: Score Fusion
        - MMD: Multi-Modal Joint-Decoder

    Example:
        >>> fusion_module = get_fusion(
        ...     fusion_name='CFF',
        ...     output_dim=256,
        ...     modality_keys=['text', 'image'],
        ...     input_dims={'text': 768, 'image': 1024}
        ... )
    """

    # Import fusion map here to avoid circular import
    from ..fusion import FUSION_MAP

    fusion_module = FUSION_MAP.get(fusion_name.upper(), None)
    if not fusion_module:
        raise ValueError(f"Unknown fusion module: {fusion_name}")
    else:
        return fusion_module(
            output_dim,
            modality_keys,
            input_dims,
            bias,
            dropout,
            unify_embeds,
            hidden_proj_dim,
            out_proj_dim,
            normalize_proj,
            **kwargs,
        )
