from typing import Dict, List, Optional
from ..fusion import FUSION_MAP, FusionType
from ..fusion.base import BaseFusion

def get_fusion(
    fusion_name: FusionType, 
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
) -> BaseFusion:
    """
    Get fusion module by name.
    
    Options:
    - CFF: Concatenation Feature Fusion
    - GFF: Gating Feature Fusion
    - AFF: Attention Feature Fusion
    - IAFF: Inter-Attention Feature Fusion
    - CAFF: Cross-Attention Feature Fusion
    - MV: Majority Voting
    - ASF: Average Score Fusion
    - SF: Score Fusion
    - MMD: Multi-Modal Joint-Decoder
    """
    fusion_module = FUSION_MAP.get(fusion_name.upper(), None)
    if not fusion_module:
        raise ValueError(f"Unknown fusion module: {fusion_name}")
    else:
        return fusion_module(
            output_dim, modality_keys, input_dims, bias, dropout, unify_embeds, hidden_proj_dim, out_proj_dim, normalize_proj, **kwargs
        )