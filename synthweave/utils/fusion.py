from ..fusion import FUSION_MAP, FusionType
from ..fusion.base import BaseFusion

def get_fusion(fusion_name: FusionType, output_dim: int, n_modals: int, **kwargs) -> BaseFusion:
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
        return fusion_module(output_dim, n_modals, **kwargs)