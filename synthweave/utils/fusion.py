from ..fusion import FUSION_MAP, FusionType
from ..fusion.base import BaseFusion

from typing import List, Optional

def get_fusion(fusion_name: FusionType, output_dim: int, dropout: bool = True, input_dims: Optional[List[int]] = None) -> BaseFusion:
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
    """
    fusion_module = FUSION_MAP.get(fusion_name.upper(), None)
    if not fusion_module:
        raise ValueError(f"Unknown fusion module: {fusion_name}")
    else:
        return fusion_module(output_dim, dropout, input_dims)