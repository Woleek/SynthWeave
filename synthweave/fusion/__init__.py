"""
Multimodal fusion module initialization.

This module provides a collection of fusion methods for combining features from
multiple modalities. It includes various fusion strategies:

Attention-based:
    - AFF (Attention Feature Fusion)
    - IAFF (Interactive Attention Feature Fusion)
    - CAFF (Cross Attention Feature Fusion)

Concatenation-based:
    - CFF (Concatenation Feature Fusion)

Gating-based:
    - GFF (Gating Feature Fusion)

Classification-based:
    - MV (Majority Voting)
    - ASF (Average Score Fusion)
    - SF (Score Fusion)

Advanced:
    - MMD (Multi-Modal Joint-Decoder)

The module provides a mapping from fusion type names to their implementations
and a type definition for fusion method selection.
"""

from typing import Literal

from .attention import AFF, IAFF, CAFF
from .concat import CFF
from .gating import GFF
from .classifier import MV, ASF, SF
from .mmd import MMD

FUSION_MAP = {
    "CFF": CFF,  # Concatenation Feature Fusion
    "GFF": GFF,  # Gating Feature Fusion
    "AFF": AFF,  # Attention Feature Fusion
    "IAFF": IAFF,  # Interactive Attention Feature Fusion
    "CAFF": CAFF,  # Cross Attention Feature Fusion
    "MV": MV,  # Majority Voting
    "ASF": ASF,  # Average Score Fusion
    "SF": SF,  # Score Fusion
    "MMD": MMD,  # Multi-Modal Joint-Decoder
}

FusionType = Literal["CFF", "GFF", "AFF", "IAFF", "CAFF", "MV", "ASF", "SF", "MMD"]
"""Type definition for available fusion methods.

This type ensures type safety when selecting fusion methods and enables
IDE autocompletion support.

Available fusion methods:
    - "CFF": Concatenation Feature Fusion
    - "GFF": Gating Feature Fusion
    - "AFF": Attention Feature Fusion
    - "IAFF": Interactive Attention Feature Fusion
    - "CAFF": Cross Attention Feature Fusion
    - "MV": Majority Voting
    - "ASF": Average Score Fusion
    - "SF": Score Fusion
    - "MMD": Multi-Modal Joint-Decoder
"""
