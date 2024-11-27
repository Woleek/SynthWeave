from typing import Literal

from .attention import AFF, IAFF, CAFF
from .concat import CFF
from .gating import GFF

FUSION_MAP = {
    "CFF": CFF,
    "GFF": GFF,
    "AFF": AFF,
    "IAFF": IAFF,
    "CAFF": CAFF
}

FusionType = Literal["CFF", "GFF", "AFF", "IAFF", "CAFF"]