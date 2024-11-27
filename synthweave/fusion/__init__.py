from typing import Literal

from .attention import AFF, IAFF, CAFF
from .concat import CFF
from .gating import GFF
from .classifier import MV, ASF, SF

FUSION_MAP = {
    "CFF": CFF,
    "GFF": GFF,
    "AFF": AFF,
    "IAFF": IAFF,
    "CAFF": CAFF,
    "MV": MV,
    "ASF": ASF,
    "SF": SF
}

FusionType = Literal["CFF", "GFF", "AFF", "IAFF", "CAFF", "MV", "ASF", "SF"]