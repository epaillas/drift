"""Galaxy EFT parameter containers."""

from dataclasses import dataclass


@dataclass
class GalaxyEFTParameters:
    """EFT bias and nuisance parameters for the galaxy tracer."""

    b1: float
    b2: float = 0.0
    bs2: float = 0.0
    b3nl: float = 0.0
    sigma_fog: float = 0.0
    c0: float = 0.0
    c2: float = 0.0
    c4: float = 0.0
    s0: float = 0.0
    s2: float = 0.0


GalaxyEFTParams = GalaxyEFTParameters
