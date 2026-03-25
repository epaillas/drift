"""Galaxy EFT parameter containers."""

from dataclasses import dataclass


@dataclass
class GalaxyEFTParameters:
    """EFT bias and nuisance parameters for the galaxy tracer.

    Attributes
    ----------
    b1 : float
        Linear galaxy bias.
    b2 : float
        Second-order density bias. Default 0.
    bs2 : float
        Second-order tidal bias. Default 0.
    b3nl : float
        Non-local third-order bias. Default 0.
    sigma_fog : float
        Finger-of-God velocity dispersion in (Mpc/h)^2. Default 0.
    c0 : float
        Isotropic EFT counterterm amplitude. Default 0.
    c2 : float
        mu^2 EFT counterterm amplitude. Default 0.
    c4 : float
        mu^4 EFT counterterm amplitude. Default 0.
    s0 : float
        Constant (white-noise) stochastic power. Default 0.
    s2 : float
        k^2-dependent stochastic amplitude. Default 0.
    """

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
