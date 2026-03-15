"""EFT bias parameter containers for density-split and galaxy tracers."""

from dataclasses import dataclass


@dataclass
class DSSplitBinEFT:
    """EFT bias parameters for one density-split bin q_i.

    Attributes
    ----------
    label : str
        Human-readable label, e.g. 'DS1'.
    bq1 : float
        Linear DS bias (replaces bq from DSSplitBin).
    bq2 : float
        Quadratic DS bias (enters at one-loop). Default 0.
    bqK2 : float
        Tidal DS bias (enters at one-loop). Default 0.
    bq_nabla2 : float
        Higher-derivative / EFT counterterm coefficient. Default 0.
    beta_q : float
        Phenomenological anisotropy parameter (kept for compat). Default 0.
    """

    label: str
    bq1: float
    bq2: float = 0.0
    bqK2: float = 0.0
    bq_nabla2: float = 0.0
    beta_q: float = 0.0


@dataclass
class GalaxyEFTParams:
    """EFT bias and nuisance parameters for the galaxy tracer.

    Attributes
    ----------
    b1 : float
        Linear galaxy bias.
    b2 : float
        Quadratic galaxy bias. Default 0.
    bs2 : float
        Tidal galaxy bias. Default 0.
    b3nl : float
        Non-local cubic galaxy bias. Default 0.
    c0 : float
        Isotropic EFT counterterm coefficient [(Mpc/h)^2]. Default 0.
    c2 : float
        mu^2 EFT counterterm coefficient [(Mpc/h)^2]. Default 0.
    c4 : float
        mu^4 EFT counterterm coefficient [(Mpc/h)^2]. Default 0.
    s0 : float
        White-noise stochastic amplitude [(Mpc/h)^3]. Default 0.
    s2 : float
        k^2 stochastic amplitude [(Mpc/h)^5]. Default 0.
    """

    b1: float
    b2: float = 0.0
    bs2: float = 0.0
    b3nl: float = 0.0
    c0: float = 0.0
    c2: float = 0.0
    c4: float = 0.0
    s0: float = 0.0
    s2: float = 0.0
