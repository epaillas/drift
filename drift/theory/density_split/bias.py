"""Density-split parameter containers."""

from dataclasses import dataclass, field


@dataclass
class DensitySplitBin:
    """Bias parameters for one density-split bin q_i.

    Attributes
    ----------
    label : str
        Human-readable label, e.g. 'DS1'.
    bq : float
        Linear density-split bias.
    cq : float
        Coefficient of the (kR)^2 derivative bias term. Default 0.
    beta_q : float
        Phenomenological anisotropy parameter. Default 0.
    """

    label: str
    bq: float
    cq: float = 0.0
    beta_q: float = 0.0


@dataclass
class DensitySplitEFTParameters:
    """EFT bias parameters for one density-split bin."""

    label: str
    bq1: float
    bq2: float = 0.0
    bqK2: float = 0.0
    bq_nabla2: float = 0.0
    beta_q: float = 0.0


def default_split_bins(n: int = 5) -> list:
    """Return a list of n DSSplitBin objects with placeholder bq values.

    The bq values are evenly spaced from -1.5 (most underdense) to +1.5
    (most overdense), matching the typical sign convention where DS1 is
    the emptiest quintile.

    Parameters
    ----------
    n : int
        Number of density bins.

    Returns
    -------
    list of DSSplitBin
    """
    import numpy as np

    bq_values = np.linspace(-1.5, 1.5, n)
    return [DensitySplitBin(label=f"DS{i+1}", bq=float(bq)) for i, bq in enumerate(bq_values)]


DSSplitBin = DensitySplitBin
DSSplitBinEFT = DensitySplitEFTParameters
