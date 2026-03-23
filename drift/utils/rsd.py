"""Redshift-space distortion (Kaiser) factors."""

import numpy as np


def matter_rsd_factor(mu: np.ndarray, f: float) -> np.ndarray:
    """Kaiser RSD factor for matter: (1 + f*mu^2).

    Parameters
    ----------
    mu : array_like
        Cosine of angle to line of sight.
    f : float
        Linear growth rate.

    Returns
    -------
    np.ndarray
    """
    mu = np.asarray(mu, dtype=float)
    return 1.0 + f * mu ** 2


def tracer_rsd_factor(mu: np.ndarray, b1: float, f: float) -> np.ndarray:
    """Kaiser RSD factor for a biased tracer: (b1 + f*mu^2).

    Parameters
    ----------
    mu : array_like
        Cosine of angle to line of sight.
    b1 : float
        Linear galaxy bias.
    f : float
        Linear growth rate.

    Returns
    -------
    np.ndarray
    """
    mu = np.asarray(mu, dtype=float)
    return b1 + f * mu ** 2
