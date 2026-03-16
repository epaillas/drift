"""Tree-level and EFT galaxy auto-power spectrum P(k, mu)."""

import numpy as np

from .cosmology import get_linear_power, get_growth_rate
from .eft_bias import GalaxyEFTParams

_VALID_MODES = ("tree_only", "eft_lite", "eft_full")


def pgg_mu(k, mu, z, cosmo, b1, space="redshift"):
    """Tree-level Kaiser galaxy auto-power spectrum P_gg(k, mu).

    In redshift space: P_gg = (b1 + f*mu^2)^2 * P_lin(k)
    In real space:     P_gg = b1^2 * P_lin(k)

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
    b1 : float
        Linear galaxy bias.
    space : str
        'redshift' (default) or 'real'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    plin = get_linear_power(cosmo, k, z)   # (nk,)

    if space == "real":
        return (b1 ** 2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))

    f = get_growth_rate(cosmo, z)
    kaiser = b1 + f * mu ** 2   # (nmu,)
    return plin[:, np.newaxis] * kaiser[np.newaxis, :] ** 2


def pgg_eft_mu(k, mu, z, cosmo, gal_params, space="redshift", mode="eft_lite"):
    """EFT galaxy auto-power spectrum P_gg(k, mu).

    Modes
    -----
    tree_only : same as ``pgg_mu``
    eft_lite  : tree + galaxy EFT counterterm
        P_gg = (b1+f*mu^2)^2 * P_lin
               - 2*k^2*(c0+c2*mu^2+c4*mu^4)*(b1+f*mu^2)*P_lin
    eft_full  : eft_lite + stochastic term  s0 + s2*k^2

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    z : float
    cosmo : cosmoprimo.Cosmology
    gal_params : GalaxyEFTParams
    space : str
    mode : str

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {_VALID_MODES}.")

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    plin = get_linear_power(cosmo, k, z)   # (nk,)

    if space == "real":
        P = (gal_params.b1 ** 2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))
        if mode == "eft_full":
            P = P + (gal_params.s0 + gal_params.s2 * k ** 2)[:, np.newaxis]
        return P

    f = get_growth_rate(cosmo, z)

    if mode == "tree_only":
        return pgg_mu(k, mu, z, cosmo, gal_params.b1, space=space)

    b1 = gal_params.b1
    kaiser = b1 + f * mu ** 2   # (nmu,)
    P = plin[:, np.newaxis] * kaiser[np.newaxis, :] ** 2

    if mode in ("eft_lite", "eft_full"):
        c0 = gal_params.c0
        c2 = gal_params.c2
        c4 = gal_params.c4
        ct_shape = c0 + c2 * mu ** 2 + c4 * mu ** 4   # (nmu,)
        counterterm = (
            -2.0 * k ** 2 * plin
        )[:, np.newaxis] * ct_shape[np.newaxis, :] * kaiser[np.newaxis, :]
        P = P + counterterm

    if mode == "eft_full":
        P = P + (gal_params.s0 + gal_params.s2 * k ** 2)[:, np.newaxis]

    return P
