"""EFT counterterms and stochastic contributions for DS x galaxy spectra."""

import numpy as np
from .eft_bias import DSSplitBinEFT, GalaxyEFTParams


def galaxy_counterterm(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    gal_params: GalaxyEFTParams,
    ds_amplitude: np.ndarray,
) -> np.ndarray:
    """Galaxy EFT counterterm for DS × galaxy cross-spectrum.

    P_ct^{gal}(k, mu) = ds_amplitude(k) * [-k^2 * P_lin(k) * (c0 + c2*mu^2 + c4*mu^4)]

    ds_amplitude encodes the DS-side weight bq1 * W_R(k).  For a galaxy
    auto-spectrum pass ds_amplitude = np.ones_like(k).

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    plin : array_like, shape (nk,)
        Linear matter power spectrum.
    gal_params : GalaxyEFTParams
    ds_amplitude : array_like, shape (nk,)
        DS-side amplitude: bq1 * W_R(k).

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    plin = np.asarray(plin, dtype=float)
    ds_amplitude = np.asarray(ds_amplitude, dtype=float)

    mu_shape = gal_params.c0 + gal_params.c2 * mu**2 + gal_params.c4 * mu**4  # (nmu,)
    k2_plin = k**2 * plin                                                        # (nk,)
    return -(ds_amplitude * k2_plin)[:, np.newaxis] * mu_shape[np.newaxis, :]


def ds_counterterm(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    ds_params: DSSplitBinEFT,
    tree_normed: np.ndarray,
    R: float,
) -> np.ndarray:
    """DS higher-derivative counterterm.

    P_ct^{DS}(k, mu) = bq_nabla2 * (kR)^2 * tree_normed(k, mu)

    where tree_normed is the tree model evaluated at bq1 = 1 (i.e., the
    angular shape factor).

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    plin : array_like, shape (nk,)
    ds_params : DSSplitBinEFT
    tree_normed : np.ndarray, shape (nk, nmu)
        Tree-level model evaluated with bq1 = 1.
    R : float
        Smoothing radius in Mpc/h.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    kR_sq = (k * R) ** 2   # (nk,)
    return ds_params.bq_nabla2 * kR_sq[:, np.newaxis] * tree_normed


def stochastic_term(
    k: np.ndarray,
    mu: np.ndarray,
    gal_params: GalaxyEFTParams,
) -> np.ndarray:
    """Stochastic (shot-noise) contribution.

    P_stoch(k, mu) = s0 + s2 * k^2   (constant in mu)

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    gal_params : GalaxyEFTParams

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    stoch_k = gal_params.s0 + gal_params.s2 * k**2   # (nk,)
    return stoch_k[:, np.newaxis] * np.ones((1, len(mu)))
