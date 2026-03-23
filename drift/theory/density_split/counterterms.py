"""EFT counterterms and stochastic contributions for density-split spectra."""

import numpy as np


def galaxy_counterterm(
    k: np.ndarray,
    mu: np.ndarray,
    gal_params,
    ds_lin: np.ndarray,
) -> np.ndarray:
    """Galaxy EFT counterterm for DS × galaxy cross-spectrum.

    P_ct^{gal}(k, mu) = P_{DS×lin}(k, mu) * [-k^2 * (c0 + c2*mu^2 + c4*mu^4)]

    where P_{DS×lin}(k, mu) is the tree-level DS × linear matter cross-spectrum
    (no galaxy bias factor), encoding the full 2D angular structure of the DS field.

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    gal_params : GalaxyEFTParams
    ds_lin : array_like, shape (nk, nmu)
        DS × linear matter cross-spectrum: P_{DS×lin}(k, mu).

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ds_lin = np.asarray(ds_lin, dtype=float)

    ct_shape = gal_params.c0 + gal_params.c2 * mu**2 + gal_params.c4 * mu**4  # (nmu,)
    k2 = k**2                                                                    # (nk,)
    return -k2[:, np.newaxis] * ds_lin * ct_shape[np.newaxis, :]


def density_split_counterterm(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    ds_params,
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
    gal_params,
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


def density_split_pair_stochastic_term(
    k: np.ndarray,
    mu: np.ndarray,
    sqq0: float = 0.0,
    sqq2: float = 0.0,
) -> np.ndarray:
    """Isotropic stochastic contribution for DS-pair spectra."""
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    stoch_k = sqq0 + sqq2 * k**2
    return stoch_k[:, np.newaxis] * np.ones((1, len(mu)))


ds_counterterm = density_split_counterterm
