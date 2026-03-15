"""EFT density-split × galaxy power spectrum assembler."""

import numpy as np
from .cosmology import get_linear_power, get_growth_rate
from .kernels import gaussian_kernel, tophat_kernel
from .eft_bias import DSSplitBinEFT, GalaxyEFTParams
from .eft_terms import galaxy_counterterm, ds_counterterm, stochastic_term

_VALID_MODES = ("tree_only", "eft_lite", "eft_full")
_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")


def _get_kernel(kernel: str, k: np.ndarray, R: float) -> np.ndarray:
    if kernel == "gaussian":
        return gaussian_kernel(k, R)
    elif kernel == "tophat":
        return tophat_kernel(k, R)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian' or 'tophat'.")


def _pqg_tree_eft(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params: DSSplitBinEFT,
    gal_params: GalaxyEFTParams,
    ds_model: str,
) -> np.ndarray:
    """Tree-level DS x galaxy cross spectrum using EFT containers.

    Mirrors pqg_mu logic but uses bq1 (not bq) and gal_params.b1 (not
    tracer_bias). Supports all three ds_model branches.

    Parameters
    ----------
    k : np.ndarray, shape (nk,)
    mu : np.ndarray, shape (nmu,)
    plin : np.ndarray, shape (nk,)
    wk : np.ndarray, shape (nk,)
    f : float
    ds_params : DSSplitBinEFT
    gal_params : GalaxyEFTParams
    ds_model : str

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    b1 = gal_params.b1
    bq_eff = ds_params.bq1 * np.ones_like(k)   # (nk,); no k^2 term at tree level

    # Galaxy RSD factor: (b1 + f*mu^2)
    gal_rsd = b1 + f * mu**2   # (nmu,)

    if ds_model == "baseline":
        amplitude = bq_eff * plin * wk   # (nk,)
        return amplitude[:, np.newaxis] * gal_rsd[np.newaxis, :]

    elif ds_model == "rsd_selection":
        ds_factor = bq_eff[:, np.newaxis] * (1.0 + f * mu[np.newaxis, :] ** 2)
        return (plin * wk)[:, np.newaxis] * ds_factor * gal_rsd[np.newaxis, :]

    else:  # phenomenological
        beta_q = ds_params.beta_q
        ds_factor = bq_eff[:, np.newaxis] + beta_q * f * mu[np.newaxis, :] ** 2
        return (plin * wk)[:, np.newaxis] * ds_factor * gal_rsd[np.newaxis, :]


def _pqg_one_loop_partial(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    p1loop_matter: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params: DSSplitBinEFT,
    gal_params: GalaxyEFTParams,
    ds_model: str,
) -> np.ndarray:
    """Partial one-loop correction: promotes bq1*b1 term from Plin to P_1loop.

    Implemented correction:
        bq1 * b1 * (P_1loop - P_lin) * W_R * Kaiser-like angular factor

    Notes
    -----
    The following higher-order terms are NOT implemented and will raise
    NotImplementedError if accessed via non-zero parameters:
    - bq2 * <delta_R^2, galaxy 1loop>
    - bqK2 * <K_R^2, galaxy linear>
    - b2, bs2 cross-terms
    - Full redshift-space one-loop EFT-RSD kernels

    Parameters
    ----------
    k : np.ndarray, shape (nk,)
    mu : np.ndarray, shape (nmu,)
    plin : np.ndarray, shape (nk,)
    p1loop_matter : np.ndarray, shape (nk,)
    wk : np.ndarray, shape (nk,)
    f : float
    ds_params : DSSplitBinEFT
    gal_params : GalaxyEFTParams
    ds_model : str

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    # Guard against unimplemented higher-order terms
    if ds_params.bq2 != 0.0 or ds_params.bqK2 != 0.0:
        raise NotImplementedError(
            "bq2 and bqK2 one-loop cross-terms are not yet implemented. "
            "Set bq2=bqK2=0 for eft_lite/eft_full modes."
        )
    if gal_params.b2 != 0.0 or gal_params.bs2 != 0.0:
        raise NotImplementedError(
            "b2 and bs2 one-loop cross-terms are not yet implemented. "
            "Set b2=bs2=0 for eft_lite/eft_full modes."
        )

    b1 = gal_params.b1
    delta_p = p1loop_matter - plin   # (nk,)

    # Use the same angular factor as the tree-level bq1*b1 term
    gal_rsd = b1 + f * mu**2        # (nmu,)

    if ds_model == "baseline":
        amplitude = ds_params.bq1 * delta_p * wk   # (nk,)
        return amplitude[:, np.newaxis] * gal_rsd[np.newaxis, :]

    elif ds_model == "rsd_selection":
        ds_factor = ds_params.bq1 * (1.0 + f * mu[np.newaxis, :] ** 2)   # (nk, nmu) broadcast
        # ds_factor here doesn't depend on k, shape (1, nmu) -> broadcast with (nk, nmu)
        return (delta_p * wk)[:, np.newaxis] * ds_factor * gal_rsd[np.newaxis, :]

    else:  # phenomenological
        beta_q = ds_params.beta_q
        ds_factor = ds_params.bq1 + beta_q * f * mu**2   # (nmu,)
        return (delta_p * wk)[:, np.newaxis] * ds_factor[np.newaxis, :] * gal_rsd[np.newaxis, :]


def pqg_eft_mu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params: DSSplitBinEFT,
    gal_params: GalaxyEFTParams,
    R: float,
    kernel: str = "gaussian",
    space: str = "redshift",
    ds_model: str = "baseline",
    mode: str = "eft_lite",
) -> np.ndarray:
    """EFT density-split × galaxy cross power spectrum P(k, mu).

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
    ds_params : DSSplitBinEFT
        EFT bias parameters for the density-split bin.
    gal_params : GalaxyEFTParams
        EFT bias and nuisance parameters for the galaxy tracer.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str
        Smoothing kernel: 'gaussian' (default) or 'tophat'.
    space : str
        'redshift' (default) or 'real'.
    ds_model : str
        Redshift-space model for the DS field: 'baseline' (default),
        'rsd_selection', or 'phenomenological'.
    mode : str
        EFT mode:
        - 'tree_only': tree-level only (matches pqg_mu exactly at bq=bq1)
        - 'eft_lite': tree + galaxy counterterm + DS counterterm (no loop promotion;
          raw SPT P13 is UV-sensitive and cannot be renormalized by the k² counterterm
          basis at the current integration range — see _pqg_one_loop_partial)
        - 'eft_full': eft_lite + stochastic terms

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {_VALID_MODES}.")
    if ds_model not in _VALID_DS_MODELS:
        raise ValueError(f"Unknown ds_model '{ds_model}'. Choose one of {_VALID_DS_MODELS}.")

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)   # (nk,)
    wk = _get_kernel(kernel, k, R)         # (nk,)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    if mode == "tree_only":
        return _pqg_tree_eft(k, mu, plin, wk, f, ds_params, gal_params, ds_model)

    # eft_lite or eft_full: tree + counterterms (no one-loop promotion yet)
    # NOTE: _pqg_one_loop_partial() exists but is intentionally not called here.
    # The raw SPT P13 produces |2P13/Plin| ~ 13-15 at k<0.05 (see
    # test_P13_not_over_normalized), which is not k²-suppressed and cannot be
    # renormalized by the k²*Plin EFT counterterm basis.  The loop-promotion
    # step will be re-enabled once a renormalized P13 (delta_p_ren = P22+P13-A*k²*Plin
    # with A from the k→0 asymptotics) is validated.

    # Tree-level (normed at bq1=1 for DS counterterm shape)
    ds_normed = DSSplitBinEFT(label=ds_params.label, bq1=1.0)
    gal_normed = GalaxyEFTParams(b1=gal_params.b1)
    tree_normed = _pqg_tree_eft(k, mu, plin, wk, f, ds_normed, gal_normed, ds_model)

    P = _pqg_tree_eft(k, mu, plin, wk, f, ds_params, gal_params, ds_model)
    ds_amplitude = ds_params.bq1 * wk
    P = P + galaxy_counterterm(k, mu, plin, gal_params, ds_amplitude)
    P = P + ds_counterterm(k, mu, plin, ds_params, tree_normed, R)

    if mode == "eft_full":
        P = P + stochastic_term(k, mu, gal_params)

    return P
