"""EFT density-split × galaxy power spectrum assembler."""

import numpy as np
from .cosmology import get_linear_power, get_growth_rate
from .kernels import gaussian_kernel, tophat_kernel
from .eft_bias import DSSplitBinEFT, GalaxyEFTParams
from .eft_terms import galaxy_counterterm, ds_counterterm, stochastic_term

_VALID_MODES = ("tree", "eft_ct", "eft", "one_loop")
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


def _pqg_ds_lin(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params: DSSplitBinEFT,
    ds_model: str,
) -> np.ndarray:
    """DS × linear matter cross-spectrum at tree level (no galaxy bias factor).

    Parameters
    ----------
    k : np.ndarray, shape (nk,)
    mu : np.ndarray, shape (nmu,)
    plin : np.ndarray, shape (nk,)
    wk : np.ndarray, shape (nk,)
    f : float
    ds_params : DSSplitBinEFT
    ds_model : str

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    bq1 = ds_params.bq1
    if ds_model == "baseline":
        return (bq1 * plin * wk)[:, np.newaxis] * np.ones((1, len(mu)))
    elif ds_model == "rsd_selection":
        ds_angular = 1.0 + f * mu**2                              # (nmu,)
        return (bq1 * plin * wk)[:, np.newaxis] * ds_angular[np.newaxis, :]
    else:  # phenomenological
        beta_q = ds_params.beta_q
        ds_angular = bq1 + beta_q * f * mu**2                    # (nmu,)
        return (plin * wk)[:, np.newaxis] * ds_angular[np.newaxis, :]


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
    mode: str = "eft_ct",
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
        - 'tree': tree-level only (matches pqg_mu exactly at bq=bq1)
        - 'eft_ct': tree + galaxy counterterm + DS counterterm (no loop promotion;
          raw SPT P13 is UV-sensitive and cannot be renormalized by the k² counterterm
          basis at the current integration range — see _pqg_one_loop_partial)
        - 'eft': eft_ct + stochastic terms (s0 + s2*k^2)

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

    if mode == "tree":
        return _pqg_tree_eft(k, mu, plin, wk, f, ds_params, gal_params, ds_model)

    if mode in ("one_loop",):
        from .galaxy_models import _compute_loop_templates
        def _plin_func(kk):
            return get_linear_power(cosmo, np.asarray(kk, dtype=float), z)
        loops = _compute_loop_templates(k, _plin_func)

        b1 = gal_params.b1
        bq1 = ds_params.bq1
        beta_q = ds_params.beta_q

        P_dd_loop = (loops["p22"] + loops["p13"]) * wk
        P_dt_loop = (loops["p22_dt"] + loops["p13_dt"]) * wk

        # Galaxy-matter cross-spectrum loop coefficients (mu^0 and mu^2)
        Pgm_l0 = b1 * P_dd_loop
        Pgm_l2 = f * P_dt_loop

        if mode == "one_loop":
            b2   = gal_params.b2
            bs2  = gal_params.bs2
            b3nl = gal_params.b3nl
            Pgm_l0 = (Pgm_l0
                      + (b2 * loops["I12"] + bs2 * loops["J12"] + 2.0 * b3nl * loops["Ib3nl"]) * wk)
            Pgm_l2 = Pgm_l2 + f * (b2 * loops["I12_v"] + bs2 * loops["J12_v"]) * wk

        # Combined coefficients: A0 (mu^0) + A2 * mu^2
        A0 = b1 * plin * wk + Pgm_l0   # (nk,)
        A2 = f  * plin * wk + Pgm_l2   # (nk,)

        # Multiply by DS_factor(mu)
        if ds_model == "baseline":
            # DS_factor = bq1
            P = bq1 * (A0[:, np.newaxis] + A2[:, np.newaxis] * mu[np.newaxis, :]**2)
        elif ds_model == "rsd_selection":
            # DS_factor = bq1 * (1 + f*mu^2)
            P = bq1 * (
                A0[:, np.newaxis]
                + (A2[:, np.newaxis] + f * A0[:, np.newaxis]) * mu[np.newaxis, :]**2
                + f * A2[:, np.newaxis] * mu[np.newaxis, :]**4
            )
        else:  # phenomenological
            # DS_factor = bq1 + beta_q*f*mu^2
            P = (
                bq1 * A0[:, np.newaxis]
                + (bq1 * A2[:, np.newaxis] + beta_q * f * A0[:, np.newaxis]) * mu[np.newaxis, :]**2
                + beta_q * f * A2[:, np.newaxis] * mu[np.newaxis, :]**4
            )

        # Galaxy EFT counterterm: -k^2*(c0+c2*mu^2+c4*mu^4) * P_{DS×lin}
        ds_lin = _pqg_ds_lin(k, mu, plin, wk, f, ds_params, ds_model)
        P = P + galaxy_counterterm(k, mu, gal_params, ds_lin)

        # DS higher-derivative counterterm
        ds_normed = DSSplitBinEFT(label=ds_params.label, bq1=1.0)
        gal_normed = GalaxyEFTParams(b1=gal_params.b1)
        tree_normed = _pqg_tree_eft(k, mu, plin, wk, f, ds_normed, gal_normed, ds_model)
        P = P + ds_counterterm(k, mu, plin, ds_params, tree_normed, R)

        # FoG for cross-spectrum: -sigma_fog * k^2*mu^2 * DS_factor * (b1 + f*mu^2) * Plin*W_R
        sigma_fog = gal_params.sigma_fog
        if sigma_fog != 0.0:
            if ds_model == "baseline":
                ds_factor_2d = bq1 * np.ones((len(k), len(mu)))
            elif ds_model == "rsd_selection":
                ds_factor_2d = bq1 * (1.0 + f * mu[np.newaxis, :]**2)
            else:
                ds_factor_2d = bq1 + beta_q * f * mu[np.newaxis, :]**2
            P = P - sigma_fog * (k**2 * plin * wk)[:, np.newaxis] * ds_factor_2d * (
                b1 * mu[np.newaxis, :]**2 + f * mu[np.newaxis, :]**4
            )

        # Stochastic: s0 + s2*k^2*mu^2
        s0 = gal_params.s0
        s2 = gal_params.s2
        if s0 != 0.0 or s2 != 0.0:
            stoch = gal_params.s0 + gal_params.s2 * k[:, np.newaxis]**2 * mu[np.newaxis, :]**2
            P = P + stoch

        return P

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
    ds_lin = _pqg_ds_lin(k, mu, plin, wk, f, ds_params, ds_model)
    P = P + galaxy_counterterm(k, mu, gal_params, ds_lin)
    P = P + ds_counterterm(k, mu, plin, ds_params, tree_normed, R)

    if mode == "eft":
        P = P + stochastic_term(k, mu, gal_params)

    return P
