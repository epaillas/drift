"""EFT density-split power spectrum assemblers."""

import numpy as np

from ...utils.cosmology import get_growth_rate, get_linear_power
from ...utils.kernels import gaussian_kernel, tophat_kernel
from ..galaxy.bias import GalaxyEFTParameters
from ..galaxy.power_spectrum import _compute_loop_templates
from .bias import DensitySplitEFTParameters
from .counterterms import (
    density_split_counterterm,
    density_split_pair_stochastic_term,
    galaxy_counterterm,
    stochastic_term,
)

_VALID_MODES = ("tree", "eft_ct", "eft", "one_loop")
_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")
_VALID_SPACES = ("redshift", "real")


def _validate_mode(mode: str) -> None:
    """Raise ValueError if mode is not in _VALID_MODES."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {_VALID_MODES}.")


def _validate_ds_model(ds_model: str) -> None:
    """Raise ValueError if ds_model is not in _VALID_DS_MODELS."""
    if ds_model not in _VALID_DS_MODELS:
        raise ValueError(
            f"Unknown ds_model '{ds_model}'. Choose one of {_VALID_DS_MODELS}."
        )


def _validate_space(space: str) -> None:
    """Raise ValueError if space is not in _VALID_SPACES."""
    if space not in _VALID_SPACES:
        raise ValueError(
            f"Unknown space '{space}'. Choose one of {_VALID_SPACES}."
        )


def _get_kernel(kernel: str, k: np.ndarray, R: float) -> np.ndarray:
    """Return the smoothing kernel W_R(k) for the given kernel name."""
    if kernel == "gaussian":
        return gaussian_kernel(k, R)
    if kernel == "tophat":
        return tophat_kernel(k, R)
    raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian' or 'tophat'.")


def _ds_eft_leg_factor(
    mu: np.ndarray,
    f: float,
    ds_params: DensitySplitEFTParameters,
    ds_model: str,
) -> np.ndarray:
    """Return the DS angular factor for one EFT leg."""
    if ds_model == "baseline":
        return ds_params.bq1 * np.ones((1, len(mu)))
    if ds_model == "rsd_selection":
        return ds_params.bq1 * (1.0 + f * mu[np.newaxis, :] ** 2)
    return ds_params.bq1 + ds_params.beta_q * f * mu[np.newaxis, :] ** 2


def _normalized_ds_params(
    ds_params: DensitySplitEFTParameters,
) -> DensitySplitEFTParameters:
    """Return a copy of ds_params with bq1 set to 1 for counterterm normalization."""
    return DensitySplitEFTParameters(
        label=ds_params.label,
        bq1=1.0,
        beta_q=ds_params.beta_q,
    )


def _reject_unimplemented_ds_loops(
    *ds_params_list: DensitySplitEFTParameters,
) -> None:
    """Raise NotImplementedError if any DS bin has nonzero bq2 or bqK2."""
    for ds_params in ds_params_list:
        if ds_params.bq2 != 0.0 or ds_params.bqK2 != 0.0:
            raise NotImplementedError(
                "bq2 and bqK2 one-loop DS terms are not yet implemented. "
                "Set bq2=bqK2=0 for mode='one_loop'."
            )


def _ds_galaxy_tree_eft_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params: DensitySplitEFTParameters,
    gal_params: GalaxyEFTParameters,
    ds_model: str,
) -> np.ndarray:
    """Tree-level DS x galaxy cross spectrum using EFT containers."""
    ds_leg = _ds_eft_leg_factor(mu, f, ds_params, ds_model)
    gal_leg = (gal_params.b1 + f * mu**2)[np.newaxis, :]
    return (plin * wk)[:, np.newaxis] * ds_leg * gal_leg


def ds_linear_matter_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params: DensitySplitEFTParameters,
    ds_model: str,
) -> np.ndarray:
    """DS × linear matter cross-spectrum at tree level (no galaxy bias factor)."""
    ds_leg = _ds_eft_leg_factor(mu, f, ds_params, ds_model)
    return (plin * wk)[:, np.newaxis] * ds_leg


def _dspair_tree_eft_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params_a: DensitySplitEFTParameters,
    ds_params_b: DensitySplitEFTParameters,
    ds_model: str,
) -> np.ndarray:
    """Tree-level DS-pair spectrum using EFT containers."""
    ds_leg_a = _ds_eft_leg_factor(mu, f, ds_params_a, ds_model)
    ds_leg_b = _ds_eft_leg_factor(mu, f, ds_params_b, ds_model)
    return (plin * wk ** 2)[:, np.newaxis] * ds_leg_a * ds_leg_b


def _dspair_one_loop_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    wk: np.ndarray,
    f: float,
    ds_params_a: DensitySplitEFTParameters,
    ds_params_b: DensitySplitEFTParameters,
    ds_model: str,
) -> np.ndarray:
    """Promote the matter kernel to one-loop while keeping DS angular factors."""
    ds_leg_a = _ds_eft_leg_factor(mu, f, ds_params_a, ds_model)
    ds_leg_b = _ds_eft_leg_factor(mu, f, ds_params_b, ds_model)
    return (plin * wk ** 2)[:, np.newaxis] * ds_leg_a * ds_leg_b


def ds_galaxy_eft_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params: DensitySplitEFTParameters,
    gal_params: GalaxyEFTParameters,
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
        Cosmology object used to evaluate P_lin and f.
    ds_params : DensitySplitEFTParameters
        EFT bias parameters for the density-split bin q_i.
    gal_params : GalaxyEFTParameters
        EFT bias and nuisance parameters for the galaxy tracer.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' (include RSD) or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant: 'baseline', 'rsd_selection', or 'phenomenological'.
    mode : str, default 'eft_ct'
        Theory level: 'tree', 'eft_ct' (tree + counterterms), 'eft'
        (eft_ct + stochastic), or 'one_loop' (full one-loop SPT).

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    _validate_mode(mode)
    _validate_ds_model(ds_model)
    _validate_space(space)

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)
    wk = _get_kernel(kernel, k, R)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    if mode == "tree":
        return _ds_galaxy_tree_eft_pkmu(
            k, mu, plin, wk, f, ds_params, gal_params, ds_model
        )

    if mode == "one_loop":
        _reject_unimplemented_ds_loops(ds_params)

        def _plin_func(kk):
            return get_linear_power(cosmo, np.asarray(kk, dtype=float), z)

        loops = _compute_loop_templates(k, _plin_func)

        b1 = gal_params.b1
        beta_q = ds_params.beta_q
        bq1 = ds_params.bq1

        p_dd_loop = (loops["p22"] + loops["p13"]) * wk
        p_dt_loop = (loops["p22_dt"] + loops["p13_dt"]) * wk

        p_gm_l0 = b1 * p_dd_loop
        p_gm_l2 = f * p_dt_loop

        b2 = gal_params.b2
        bs2 = gal_params.bs2
        b3nl = gal_params.b3nl
        p_gm_l0 = p_gm_l0 + (
            b2 * loops["I12"] + bs2 * loops["J12"] + 2.0 * b3nl * loops["Ib3nl"]
        ) * wk
        p_gm_l2 = p_gm_l2 + f * (
            b2 * loops["I12_v"] + bs2 * loops["J12_v"]
        ) * wk

        a0 = b1 * plin * wk + p_gm_l0
        a2 = f * plin * wk + p_gm_l2

        if ds_model == "baseline":
            p = bq1 * (a0[:, np.newaxis] + a2[:, np.newaxis] * mu[np.newaxis, :] ** 2)
        elif ds_model == "rsd_selection":
            p = bq1 * (
                a0[:, np.newaxis]
                + (a2[:, np.newaxis] + f * a0[:, np.newaxis]) * mu[np.newaxis, :] ** 2
                + f * a2[:, np.newaxis] * mu[np.newaxis, :] ** 4
            )
        else:
            p = (
                bq1 * a0[:, np.newaxis]
                + (bq1 * a2[:, np.newaxis] + beta_q * f * a0[:, np.newaxis])
                * mu[np.newaxis, :] ** 2
                + beta_q * f * a2[:, np.newaxis] * mu[np.newaxis, :] ** 4
            )

        ds_lin = ds_linear_matter_pkmu(k, mu, plin, wk, f, ds_params, ds_model)
        p = p + galaxy_counterterm(k, mu, gal_params, ds_lin)

        tree_normed = _ds_galaxy_tree_eft_pkmu(
            k,
            mu,
            plin,
            wk,
            f,
            _normalized_ds_params(ds_params),
            GalaxyEFTParameters(b1=gal_params.b1),
            ds_model,
        )
        p = p + density_split_counterterm(k, mu, plin, ds_params, tree_normed, R)

        sigma_fog = gal_params.sigma_fog
        if sigma_fog != 0.0:
            ds_factor_2d = _ds_eft_leg_factor(mu, f, ds_params, ds_model)
            p = p - sigma_fog * (k**2 * plin * wk)[:, np.newaxis] * ds_factor_2d * (
                b1 * mu[np.newaxis, :] ** 2 + f * mu[np.newaxis, :] ** 4
            )

        if gal_params.s0 != 0.0 or gal_params.s2 != 0.0:
            p = p + gal_params.s0 + gal_params.s2 * (
                k[:, np.newaxis] * mu[np.newaxis, :]
            ) ** 2

        return p

    tree_normed = _ds_galaxy_tree_eft_pkmu(
        k,
        mu,
        plin,
        wk,
        f,
        _normalized_ds_params(ds_params),
        GalaxyEFTParameters(b1=gal_params.b1),
        ds_model,
    )
    p = _ds_galaxy_tree_eft_pkmu(k, mu, plin, wk, f, ds_params, gal_params, ds_model)
    ds_lin = ds_linear_matter_pkmu(k, mu, plin, wk, f, ds_params, ds_model)
    p = p + galaxy_counterterm(k, mu, gal_params, ds_lin)
    p = p + density_split_counterterm(k, mu, plin, ds_params, tree_normed, R)

    if mode == "eft":
        p = p + stochastic_term(k, mu, gal_params)

    return p


def dspair_eft_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params_a: DensitySplitEFTParameters,
    ds_params_b: DensitySplitEFTParameters,
    R: float,
    kernel: str = "gaussian",
    space: str = "redshift",
    ds_model: str = "baseline",
    mode: str = "eft_ct",
    sqq0: float = 0.0,
    sqq2: float = 0.0,
) -> np.ndarray:
    """EFT density-split pair power spectrum P_{q_i q_j}(k, mu).

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
        Cosmology object used to evaluate P_lin and f.
    ds_params_a : DensitySplitEFTParameters
        EFT bias parameters for bin q_i.
    ds_params_b : DensitySplitEFTParameters
        EFT bias parameters for bin q_j.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' (include RSD) or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant: 'baseline', 'rsd_selection', or 'phenomenological'.
    mode : str, default 'eft_ct'
        Theory level: 'tree', 'eft_ct', 'eft', or 'one_loop'.
    sqq0 : float, default 0.0
        Isotropic stochastic amplitude (constant in k).
    sqq2 : float, default 0.0
        k^2-dependent stochastic amplitude.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    _validate_mode(mode)
    _validate_ds_model(ds_model)
    _validate_space(space)

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)
    wk = _get_kernel(kernel, k, R)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    if mode == "tree":
        return _dspair_tree_eft_pkmu(
            k, mu, plin, wk, f, ds_params_a, ds_params_b, ds_model
        )

    p_base = plin
    if mode == "one_loop":
        _reject_unimplemented_ds_loops(ds_params_a, ds_params_b)

        def _plin_func(kk):
            return get_linear_power(cosmo, np.asarray(kk, dtype=float), z)

        loops = _compute_loop_templates(k, _plin_func)
        p_base = plin + loops["p22"] + loops["p13"]

    p = _dspair_tree_eft_pkmu(
        k, mu, p_base, wk, f, ds_params_a, ds_params_b, ds_model
    )

    norm_a = _dspair_tree_eft_pkmu(
        k,
        mu,
        plin,
        wk,
        f,
        _normalized_ds_params(ds_params_a),
        ds_params_b,
        ds_model,
    )
    norm_b = _dspair_tree_eft_pkmu(
        k,
        mu,
        plin,
        wk,
        f,
        ds_params_a,
        _normalized_ds_params(ds_params_b),
        ds_model,
    )
    p = p + density_split_counterterm(k, mu, plin, ds_params_a, norm_a, R)
    p = p + density_split_counterterm(k, mu, plin, ds_params_b, norm_b, R)

    if mode in ("eft", "one_loop"):
        p = p + density_split_pair_stochastic_term(k, mu, sqq0=sqq0, sqq2=sqq2)

    return p


_pqg_tree_eft = _ds_galaxy_tree_eft_pkmu
_pqg_ds_lin = ds_linear_matter_pkmu
pqg_eft_mu = ds_galaxy_eft_pkmu
pqq_eft_mu = dspair_eft_pkmu
DSSplitBinEFT = DensitySplitEFTParameters
GalaxyEFTParams = GalaxyEFTParameters
