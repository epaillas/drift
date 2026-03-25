"""Tree-level density-split power spectrum models in (k, mu) space."""

import numpy as np

from ...utils.cosmology import get_growth_rate, get_linear_power
from ...utils.kernels import gaussian_kernel, tophat_kernel
from ...utils.rsd import matter_rsd_factor, tracer_rsd_factor
from .bias import DensitySplitBin

_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")
_VALID_SPACES = ("redshift", "real")


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


def _ds_bias_eff(
    k: np.ndarray,
    R: float,
    ds_params: DensitySplitBin,
) -> np.ndarray:
    """Return the isotropic DS bias factor bq + cq (kR)^2."""
    return ds_params.bq + ds_params.cq * (k * R) ** 2


def _ds_leg_factor(
    k: np.ndarray,
    mu: np.ndarray,
    f: float,
    R: float,
    ds_params: DensitySplitBin,
    ds_model: str,
) -> np.ndarray:
    """Return the DS angular factor for one leg."""
    bq_eff = _ds_bias_eff(k, R, ds_params)[:, np.newaxis]
    if ds_model == "baseline":
        return bq_eff * np.ones((1, len(mu)))
    if ds_model == "rsd_selection":
        return bq_eff * (1.0 + f * mu[np.newaxis, :] ** 2)
    return bq_eff + ds_params.beta_q * f * mu[np.newaxis, :] ** 2


def ds_matter_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params: DensitySplitBin,
    R: float,
    kernel: str = "gaussian",
    space: str = "redshift",
    ds_model: str = "baseline",
) -> np.ndarray:
    """Density-split-matter cross power spectrum P_{q_i, m}(k, mu, z).

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
    ds_params : DensitySplitBin
        Density-split bias parameters for bin q_i.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' (include RSD via growth rate f) or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant: 'baseline', 'rsd_selection', or 'phenomenological'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    _validate_ds_model(ds_model)
    _validate_space(space)

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)
    wk = _get_kernel(kernel, k, R)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    ds_leg = _ds_leg_factor(k, mu, f, R, ds_params, ds_model)
    matter_leg = matter_rsd_factor(mu, f)[np.newaxis, :]
    return (plin * wk)[:, np.newaxis] * ds_leg * matter_leg


def ds_galaxy_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params: DensitySplitBin,
    tracer_bias: float,
    R: float,
    kernel: str = "gaussian",
    space: str = "redshift",
    ds_model: str = "baseline",
) -> np.ndarray:
    """Density-split-galaxy cross power spectrum P_{q_i, g}(k, mu, z).

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
    ds_params : DensitySplitBin
        Density-split bias parameters for bin q_i.
    tracer_bias : float
        Linear galaxy bias b_1.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' (include RSD) or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant: 'baseline', 'rsd_selection', or 'phenomenological'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    _validate_ds_model(ds_model)
    _validate_space(space)

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)
    wk = _get_kernel(kernel, k, R)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    ds_leg = _ds_leg_factor(k, mu, f, R, ds_params, ds_model)
    galaxy_leg = tracer_rsd_factor(mu, tracer_bias, f)[np.newaxis, :]
    return (plin * wk)[:, np.newaxis] * ds_leg * galaxy_leg


def dspair_pkmu(
    k: np.ndarray,
    mu: np.ndarray,
    z: float,
    cosmo,
    ds_params_a: DensitySplitBin,
    ds_params_b: DensitySplitBin,
    R: float,
    kernel: str = "gaussian",
    space: str = "redshift",
    ds_model: str = "baseline",
) -> np.ndarray:
    """Density-split pair power spectrum P_{q_i q_j}(k, mu, z).

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
    ds_params_a : DensitySplitBin
        Density-split bias parameters for bin q_i.
    ds_params_b : DensitySplitBin
        Density-split bias parameters for bin q_j.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' (include RSD) or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant: 'baseline', 'rsd_selection', or 'phenomenological'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    _validate_ds_model(ds_model)
    _validate_space(space)

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)
    wk = _get_kernel(kernel, k, R)
    f = get_growth_rate(cosmo, z) if space == "redshift" else 0.0

    ds_leg_a = _ds_leg_factor(k, mu, f, R, ds_params_a, ds_model)
    ds_leg_b = _ds_leg_factor(k, mu, f, R, ds_params_b, ds_model)
    return (plin * wk ** 2)[:, np.newaxis] * ds_leg_a * ds_leg_b


pqm_mu = ds_matter_pkmu
pqg_mu = ds_galaxy_pkmu
pqq_mu = dspair_pkmu
DSSplitBin = DensitySplitBin
