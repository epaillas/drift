"""Tree-level density-split power spectrum models in (k, mu) space."""

import numpy as np
from ...utils.cosmology import get_growth_rate, get_linear_power
from ...utils.kernels import gaussian_kernel, tophat_kernel
from ...utils.rsd import matter_rsd_factor, tracer_rsd_factor
from .bias import DensitySplitBin

_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")


def _get_kernel(kernel: str, k: np.ndarray, R: float) -> np.ndarray:
    if kernel == "gaussian":
        return gaussian_kernel(k, R)
    elif kernel == "tophat":
        return tophat_kernel(k, R)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian' or 'tophat'.")


def ds_matter_cross_spectrum_mu(
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

    Three redshift-space selection models are supported (``space="redshift"``):

    ``"baseline"``
        The DS field is selected in real space; only the matter field acquires
        a Kaiser factor::

            P_{q_i,m}(k,mu) = [bq + cq*(kR)^2] * [1 + f*mu^2] * P_lin * W_R

        Multipoles: P0 = A(k)*(1 + f/3), P2 = A(k)*(2f/3), P4 = 0.

    ``"rsd_selection"``
        Both the DS and matter fields are selected in redshift space; the DS
        field acquires a multiplicative Kaiser factor::

            P_{q_i,m}(k,mu) = [bq + cq*(kR)^2] * (1 + f*mu^2)^2 * P_lin * W_R

        With A(k) = [bq + cq*(kR)^2] * W_R(k) * P_lin(k):
        P0 = A(k)*(1 + 2f/3 + f²/5), P2 = A(k)*(4f/3 + 4f²/7),
        P4 = A(k)*(8f²/35).

    ``"phenomenological"``
        The DS factor is written as a free function of mu with an explicit
        anisotropy parameter ``beta_q``::

            DS_factor(k,mu) = [bq + cq*(kR)^2] + beta_q * f * mu^2
            P_{q_i,m}(k,mu) = DS_factor(k,mu) * [1 + f*mu^2] * P_lin * W_R

        When ``beta_q = 0`` this reduces to the baseline; when ``beta_q = 1``
        and ``cq = 0`` the DS factor becomes ``bq*(1 + f/bq * mu^2)``.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
    ds_params : DSSplitBin
        Bias parameters for the density-split bin.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str
        Smoothing kernel: 'gaussian' (default) or 'tophat'.
    space : str
        'redshift' (default) or 'real'.
    ds_model : str
        Redshift-space model for the DS field: 'baseline' (default),
        'rsd_selection', or 'phenomenological'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    if ds_model not in _VALID_DS_MODELS:
        raise ValueError(
            f"Unknown ds_model '{ds_model}'. Choose one of {_VALID_DS_MODELS}."
        )

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)   # (nk,)
    wk = _get_kernel(kernel, k, R)          # (nk,)
    bq_eff = ds_params.bq + ds_params.cq * (k * R) ** 2  # (nk,)

    if space == "redshift":
        f = get_growth_rate(cosmo, z)
        mat_factor = matter_rsd_factor(mu, f)   # (nmu,)

        if ds_model == "baseline":
            amplitude = bq_eff * plin * wk   # (nk,)
            return amplitude[:, np.newaxis] * mat_factor[np.newaxis, :]

        elif ds_model == "rsd_selection":
            # bq_eff * (1 + f*mu^2) * (1 + f*mu^2)
            ds_factor = bq_eff[:, np.newaxis] * (1.0 + f * mu[np.newaxis, :] ** 2)  # (nk, nmu)
            return (plin * wk)[:, np.newaxis] * ds_factor * mat_factor[np.newaxis, :]

        else:  # phenomenological
            beta_q = ds_params.beta_q
            ds_factor = bq_eff[:, np.newaxis] + beta_q * f * mu[np.newaxis, :] ** 2  # (nk, nmu)
            return (plin * wk)[:, np.newaxis] * ds_factor * mat_factor[np.newaxis, :]

    elif space == "real":
        amplitude = bq_eff * plin * wk   # (nk,)
        return amplitude[:, np.newaxis] * np.ones_like(mu)[np.newaxis, :]
    else:
        raise ValueError(f"Unknown space '{space}'. Choose 'redshift' or 'real'.")


def ds_galaxy_cross_spectrum_mu(
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

    Three redshift-space selection models are supported (``space="redshift"``):

    ``"baseline"``
        The DS field is selected in real space; only the galaxy field acquires
        a Kaiser factor::

            P_{q_i,g}(k,mu) = [bq + cq*(kR)^2] * [b1 + f*mu^2] * P_lin * W_R

        Multipoles: P0 = A(k)*(b1 + f/3), P2 = A(k)*(2f/3), P4 = 0.

    ``"rsd_selection"``
        Both the DS and galaxy fields are selected in redshift space; the DS
        field acquires a multiplicative Kaiser factor::

            P_{q_i,g}(k,mu) = [bq + cq*(kR)^2] * (1 + f*mu^2) * (b1 + f*mu^2) * P_lin * W_R

        With A(k) = [bq + cq*(kR)^2] * W_R(k) * P_lin(k):
        P0 = A(k)*(b1 + f(1+b1)/3 + f²/5),
        P2 = A(k)*(2f(1+b1)/3 + 4f²/7), P4 = A(k)*(8f²/35).

    ``"phenomenological"``
        The DS factor is written with an explicit anisotropy parameter
        ``beta_q``::

            DS_factor(k,mu) = [bq + cq*(kR)^2] + beta_q * f * mu^2
            P_{q_i,g}(k,mu) = DS_factor(k,mu) * [b1 + f*mu^2] * P_lin * W_R

        When ``beta_q = 0`` this reduces to the baseline.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
    ds_params : DSSplitBin
        Bias parameters for the density-split bin.
    tracer_bias : float
        Linear galaxy bias b1.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str
        Smoothing kernel: 'gaussian' (default) or 'tophat'.
    space : str
        'redshift' (default) or 'real'.
    ds_model : str
        Redshift-space model for the DS field: 'baseline' (default),
        'rsd_selection', or 'phenomenological'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    if ds_model not in _VALID_DS_MODELS:
        raise ValueError(
            f"Unknown ds_model '{ds_model}'. Choose one of {_VALID_DS_MODELS}."
        )

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    plin = get_linear_power(cosmo, k, z)   # (nk,)
    wk = _get_kernel(kernel, k, R)          # (nk,)
    bq_eff = ds_params.bq + ds_params.cq * (k * R) ** 2  # (nk,)

    if space == "redshift":
        f = get_growth_rate(cosmo, z)
        gal_rsd = tracer_rsd_factor(mu, tracer_bias, f)   # (nmu,)

        if ds_model == "baseline":
            amplitude = bq_eff * plin * wk   # (nk,)
            return amplitude[:, np.newaxis] * gal_rsd[np.newaxis, :]

        elif ds_model == "rsd_selection":
            # bq_eff * (1 + f*mu^2) * (b1 + f*mu^2)
            ds_factor = bq_eff[:, np.newaxis] * (1.0 + f * mu[np.newaxis, :] ** 2)  # (nk, nmu)
            return (plin * wk)[:, np.newaxis] * ds_factor * gal_rsd[np.newaxis, :]

        else:  # phenomenological
            beta_q = ds_params.beta_q
            ds_factor = bq_eff[:, np.newaxis] + beta_q * f * mu[np.newaxis, :] ** 2  # (nk, nmu)
            return (plin * wk)[:, np.newaxis] * ds_factor * gal_rsd[np.newaxis, :]

    elif space == "real":
        amplitude = bq_eff * plin * wk   # (nk,)
        return amplitude[:, np.newaxis] * (tracer_bias * np.ones_like(mu))[np.newaxis, :]
    else:
        raise ValueError(f"Unknown space '{space}'. Choose 'redshift' or 'real'.")


pqm_mu = ds_matter_cross_spectrum_mu
pqg_mu = ds_galaxy_cross_spectrum_mu
DSSplitBin = DensitySplitBin
