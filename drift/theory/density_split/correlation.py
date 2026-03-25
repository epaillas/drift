"""Density-split configuration-space multipole helpers."""

from __future__ import annotations

from ...utils.multipoles import compute_correlation_multipoles
from .eft_power_spectrum import ds_galaxy_eft_pkmu, dspair_eft_pkmu
from .power_spectrum import ds_galaxy_pkmu, dspair_pkmu


def compute_ds_galaxy_correlation_multipoles(
    k,
    z,
    cosmo,
    ds_params,
    R,
    tracer_bias=None,
    gal_params=None,
    kernel="gaussian",
    space="redshift",
    ds_model="baseline",
    mode=None,
    ells=(0, 2, 4),
    mu_grid=None,
    q=1.0,
    extrap="log",
    fftlog_kwargs=None,
):
    """Return DS×galaxy correlation-function multipoles xi_ell(s).

    Delegates to ``ds_galaxy_pkmu`` (tree-level, when ``gal_params`` is None)
    or ``ds_galaxy_eft_pkmu`` (EFT, when ``gal_params`` is provided), then
    applies an FFTLog power-to-correlation transform.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Log-spaced wavenumbers in h/Mpc.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
        Cosmology used for P_lin and f.
    ds_params : DensitySplitBin or DensitySplitEFTParameters
        Density-split bias parameters.
    R : float
        Smoothing radius in Mpc/h.
    tracer_bias : float, optional
        Linear galaxy bias (required when ``gal_params`` is None).
    gal_params : GalaxyEFTParameters, optional
        EFT galaxy bias parameters. If provided, EFT model is used.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant.
    mode : str, optional
        EFT mode passed to ``ds_galaxy_eft_pkmu``. Defaults to 'eft_ct'.
    ells : tuple of int, default (0, 2, 4)
        Multipole orders to compute.
    mu_grid : tuple or None
        Gauss-Legendre quadrature grid (nodes, weights).
    q : float, default 1.0
        FFTLog bias parameter.
    extrap : str, default 'log'
        FFTLog extrapolation mode.
    fftlog_kwargs : dict or None
        Extra keyword arguments forwarded to the FFTLog transform.

    Returns
    -------
    s : np.ndarray, shape (ns,)
        Separation array in Mpc/h.
    xi : dict
        {ell: np.ndarray of shape (ns,)} for each ell in ells.
    """
    if gal_params is None:
        if tracer_bias is None:
            raise ValueError("tracer_bias must be provided for non-EFT DS×galaxy predictions.")

        def model(kk, mu):
            return ds_galaxy_pkmu(
                kk,
                mu,
                z=z,
                cosmo=cosmo,
                ds_params=ds_params,
                tracer_bias=tracer_bias,
                R=R,
                kernel=kernel,
                space=space,
                ds_model=ds_model,
            )
    else:
        eft_mode = "eft_ct" if mode is None else mode

        def model(kk, mu):
            return ds_galaxy_eft_pkmu(
                kk,
                mu,
                z=z,
                cosmo=cosmo,
                ds_params=ds_params,
                gal_params=gal_params,
                R=R,
                kernel=kernel,
                space=space,
                ds_model=ds_model,
                mode=eft_mode,
            )

    return compute_correlation_multipoles(
        k,
        model,
        ells=ells,
        mu_grid=mu_grid,
        q=q,
        extrap=extrap,
        fftlog_kwargs=fftlog_kwargs,
    )


def compute_dspair_correlation_multipoles(
    k,
    z,
    cosmo,
    ds_params_a,
    ds_params_b,
    R,
    kernel="gaussian",
    space="redshift",
    ds_model="baseline",
    mode=None,
    sqq0=0.0,
    sqq2=0.0,
    ells=(0, 2, 4),
    mu_grid=None,
    q=1.0,
    extrap="log",
    fftlog_kwargs=None,
):
    """Return DS-pair correlation-function multipoles xi_ell(s).

    Delegates to ``dspair_pkmu`` (tree-level, when ``mode`` is None) or
    ``dspair_eft_pkmu`` (EFT, when ``mode`` is provided), then applies
    an FFTLog power-to-correlation transform.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Log-spaced wavenumbers in h/Mpc.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
        Cosmology used for P_lin and f.
    ds_params_a : DensitySplitBin or DensitySplitEFTParameters
        Density-split bias parameters for bin q_i.
    ds_params_b : DensitySplitBin or DensitySplitEFTParameters
        Density-split bias parameters for bin q_j.
    R : float
        Smoothing radius in Mpc/h.
    kernel : str, default 'gaussian'
        Smoothing kernel: 'gaussian' or 'tophat'.
    space : str, default 'redshift'
        'redshift' or 'real'.
    ds_model : str, default 'baseline'
        Angular model variant.
    mode : str, optional
        EFT mode passed to ``dspair_eft_pkmu``. If None, tree-level is used.
    sqq0 : float, default 0.0
        Isotropic stochastic amplitude for EFT modes.
    sqq2 : float, default 0.0
        k^2-dependent stochastic amplitude for EFT modes.
    ells : tuple of int, default (0, 2, 4)
        Multipole orders to compute.
    mu_grid : tuple or None
        Gauss-Legendre quadrature grid (nodes, weights).
    q : float, default 1.0
        FFTLog bias parameter.
    extrap : str, default 'log'
        FFTLog extrapolation mode.
    fftlog_kwargs : dict or None
        Extra keyword arguments forwarded to the FFTLog transform.

    Returns
    -------
    s : np.ndarray, shape (ns,)
        Separation array in Mpc/h.
    xi : dict
        {ell: np.ndarray of shape (ns,)} for each ell in ells.
    """
    if mode is None:

        def model(kk, mu):
            return dspair_pkmu(
                kk,
                mu,
                z=z,
                cosmo=cosmo,
                ds_params_a=ds_params_a,
                ds_params_b=ds_params_b,
                R=R,
                kernel=kernel,
                space=space,
                ds_model=ds_model,
            )
    else:

        def model(kk, mu):
            return dspair_eft_pkmu(
                kk,
                mu,
                z=z,
                cosmo=cosmo,
                ds_params_a=ds_params_a,
                ds_params_b=ds_params_b,
                R=R,
                kernel=kernel,
                space=space,
                ds_model=ds_model,
                mode=mode,
                sqq0=sqq0,
                sqq2=sqq2,
            )

    return compute_correlation_multipoles(
        k,
        model,
        ells=ells,
        mu_grid=mu_grid,
        q=q,
        extrap=extrap,
        fftlog_kwargs=fftlog_kwargs,
    )
