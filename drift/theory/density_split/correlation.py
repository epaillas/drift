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
    """Return DS×galaxy correlation-function multipoles xi_ell(s)."""
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
    """Return DS-pair correlation-function multipoles xi_ell(s)."""
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
