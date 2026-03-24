"""Tests for density-split configuration-space multipole helpers."""

import numpy as np

from drift import (
    compute_ds_galaxy_correlation_multipoles,
    compute_dspair_correlation_multipoles,
    compute_multipoles,
    get_cosmology,
    power_to_correlation_multipoles,
)
from drift.theory.density_split.bias import DSSplitBin, DSSplitBinEFT
from drift.theory.density_split.eft_power_spectrum import pqq_eft_mu, pqg_eft_mu
from drift.theory.density_split.power_spectrum import pqq_mu, pqg_mu
from drift.theory.galaxy.bias import GalaxyEFTParams


def test_compute_ds_galaxy_correlation_tree_matches_manual():
    cosmo = get_cosmology()
    k = np.logspace(-3, -0.3, 64)
    ds_bin = DSSplitBin(label="DS3", bq=0.5, cq=0.1)

    def model(kk, mu):
        return pqg_mu(kk, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, tracer_bias=1.8, R=10.0)

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    s_ref, xi_ref = power_to_correlation_multipoles(k, poles, ells=(0, 2, 4))
    s, xi = compute_ds_galaxy_correlation_multipoles(
        k,
        z=0.5,
        cosmo=cosmo,
        ds_params=ds_bin,
        tracer_bias=1.8,
        R=10.0,
        ells=(0, 2, 4),
    )

    np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=0.0)
    for ell in (0, 2, 4):
        np.testing.assert_allclose(xi[ell], xi_ref[ell], rtol=1e-12, atol=0.0)


def test_compute_ds_galaxy_correlation_eft_matches_manual():
    cosmo = get_cosmology()
    k = np.logspace(-3, -0.3, 64)
    ds_bin = DSSplitBinEFT(label="DS3", bq1=0.5, bq_nabla2=0.1)
    gal = GalaxyEFTParams(b1=1.8, c0=5.0, s0=100.0)

    def model(kk, mu):
        return pqg_eft_mu(
            kk,
            mu,
            z=0.5,
            cosmo=cosmo,
            ds_params=ds_bin,
            gal_params=gal,
            R=10.0,
            mode="eft",
        )

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    s_ref, xi_ref = power_to_correlation_multipoles(k, poles, ells=(0, 2, 4))
    s, xi = compute_ds_galaxy_correlation_multipoles(
        k,
        z=0.5,
        cosmo=cosmo,
        ds_params=ds_bin,
        gal_params=gal,
        R=10.0,
        mode="eft",
        ells=(0, 2, 4),
    )

    np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=0.0)
    for ell in (0, 2, 4):
        np.testing.assert_allclose(xi[ell], xi_ref[ell], rtol=1e-12, atol=0.0)


def test_compute_dspair_correlation_tree_matches_manual():
    cosmo = get_cosmology()
    k = np.logspace(-3, -0.3, 64)
    ds_a = DSSplitBin(label="DS2", bq=-0.4, cq=0.0)
    ds_b = DSSplitBin(label="DS4", bq=0.9, cq=0.0)

    def model(kk, mu):
        return pqq_mu(kk, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_b, R=10.0)

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    s_ref, xi_ref = power_to_correlation_multipoles(k, poles, ells=(0, 2, 4))
    s, xi = compute_dspair_correlation_multipoles(
        k,
        z=0.5,
        cosmo=cosmo,
        ds_params_a=ds_a,
        ds_params_b=ds_b,
        R=10.0,
        ells=(0, 2, 4),
    )

    np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=0.0)
    for ell in (0, 2, 4):
        np.testing.assert_allclose(xi[ell], xi_ref[ell], rtol=1e-12, atol=0.0)


def test_compute_dspair_correlation_eft_matches_manual():
    cosmo = get_cosmology()
    k = np.logspace(-3, -0.3, 64)
    ds_a = DSSplitBinEFT(label="DS2", bq1=-0.4)
    ds_b = DSSplitBinEFT(label="DS4", bq1=0.9)

    def model(kk, mu):
        return pqq_eft_mu(
            kk,
            mu,
            z=0.5,
            cosmo=cosmo,
            ds_params_a=ds_a,
            ds_params_b=ds_b,
            R=10.0,
            mode="eft",
            sqq0=10.0,
        )

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    s_ref, xi_ref = power_to_correlation_multipoles(k, poles, ells=(0, 2, 4))
    s, xi = compute_dspair_correlation_multipoles(
        k,
        z=0.5,
        cosmo=cosmo,
        ds_params_a=ds_a,
        ds_params_b=ds_b,
        R=10.0,
        mode="eft",
        sqq0=10.0,
        ells=(0, 2, 4),
    )

    np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=0.0)
    for ell in (0, 2, 4):
        np.testing.assert_allclose(xi[ell], xi_ref[ell], rtol=1e-12, atol=0.0)
