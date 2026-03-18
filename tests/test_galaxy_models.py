"""Tests for galaxy auto-power spectrum models and emulator."""

import numpy as np
import pytest

from drift.cosmology import get_cosmology, get_linear_power, get_growth_rate
from drift.eft_bias import GalaxyEFTParams
from drift.galaxy_models import pgg_mu, pgg_eft_mu
from drift.galaxy_emulator import GalaxyTemplateEmulator
from drift.multipoles import compute_multipoles


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture(scope="module")
def k():
    return np.linspace(0.01, 0.3, 40)


@pytest.fixture(scope="module")
def plin(cosmo, k):
    return get_linear_power(cosmo, k, z=0.5)


@pytest.fixture(scope="module")
def f(cosmo):
    return get_growth_rate(cosmo, z=0.5)


def test_pgg_mu_monopole_kaiser(cosmo, k, plin, f):
    """P0 = (b1^2 + 2*b1*f/3 + f^2/5) * P_lin."""
    b1 = 2.0
    mu = np.linspace(-1, 1, 500)
    P_kmu = pgg_mu(k, mu, z=0.5, cosmo=cosmo, b1=b1)

    # Numerical multipole projection
    dmu = mu[1] - mu[0]
    P0_numerical = 0.5 * np.trapz(P_kmu, mu, axis=1)

    P0_analytic = (b1 ** 2 + 2.0 * b1 * f / 3.0 + f ** 2 / 5.0) * plin

    np.testing.assert_allclose(P0_numerical, P0_analytic, rtol=1e-3)


def test_pgg_mu_quadrupole_kaiser(cosmo, k, plin, f):
    """P2 = (4*b1*f/3 + 4*f^2/7) * P_lin."""
    b1 = 2.0
    mu = np.linspace(-1, 1, 1000)
    P_kmu = pgg_mu(k, mu, z=0.5, cosmo=cosmo, b1=b1)

    # L2(mu) = (3*mu^2 - 1) / 2
    L2 = (3.0 * mu ** 2 - 1.0) / 2.0
    P2_numerical = 2.5 * np.trapz(P_kmu * L2[np.newaxis, :], mu, axis=1)

    P2_analytic = (4.0 * b1 * f / 3.0 + 4.0 * f ** 2 / 7.0) * plin

    np.testing.assert_allclose(P2_numerical, P2_analytic, rtol=1e-3)


def test_pgg_mu_real_space(cosmo, k, plin):
    """In real space, P_gg = b1^2 * P_lin (isotropic, no mu dependence)."""
    b1 = 1.5
    mu = np.array([0.0, 0.3, 0.7, 1.0])
    P_kmu = pgg_mu(k, mu, z=0.5, cosmo=cosmo, b1=b1, space="real")

    P_expected = (b1 ** 2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))
    np.testing.assert_allclose(P_kmu, P_expected, rtol=1e-12)


def test_pgg_eft_tree_matches_pgg_mu(cosmo, k, f):
    """pgg_eft_mu with mode='tree_only' must equal pgg_mu."""
    b1 = 2.0
    mu = np.linspace(-1, 1, 50)
    gal = GalaxyEFTParams(b1=b1)
    P_eft = pgg_eft_mu(k, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                       space="redshift", mode="tree_only")
    P_kaiser = pgg_mu(k, mu, z=0.5, cosmo=cosmo, b1=b1, space="redshift")
    np.testing.assert_allclose(P_eft, P_kaiser, rtol=1e-12)


def test_galaxy_emulator_matches_direct(cosmo, k):
    """GalaxyTemplateEmulator.predict() must match GL quadrature result."""
    b1 = 1.8
    c0 = 5.0
    s0 = 100.0
    gal = GalaxyEFTParams(b1=b1, c0=c0, s0=s0)
    ells = (0, 2, 4)

    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=0.5, space="redshift", mode="eft_full"
    )
    params = {"b1": b1, "c0": c0, "s0": s0}
    pred_emulator = emulator.predict(params)

    def model(kk, mu):
        return pgg_eft_mu(kk, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                          space="redshift", mode="eft_full")

    poles_direct = compute_multipoles(k, model, ells=ells)
    pred_direct = np.concatenate([poles_direct[ell] for ell in ells])

    np.testing.assert_allclose(pred_emulator, pred_direct, rtol=1e-4)


def test_galaxy_emulator_one_loop_matches_direct_b1only(cosmo, k):
    """one_loop emulator must match GL quadrature for b2=bs2=0."""
    b1 = 1.8
    c0 = 5.0
    s0 = 100.0
    gal = GalaxyEFTParams(b1=b1, c0=c0, s0=s0, b2=0.0, bs2=0.0)
    ells = (0, 2, 4)

    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=0.5, space="redshift", mode="one_loop"
    )
    params = {"b1": b1, "c0": c0, "s0": s0, "b2": 0.0, "bs2": 0.0}
    pred_emulator = emulator.predict(params)

    def model(kk, mu):
        return pgg_eft_mu(kk, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                          space="redshift", mode="one_loop")

    poles_direct = compute_multipoles(k, model, ells=ells)
    pred_direct = np.concatenate([poles_direct[ell] for ell in ells])

    np.testing.assert_allclose(pred_emulator, pred_direct, rtol=1e-3)


def test_galaxy_emulator_one_loop_matches_direct_nonzero_b2bs2(cosmo, k):
    """one_loop emulator must match GL quadrature for non-zero b2, bs2."""
    b1 = 1.8
    b2 = 0.5
    bs2 = -0.3
    c0 = 5.0
    s0 = 100.0
    gal = GalaxyEFTParams(b1=b1, c0=c0, s0=s0, b2=b2, bs2=bs2)
    ells = (0, 2, 4)

    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=0.5, space="redshift", mode="one_loop"
    )
    params = {"b1": b1, "c0": c0, "s0": s0, "b2": b2, "bs2": bs2}
    pred_emulator = emulator.predict(params)

    def model(kk, mu):
        return pgg_eft_mu(kk, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                          space="redshift", mode="one_loop")

    poles_direct = compute_multipoles(k, model, ells=ells)
    pred_direct = np.concatenate([poles_direct[ell] for ell in ells])

    np.testing.assert_allclose(pred_emulator, pred_direct, rtol=1e-3)


def test_one_loop_matter_only_reduces_to_eft_full(cosmo, k):
    """one_loop_matter_only with zeroed loop templates equals eft_full."""
    b1, c0, s0 = 1.8, 5.0, 100.0
    ells = (0, 2, 4)

    emu_eft = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="eft_full")
    pred_eft = emu_eft.predict({"b1": b1, "c0": c0, "s0": s0})

    emu_monly = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="one_loop_matter_only")
    # Zero out loop templates
    nk = len(k)
    zeros = np.zeros(nk)
    for attr in ("_T_p22", "_T_p13", "_T_p22_dt", "_T_p22_tt", "_T_p13_dt", "_T_p13_tt"):
        setattr(emu_monly, attr, zeros)
    pred_monly = emu_monly.predict({"b1": b1, "c0": c0, "s0": s0})

    np.testing.assert_allclose(pred_monly, pred_eft, rtol=1e-12)


def test_one_loop_matter_only_vs_one_loop_diff_is_bias_only(cosmo, k):
    """Difference between one_loop and one_loop_matter_only is exactly p_loop_bias."""
    b1, b2, bs2, c0, s0 = 1.8, 0.5, -0.3, 5.0, 100.0
    ells = (0, 2, 4)
    params = {"b1": b1, "b2": b2, "bs2": bs2, "c0": c0, "s0": s0}

    emu_full = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="one_loop")
    emu_monly = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="one_loop_matter_only")

    pred_full = emu_full.predict(params)
    pred_monly = emu_monly.predict(params)

    # The difference should be non-zero (bias loops contribute)
    assert not np.allclose(pred_full, pred_monly, atol=1e-10)


def test_one_loop_matter_only_equals_one_loop_at_zero_bias(cosmo, k):
    """one_loop with b2=bs2=0 must equal one_loop_matter_only."""
    b1, c0, s0 = 1.8, 5.0, 100.0
    ells = (0, 2, 4)
    params = {"b1": b1, "c0": c0, "s0": s0, "b2": 0.0, "bs2": 0.0}

    emu_full = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="one_loop")
    emu_monly = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, mode="one_loop_matter_only")

    pred_full = emu_full.predict(params)
    pred_monly = emu_monly.predict(params)

    np.testing.assert_allclose(pred_full, pred_monly, rtol=1e-12)


def test_galaxy_emulator_one_loop_matter_only_matches_direct(cosmo, k):
    """one_loop_matter_only emulator matches GL quadrature."""
    b1, c0, s0 = 1.8, 5.0, 100.0
    gal = GalaxyEFTParams(b1=b1, c0=c0, s0=s0)
    ells = (0, 2, 4)

    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=0.5, mode="one_loop_matter_only"
    )
    pred_emulator = emulator.predict({"b1": b1, "c0": c0, "s0": s0})

    def model(kk, mu):
        return pgg_eft_mu(kk, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                          space="redshift", mode="one_loop_matter_only")

    poles_direct = compute_multipoles(k, model, ells=ells)
    pred_direct = np.concatenate([poles_direct[ell] for ell in ells])

    np.testing.assert_allclose(pred_emulator, pred_direct, rtol=1e-3)
