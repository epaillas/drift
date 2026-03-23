"""Tests for galaxy auto-power spectrum models and emulator."""

import numpy as np
import pytest

from drift.utils.cosmology import get_cosmology, get_linear_power, get_growth_rate
from drift.theory.galaxy.bias import GalaxyEFTParams
from drift.theory.galaxy.power_spectrum import pgg_mu, pgg_eft_mu
from drift.emulators.galaxy import GalaxyTemplateEmulator
from drift.utils.multipoles import compute_multipoles


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
    """pgg_eft_mu with mode='tree' must equal pgg_mu."""
    b1 = 2.0
    mu = np.linspace(-1, 1, 50)
    gal = GalaxyEFTParams(b1=b1)
    P_eft = pgg_eft_mu(k, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                       space="redshift", mode="tree")
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
        cosmo, k, ells=ells, z=0.5, space="redshift", mode="eft"
    )
    params = {"b1": b1, "c0": c0, "s0": s0}
    pred_emulator = emulator.predict(params)

    def model(kk, mu):
        return pgg_eft_mu(kk, mu, z=0.5, cosmo=cosmo, gal_params=gal,
                          space="redshift", mode="eft")

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


