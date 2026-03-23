"""Tests for tree-level density-split pair power spectrum models."""

import numpy as np
import pytest

from drift.theory.density_split.bias import DSSplitBin
from drift.theory.density_split.power_spectrum import pqq_mu
from drift.utils.cosmology import get_cosmology
from drift.utils.multipoles import compute_multipoles


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture
def k():
    return np.logspace(-2, -0.5, 24)


@pytest.fixture
def mu():
    return np.linspace(-1, 1, 17)


@pytest.fixture
def ds_a():
    return DSSplitBin(label="DS2", bq=-0.8, cq=0.1, beta_q=0.0)


@pytest.fixture
def ds_b():
    return DSSplitBin(label="DS5", bq=1.2, cq=0.2, beta_q=0.3)


def test_pqq_shape_and_finiteness(cosmo, k, mu, ds_a, ds_b):
    result = pqq_mu(k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_b, R=10.0)
    assert result.shape == (len(k), len(mu))
    assert np.all(np.isfinite(result))


def test_pqq_is_symmetric_under_leg_exchange(cosmo, k, mu, ds_a, ds_b):
    pab = pqq_mu(k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_b, R=10.0)
    pba = pqq_mu(k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_b, ds_params_b=ds_a, R=10.0)
    np.testing.assert_allclose(pab, pba, rtol=1e-12)


def test_pqq_auto_is_finite(cosmo, k, mu, ds_a):
    paa = pqq_mu(k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_a, R=10.0)
    assert np.all(np.isfinite(paa))


def test_pqq_pheno_beta0_matches_baseline(cosmo, k, mu):
    ds_a = DSSplitBin(label="DS1", bq=-1.0, cq=0.0, beta_q=0.0)
    ds_b = DSSplitBin(label="DS5", bq=1.3, cq=0.0, beta_q=0.0)
    baseline = pqq_mu(
        k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="baseline",
    )
    pheno = pqq_mu(
        k, mu, z=0.5, cosmo=cosmo, ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="phenomenological",
    )
    np.testing.assert_allclose(pheno, baseline, rtol=1e-12)


def test_pqq_baseline_only_has_monopole(cosmo, k):
    ds_a = DSSplitBin(label="DS2", bq=-0.8)
    ds_b = DSSplitBin(label="DS5", bq=1.2)

    def model(kk, mu):
        return pqq_mu(
            kk, mu, z=0.5, cosmo=cosmo,
            ds_params_a=ds_a, ds_params_b=ds_b,
            R=10.0, ds_model="baseline",
        )

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    np.testing.assert_allclose(poles[2], 0.0, atol=1e-8)
    np.testing.assert_allclose(poles[4], 0.0, atol=1e-8)
    assert np.any(np.abs(poles[0]) > 0.0)


def test_pqq_rsd_selection_has_quadrupole_and_hexadecapole(cosmo, k):
    ds_a = DSSplitBin(label="DS2", bq=-0.8)
    ds_b = DSSplitBin(label="DS5", bq=1.2)

    def model(kk, mu):
        return pqq_mu(
            kk, mu, z=0.5, cosmo=cosmo,
            ds_params_a=ds_a, ds_params_b=ds_b,
            R=10.0, ds_model="rsd_selection",
        )

    poles = compute_multipoles(k, model, ells=(2, 4))
    assert np.any(np.abs(poles[2]) > 1e-10)
    assert np.any(np.abs(poles[4]) > 1e-10)
