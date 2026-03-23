"""Tests for EFT density-split pair power spectrum models."""

import numpy as np
import pytest

from drift.theory.density_split.bias import DSSplitBin, DSSplitBinEFT
from drift.theory.density_split.counterterms import density_split_pair_stochastic_term
from drift.theory.density_split.eft_power_spectrum import pqq_eft_mu
from drift.theory.density_split.power_spectrum import pqq_mu
from drift.utils.cosmology import get_cosmology


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture
def k():
    return np.logspace(-2, -0.5, 20)


@pytest.fixture
def mu():
    return np.linspace(-1, 1, 15)


@pytest.fixture
def ds_a():
    return DSSplitBinEFT(label="DS2", bq1=-0.7, beta_q=0.2)


@pytest.fixture
def ds_b():
    return DSSplitBinEFT(label="DS5", bq1=1.1, beta_q=-0.1)


def test_pqq_tree_matches_non_eft_model(cosmo, k, mu):
    ds_tree_a = DSSplitBin(label="DS2", bq=-0.7, cq=0.0, beta_q=0.2)
    ds_tree_b = DSSplitBin(label="DS5", bq=1.1, cq=0.0, beta_q=-0.1)
    expected = pqq_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_tree_a, ds_params_b=ds_tree_b,
        R=10.0, ds_model="phenomenological",
    )
    got = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=DSSplitBinEFT(label="DS2", bq1=-0.7, beta_q=0.2),
        ds_params_b=DSSplitBinEFT(label="DS5", bq1=1.1, beta_q=-0.1),
        R=10.0, ds_model="phenomenological", mode="tree",
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_pqq_eft_ct_zero_corrections_is_tree(cosmo, k, mu, ds_a, ds_b):
    tree = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="rsd_selection", mode="tree",
    )
    eft_ct = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="rsd_selection", mode="eft_ct",
    )
    np.testing.assert_allclose(eft_ct, tree, rtol=1e-12)


def test_pqq_eft_is_symmetric(cosmo, k, mu, ds_a, ds_b):
    pab = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="phenomenological", mode="eft_ct",
    )
    pba = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_b, ds_params_b=ds_a,
        R=10.0, ds_model="phenomenological", mode="eft_ct",
    )
    np.testing.assert_allclose(pab, pba, rtol=1e-12)


def test_pqq_counterterm_scales_linearly_per_leg(cosmo, k, mu, ds_b):
    base_a = DSSplitBinEFT(label="DS2", bq1=-0.7, beta_q=0.2, bq_nabla2=0.4)
    dbl_a = DSSplitBinEFT(label="DS2", bq1=-0.7, beta_q=0.2, bq_nabla2=0.8)

    p0 = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=DSSplitBinEFT(label="DS2", bq1=-0.7, beta_q=0.2, bq_nabla2=0.0),
        ds_params_b=ds_b, R=10.0, ds_model="phenomenological", mode="eft_ct",
    )
    p1 = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=base_a, ds_params_b=ds_b,
        R=10.0, ds_model="phenomenological", mode="eft_ct",
    )
    p2 = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=dbl_a, ds_params_b=ds_b,
        R=10.0, ds_model="phenomenological", mode="eft_ct",
    )
    np.testing.assert_allclose(p2 - p0, 2.0 * (p1 - p0), rtol=1e-12)


def test_density_split_pair_stochastic_is_isotropic(k, mu):
    stoch = density_split_pair_stochastic_term(k, mu, sqq0=5.0, sqq2=2.0)
    expected = (5.0 + 2.0 * k**2)[:, np.newaxis] * np.ones((1, len(mu)))
    np.testing.assert_allclose(stoch, expected, rtol=1e-12)


def test_pqq_eft_adds_pair_stochastic(cosmo, k, mu, ds_a, ds_b):
    no_stoch = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="baseline", mode="eft", sqq0=0.0, sqq2=0.0,
    )
    with_stoch = pqq_eft_mu(
        k, mu, z=0.5, cosmo=cosmo,
        ds_params_a=ds_a, ds_params_b=ds_b,
        R=10.0, ds_model="baseline", mode="eft", sqq0=3.0, sqq2=1.5,
    )
    expected_shift = density_split_pair_stochastic_term(k, mu, sqq0=3.0, sqq2=1.5)
    np.testing.assert_allclose(with_stoch - no_stoch, expected_shift, rtol=1e-12)


def test_pqq_one_loop_rejects_unimplemented_ds_terms(cosmo, k, mu, ds_b):
    ds_bad = DSSplitBinEFT(label="DS2", bq1=-0.7, bq2=1.0)
    with pytest.raises(NotImplementedError, match="bq2 and bqK2"):
        pqq_eft_mu(
            k, mu, z=0.5, cosmo=cosmo,
            ds_params_a=ds_bad, ds_params_b=ds_b,
            R=10.0, ds_model="baseline", mode="one_loop",
        )
