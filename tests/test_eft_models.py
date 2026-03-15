"""Tests for EFT density-split × galaxy power spectrum models."""

import numpy as np
import pytest

from drift.cosmology import get_cosmology, get_linear_power, get_growth_rate
from drift.bias import DSSplitBin
from drift.models import pqg_mu
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.eft_models import pqg_eft_mu
from drift.eft_terms import galaxy_counterterm, ds_counterterm, stochastic_term
from drift.multipoles import compute_multipoles


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
def ds_eft():
    return DSSplitBinEFT(label="DS3", bq1=0.5)


@pytest.fixture
def gal():
    return GalaxyEFTParams(b1=1.5)


@pytest.fixture
def plin(cosmo, k):
    return get_linear_power(cosmo, k, 0.5)


# ---------------------------------------------------------------------------
# tree_only mode matches legacy pqg_mu
# ---------------------------------------------------------------------------

def test_tree_only_matches_pqg_mu(cosmo, k, mu, ds_eft, gal):
    """tree_only mode must reproduce pqg_mu exactly at machine precision."""
    ds_legacy = DSSplitBin(label="DS3", bq=ds_eft.bq1, cq=0.0)
    expected = pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_legacy,
                      tracer_bias=gal.b1, R=10.0, ds_model="baseline")
    result = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                        gal_params=gal, R=10.0, ds_model="baseline", mode="tree_only")
    np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# eft_lite with all-zero corrections equals tree_only
# ---------------------------------------------------------------------------

def test_eft_lite_zero_corrections_is_tree(cosmo, k, mu, ds_eft, gal, plin):
    """eft_lite with zero loop/ct params should equal tree_only."""
    # Precompute p1loop = plin so delta_p = 0
    p1loop_pre = plin.copy()
    tree = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal, R=10.0, mode="tree_only")
    lite = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal, R=10.0, mode="eft_lite",
                      loop_kwargs={"p1loop_precomputed": p1loop_pre})
    np.testing.assert_allclose(lite, tree, rtol=1e-12)


# ---------------------------------------------------------------------------
# galaxy counterterm
# ---------------------------------------------------------------------------

def test_galaxy_counterterm_vanishes_when_c_zero(k, mu, plin):
    gal_zero = GalaxyEFTParams(b1=1.5, c0=0.0, c2=0.0, c4=0.0)
    ct = galaxy_counterterm(k, mu, plin, gal_zero)
    np.testing.assert_allclose(ct, 0.0, atol=1e-30)


def test_galaxy_counterterm_k2_scaling(k, mu, plin):
    """With c0 only, ct = -c0 * k^2 * P_lin (independent of mu)."""
    c0 = 2.5
    gal = GalaxyEFTParams(b1=1.0, c0=c0)
    ct = galaxy_counterterm(k, mu, plin, gal)
    expected = -c0 * (k**2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))
    np.testing.assert_allclose(ct, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# DS nabla2 counterterm
# ---------------------------------------------------------------------------

def test_ds_nabla2_counterterm_k2R2_scaling(cosmo, k, mu, plin, gal):
    """DS counterterm should scale as bq_nabla2 * (kR)^2."""
    R = 10.0
    bq_nabla2 = 0.3
    ds_a = DSSplitBinEFT(label="A", bq1=1.0, bq_nabla2=bq_nabla2)
    ds_b = DSSplitBinEFT(label="B", bq1=1.0, bq_nabla2=2.0 * bq_nabla2)

    # tree_normed: bq1=1 tree model used as shape
    from drift.cosmology import get_growth_rate
    from drift.kernels import gaussian_kernel
    from drift.eft_models import _pqg_tree_eft
    from drift.eft_bias import GalaxyEFTParams as GP

    f = get_growth_rate(cosmo, 0.5)
    wk = gaussian_kernel(k, R)
    ds_n = DSSplitBinEFT(label="N", bq1=1.0)
    gn = GP(b1=gal.b1)
    tree_normed = _pqg_tree_eft(k, mu, plin, wk, f, ds_n, gn, "baseline")

    ct_a = ds_counterterm(k, mu, plin, ds_a, tree_normed, R)
    ct_b = ds_counterterm(k, mu, plin, ds_b, tree_normed, R)

    # ct_b / ct_a should equal 2 everywhere
    np.testing.assert_allclose(ct_b, 2.0 * ct_a, rtol=1e-12)


# ---------------------------------------------------------------------------
# stochastic term
# ---------------------------------------------------------------------------

def test_stochastic_zero_when_s_zero(k, mu):
    gal = GalaxyEFTParams(b1=1.0, s0=0.0, s2=0.0)
    st = stochastic_term(k, mu, gal)
    np.testing.assert_allclose(st, 0.0, atol=1e-30)


def test_stochastic_s0_is_white_noise(k, mu):
    """s0-only stochastic term is constant in k and mu."""
    s0 = 1234.5
    gal = GalaxyEFTParams(b1=1.0, s0=s0, s2=0.0)
    st = stochastic_term(k, mu, gal)
    np.testing.assert_allclose(st, s0, rtol=1e-12)


# ---------------------------------------------------------------------------
# multipoles from eft_lite
# ---------------------------------------------------------------------------

def test_eft_lite_multipoles_finite(cosmo, k, ds_eft, gal):
    """Projected multipoles (ell=0,2,4) from eft_lite should all be finite."""
    # Use a small integration grid for speed
    plin_func = lambda kk: get_linear_power(cosmo, kk, 0.5)
    loop_result = __import__("drift").compute_one_loop_matter(
        k, plin_func, n_q_22=32, n_mu_22=32, n_q_13=64
    )
    p1loop_pre = loop_result["p1loop"]

    def model(kk, mu):
        return pqg_eft_mu(
            kk, mu, z=0.5, cosmo=cosmo,
            ds_params=ds_eft, gal_params=gal, R=10.0,
            mode="eft_lite",
            loop_kwargs={"p1loop_precomputed": p1loop_pre},
        )

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    for ell, arr in poles.items():
        assert np.all(np.isfinite(arr)), f"P{ell} has non-finite values: {arr}"


# ---------------------------------------------------------------------------
# invalid mode raises ValueError
# ---------------------------------------------------------------------------

def test_mode_invalid_raises(cosmo, k, mu, ds_eft, gal):
    with pytest.raises(ValueError, match="Unknown mode"):
        pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                   gal_params=gal, R=10.0, mode="garbage")


# ---------------------------------------------------------------------------
# bq2, bqK2 irrelevant in tree_only mode
# ---------------------------------------------------------------------------

def test_ds_quadratic_zero_has_no_effect_in_tree_only(cosmo, k, mu, gal):
    """bq2 and bqK2 are one-loop terms; tree_only ignores them."""
    ds_a = DSSplitBinEFT(label="A", bq1=0.5, bq2=0.0, bqK2=0.0)
    ds_b = DSSplitBinEFT(label="B", bq1=0.5, bq2=9.9, bqK2=9.9)

    pa = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_a,
                    gal_params=gal, R=10.0, mode="tree_only")
    pb = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_b,
                    gal_params=gal, R=10.0, mode="tree_only")
    np.testing.assert_allclose(pa, pb, rtol=1e-12)
