"""Tests for EFT density-split × galaxy power spectrum models."""

import numpy as np
import pytest

from drift.utils.cosmology import get_cosmology, get_linear_power, get_growth_rate
from drift.theory.density_split.bias import DSSplitBin
from drift.theory.density_split.power_spectrum import pqg_mu
from drift.theory.density_split.bias import DSSplitBinEFT
from drift.theory.galaxy.bias import GalaxyEFTParams
from drift.theory.density_split.eft_power_spectrum import pqg_eft_mu, _pqg_ds_lin
from drift.theory.density_split.counterterms import galaxy_counterterm, ds_counterterm, stochastic_term
from drift.utils.multipoles import compute_multipoles


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
                        gal_params=gal, R=10.0, ds_model="baseline", mode="tree")
    np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# eft_lite with all-zero corrections equals tree_only
# ---------------------------------------------------------------------------

def test_eft_lite_zero_corrections_is_tree(cosmo, k, mu, ds_eft, gal):
    """eft_lite with zero counterterm params should equal tree_only."""
    # gal has c0=c2=c4=0 and ds_eft has bq_nabla2=0 by default
    tree = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal, R=10.0, mode="tree")
    lite = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal, R=10.0, mode="eft_ct")
    np.testing.assert_allclose(lite, tree, rtol=1e-12)


# ---------------------------------------------------------------------------
# galaxy counterterm
# ---------------------------------------------------------------------------

def test_galaxy_counterterm_vanishes_when_c_zero(k, mu, plin):
    gal_zero = GalaxyEFTParams(b1=1.5, c0=0.0, c2=0.0, c4=0.0)
    ds_lin = plin[:, np.newaxis] * np.ones((1, len(mu)))
    ct = galaxy_counterterm(k, mu, gal_zero, ds_lin)
    np.testing.assert_allclose(ct, 0.0, atol=1e-30)


def test_galaxy_counterterm_k2_scaling(k, mu, plin):
    """With c0 only and ds_lin=P_lin, ct = -c0 * k^2 * P_lin (independent of mu)."""
    c0 = 2.5
    gal = GalaxyEFTParams(b1=1.0, c0=c0)
    ds_lin = plin[:, np.newaxis] * np.ones((1, len(mu)))
    ct = galaxy_counterterm(k, mu, gal, ds_lin)
    expected = -c0 * (k**2)[:, np.newaxis] * ds_lin
    np.testing.assert_allclose(ct, expected, rtol=1e-12)


def test_galaxy_ct_scales_with_ds_amplitude(k, mu, plin):
    """Doubling ds_lin should double the galaxy counterterm."""
    gal = GalaxyEFTParams(b1=1.0, c0=2.0)
    ds_lin = (plin * 0.7)[:, np.newaxis] * np.ones((1, len(mu)))
    ct1 = galaxy_counterterm(k, mu, gal, ds_lin)
    ct2 = galaxy_counterterm(k, mu, gal, 2.0 * ds_lin)
    np.testing.assert_allclose(ct2, 2.0 * ct1, rtol=1e-12)


def test_galaxy_ct_opposite_sign_for_ds1_and_ds5(cosmo, k, mu, gal, plin):
    """DS1 (bq1 < 0) and DS5 (bq1 > 0) receive opposite-sign galaxy counterterms."""
    from drift.utils.kernels import gaussian_kernel
    from drift.utils.cosmology import get_growth_rate
    R = 10.0
    wk = gaussian_kernel(k, R)
    f = get_growth_rate(cosmo, 0.5)
    c0 = 2.0
    gal_ct = GalaxyEFTParams(b1=gal.b1, c0=c0)

    ds1 = DSSplitBinEFT(label="DS1", bq1=-1.5)
    ds5 = DSSplitBinEFT(label="DS5", bq1=1.5)
    ds_lin_ds1 = _pqg_ds_lin(k, mu, plin, wk, f, ds1, "baseline")
    ds_lin_ds5 = _pqg_ds_lin(k, mu, plin, wk, f, ds5, "baseline")
    ct_ds1 = galaxy_counterterm(k, mu, gal_ct, ds_lin_ds1)
    ct_ds5 = galaxy_counterterm(k, mu, gal_ct, ds_lin_ds5)

    # ct_ds5 / ct_ds1 should equal bq1_ds5 / bq1_ds1 = -1 everywhere
    np.testing.assert_allclose(ct_ds5, -1.0 * ct_ds1, rtol=1e-12)


def test_galaxy_ct_inherits_ds_angular_structure_rsd_selection(cosmo, k, mu, plin):
    """For rsd_selection, the galaxy CT should carry the DS (1+f*mu^2) angular factor.

    ds_lin for rsd_selection = bq1 * plin * wk * (1 + f*mu^2), so the galaxy CT
    must differ from baseline by the factor (1 + f*mu^2) at each (k, mu).
    """
    from drift.utils.kernels import gaussian_kernel
    R = 10.0
    wk = gaussian_kernel(k, R)
    f = get_growth_rate(cosmo, 0.5)
    bq1 = 0.8
    ds = DSSplitBinEFT(label="DS3", bq1=bq1)
    gal_ct = GalaxyEFTParams(b1=1.5, c0=2.0)

    ds_lin_base = _pqg_ds_lin(k, mu, plin, wk, f, ds, "baseline")
    ds_lin_rsd = _pqg_ds_lin(k, mu, plin, wk, f, ds, "rsd_selection")

    ct_base = galaxy_counterterm(k, mu, gal_ct, ds_lin_base)
    ct_rsd = galaxy_counterterm(k, mu, gal_ct, ds_lin_rsd)

    # ct_rsd / ct_base should equal (1 + f*mu^2) at each (k, mu)
    angular_factor = (1.0 + f * mu**2)[np.newaxis, :]
    np.testing.assert_allclose(ct_rsd, ct_base * angular_factor, rtol=1e-12)


def test_galaxy_ct_perturbative_at_low_k(cosmo, k, mu, ds_eft, gal, plin):
    """Galaxy CT contribution should be small relative to tree at low k.

    A well-behaved EFT counterterm scales as k^2 and is negligible compared
    to the tree-level cross-spectrum at k < 0.05 h/Mpc.
    """
    from drift.utils.kernels import gaussian_kernel
    R = 10.0
    wk = gaussian_kernel(k, R)

    gal_ct = GalaxyEFTParams(b1=gal.b1, c0=5.0)   # deliberately large c0
    from drift.theory.density_split.eft_power_spectrum import _pqg_tree_eft
    f = get_growth_rate(cosmo, 0.5)
    ds_lin = _pqg_ds_lin(k, mu, plin, wk, f, ds_eft, "baseline")
    ct = galaxy_counterterm(k, mu, gal_ct, ds_lin)

    # Tree-level for the same ds_params
    tree = _pqg_tree_eft(k, mu, plin, wk, f, ds_eft, gal, "baseline")

    k_low_mask = k < 0.05
    if not np.any(k_low_mask):
        pytest.skip("No k < 0.05 in k array")

    ratio = np.abs(ct[k_low_mask]) / np.abs(tree[k_low_mask])
    assert np.all(ratio < 0.1), (
        f"Galaxy CT / tree > 10% at low k — CT may not be perturbative: ratio = {ratio}"
    )


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
    from drift.utils.cosmology import get_growth_rate
    from drift.utils.kernels import gaussian_kernel
    from drift.theory.density_split.eft_power_spectrum import _pqg_tree_eft
    from drift.theory.galaxy.bias import GalaxyEFTParams as GP

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
    def model(kk, mu):
        return pqg_eft_mu(
            kk, mu, z=0.5, cosmo=cosmo,
            ds_params=ds_eft, gal_params=gal, R=10.0,
            mode="eft_ct",
        )

    poles = compute_multipoles(k, model, ells=(0, 2, 4))
    for ell, arr in poles.items():
        assert np.all(np.isfinite(arr)), f"P{ell} has non-finite values: {arr}"


# ---------------------------------------------------------------------------
# eft_lite is perturbative at low k
# ---------------------------------------------------------------------------

def test_eft_lite_perturbative_at_low_k(cosmo, k, mu, ds_eft, gal):
    """eft_lite / tree_only should be close at k < 0.05 (counterterms are small there)."""
    gal_ct = GalaxyEFTParams(b1=gal.b1, c0=5.0)
    tree = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal_ct, R=10.0, mode="tree")
    lite = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_eft,
                      gal_params=gal_ct, R=10.0, mode="eft_ct")
    k_low = k < 0.05
    if not np.any(k_low):
        pytest.skip("No k < 0.05")
    ratio = np.abs((lite[k_low] - tree[k_low]) / tree[k_low])
    assert np.all(ratio < 0.15), (
        f"eft_lite deviates > 15% from tree at low k: max ratio = {ratio.max():.3f}"
    )


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
                    gal_params=gal, R=10.0, mode="tree")
    pb = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_b,
                    gal_params=gal, R=10.0, mode="tree")
    np.testing.assert_allclose(pa, pb, rtol=1e-12)


def test_ds_quadratic_raises_in_one_loop(cosmo, k, mu, gal):
    """one_loop must reject unimplemented bq2 and bqK2 contributions."""
    ds = DSSplitBinEFT(label="A", bq1=0.5, bq2=1.0)
    with pytest.raises(NotImplementedError, match="bq2 and bqK2"):
        pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds,
                   gal_params=gal, R=10.0, mode="one_loop")
