"""Correctness tests for TemplateEmulator.

Verifies that the analytic template multipoles match direct numerical
Gauss-Legendre projection of pqg_eft_mu to rtol=1e-10.
"""

import numpy as np
import pytest

from drift.cosmology import get_cosmology, get_linear_power, get_growth_rate
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.eft_models import pqg_eft_mu
from drift.multipoles import compute_multipoles
from drift.emulator import TemplateEmulator

Z = 0.5
R = 10.0
ELLS = (0, 2)


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture(scope="module")
def k():
    return np.logspace(-2, np.log10(0.3), 60)


# ---------------------------------------------------------------------------
# Helper: reference multipoles via direct GL quadrature
# ---------------------------------------------------------------------------

def _reference_multipoles(cosmo, k, ds_params, gal_params, ds_model, mode):
    """Compute reference multipoles using pqg_eft_mu + Gauss-Legendre."""
    def model(kk, mu):
        return pqg_eft_mu(
            kk, mu, z=Z, cosmo=cosmo,
            ds_params=ds_params, gal_params=gal_params,
            R=R, kernel="gaussian", space="redshift",
            ds_model=ds_model, mode=mode,
        )
    poles = compute_multipoles(k, model, ells=ELLS)
    return np.concatenate([poles[ell] for ell in ELLS])


def _emulator_predict(cosmo, k, ds_params, gal_params, ds_model, mode):
    """Predict multipoles via TemplateEmulator."""
    em = TemplateEmulator(
        cosmo, k, ells=ELLS, z=Z, R=R,
        kernel="gaussian", space="redshift",
        ds_model=ds_model, mode=mode,
    )
    params = {
        "b1": gal_params.b1,
        "bq1": [ds_params.bq1],
        "c0": gal_params.c0,
        "c2": gal_params.c2,
        "c4": gal_params.c4,
        "s0": gal_params.s0,
        "s2": gal_params.s2,
        "beta_q": [ds_params.beta_q],
        "bq_nabla2": [ds_params.bq_nabla2],
    }
    return em.predict(params)


# ---------------------------------------------------------------------------
# tree_only × all ds_models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_model", ["baseline", "rsd_selection", "phenomenological"])
def test_tree_only_matches_reference(cosmo, k, ds_model):
    ds = DSSplitBinEFT(label="DS3", bq1=0.7, beta_q=0.4)
    gal = GalaxyEFTParams(b1=1.8)
    ref = _reference_multipoles(cosmo, k, ds, gal, ds_model, "tree")
    got = _emulator_predict(cosmo, k, ds, gal, ds_model, "tree")
    np.testing.assert_allclose(got, ref, rtol=1e-10,
        err_msg=f"tree_only / {ds_model}: emulator vs GL mismatch")


# ---------------------------------------------------------------------------
# eft_lite × all ds_models (galaxy counterterm c0 only, bq_nabla2=0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_model", ["baseline", "rsd_selection", "phenomenological"])
def test_eft_lite_matches_reference(cosmo, k, ds_model):
    ds = DSSplitBinEFT(label="DS1", bq1=-1.3, beta_q=0.3)
    gal = GalaxyEFTParams(b1=2.1, c0=5.0)
    ref = _reference_multipoles(cosmo, k, ds, gal, ds_model, "eft_ct")
    got = _emulator_predict(cosmo, k, ds, gal, ds_model, "eft_ct")
    np.testing.assert_allclose(got, ref, rtol=1e-10,
        err_msg=f"eft_lite / {ds_model}: emulator vs GL mismatch")


# ---------------------------------------------------------------------------
# eft_full × all ds_models (c0 + s0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_model", ["baseline", "rsd_selection", "phenomenological"])
def test_eft_full_matches_reference(cosmo, k, ds_model):
    ds = DSSplitBinEFT(label="DS5", bq1=1.5, beta_q=-0.2)
    gal = GalaxyEFTParams(b1=1.6, c0=3.0, s0=200.0)
    ref = _reference_multipoles(cosmo, k, ds, gal, ds_model, "eft")
    got = _emulator_predict(cosmo, k, ds, gal, ds_model, "eft")
    np.testing.assert_allclose(got, ref, rtol=1e-10,
        err_msg=f"eft_full / {ds_model}: emulator vs GL mismatch")


# ---------------------------------------------------------------------------
# Non-zero bq_nabla2 (DS higher-derivative counterterm)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_model", ["baseline", "rsd_selection", "phenomenological"])
def test_bq_nabla2_matches_reference(cosmo, k, ds_model):
    ds = DSSplitBinEFT(label="DS2", bq1=0.9, bq_nabla2=0.5, beta_q=0.2)
    gal = GalaxyEFTParams(b1=2.0, c0=4.0)
    ref = _reference_multipoles(cosmo, k, ds, gal, ds_model, "eft_ct")
    got = _emulator_predict(cosmo, k, ds, gal, ds_model, "eft_ct")
    np.testing.assert_allclose(got, ref, rtol=1e-10,
        err_msg=f"bq_nabla2 / {ds_model}: emulator vs GL mismatch")


# ---------------------------------------------------------------------------
# Non-zero c2 and c4 galaxy counterterms
# ---------------------------------------------------------------------------

def test_c2_c4_counterterms_match_reference(cosmo, k):
    ds = DSSplitBinEFT(label="DS3", bq1=0.6)
    gal = GalaxyEFTParams(b1=1.9, c0=2.0, c2=1.0, c4=0.5)
    ref = _reference_multipoles(cosmo, k, ds, gal, "baseline", "eft_ct")
    got = _emulator_predict(cosmo, k, ds, gal, "baseline", "eft_ct")
    np.testing.assert_allclose(got, ref, rtol=1e-10,
        err_msg="c2/c4 counterterms: emulator vs GL mismatch")


# ---------------------------------------------------------------------------
# Multiple quantiles: predict returns correct ordering
# ---------------------------------------------------------------------------

def test_multi_quantile_ordering(cosmo, k):
    """Two-quantile predict should equal two independent single-quantile calls."""
    em = TemplateEmulator(
        cosmo, k, ells=ELLS, z=Z, R=R,
        ds_model="phenomenological", mode="eft_ct",
    )
    params_multi = {
        "b1": 2.0,
        "bq1": [-1.2, 1.4],
        "beta_q": [0.3, -0.2],
        "c0": 5.0,
    }
    multi = em.predict(params_multi)

    expected = []
    for bq1_val, beta_val in zip(params_multi["bq1"], params_multi["beta_q"]):
        params_single = {
            "b1": 2.0,
            "bq1": [bq1_val],
            "beta_q": [beta_val],
            "c0": 5.0,
        }
        expected.append(em.predict(params_single))

    np.testing.assert_array_equal(multi, np.concatenate(expected))


# ---------------------------------------------------------------------------
# Linearity in bias parameters (analytic property of the model)
# ---------------------------------------------------------------------------

def test_linearity_in_bq1(cosmo, k):
    """predict is linear in bq1 (tree + EFT contributions both linear in bq1)."""
    em = TemplateEmulator(
        cosmo, k, ells=(0, 2), z=Z, R=R,
        ds_model="baseline", mode="eft_ct",
    )
    base = {"b1": 1.8, "bq1": [1.0], "c0": 3.0}
    p1 = em.predict(base)
    p2 = em.predict({**base, "bq1": [2.0]})
    # p2 should equal 2 * p1 only if c0=0; with c0≠0 and tree scaling, not simply 2x.
    # Instead verify: p(2*bq1) - p(bq1) = p(bq1) - p(0*bq1)
    p0 = em.predict({**base, "bq1": [0.0]})
    np.testing.assert_allclose(p2 - p1, p1 - p0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Real-space (f=0): only ell=0 is non-zero
# ---------------------------------------------------------------------------

def test_real_space_ell0_only(cosmo, k):
    """In real space (f=0), monopole should be non-zero but quadrupole zero."""
    em = TemplateEmulator(
        cosmo, k, ells=(0, 2), z=Z, R=R,
        space="real", ds_model="baseline", mode="tree",
    )
    out = em.predict({"b1": 2.0, "bq1": [1.0]})
    nk = len(k)
    p0 = out[:nk]
    p2 = out[nk:]
    assert np.all(p0 != 0), "Monopole should be non-zero in real space"
    np.testing.assert_allclose(p2, 0.0, atol=1e-30,
        err_msg="Quadrupole should vanish in real space (f=0)")


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_invalid_ds_model_raises(cosmo, k):
    with pytest.raises(ValueError, match="Unknown ds_model"):
        TemplateEmulator(cosmo, k, ds_model="bogus")


def test_invalid_mode_raises(cosmo, k):
    with pytest.raises(ValueError, match="Unknown mode"):
        TemplateEmulator(cosmo, k, mode="bogus")


def test_invalid_ell_raises(cosmo, k):
    with pytest.raises(ValueError, match="Unsupported ell"):
        TemplateEmulator(cosmo, k, ells=(0, 3))


# ---------------------------------------------------------------------------
# update_cosmology: must match a freshly constructed emulator
# ---------------------------------------------------------------------------

def test_update_cosmology_matches_new_instance(k):
    """update_cosmology(plin, f) must produce identical output to a fresh instance."""
    cosmo_old = get_cosmology({"sigma8": 0.75, "Omega_m": 0.28})
    cosmo_new = get_cosmology({"sigma8": 0.85, "Omega_m": 0.33})

    plin_new = get_linear_power(cosmo_new, k, Z)
    f_new    = get_growth_rate(cosmo_new, Z)

    # Reference: fresh emulator at new cosmology
    em_ref = TemplateEmulator(cosmo_new, k, ells=ELLS, z=Z, R=R,
                              ds_model="baseline", mode="eft_ct")
    params = {"b1": 2.0, "bq1": [0.5, -1.0], "c0": 3.0}
    pred_ref = em_ref.predict(params)

    # Test: old emulator updated in-place
    em_upd = TemplateEmulator(cosmo_old, k, ells=ELLS, z=Z, R=R,
                              ds_model="baseline", mode="eft_ct")
    em_upd.update_cosmology(plin_new, f_new)
    pred_upd = em_upd.predict(params)

    np.testing.assert_allclose(pred_upd, pred_ref, rtol=1e-12,
        err_msg="update_cosmology result differs from fresh TemplateEmulator")
