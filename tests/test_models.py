"""Tests for drift.models."""

import numpy as np
import pytest
from drift.theory.density_split.bias import DSSplitBin
from drift.theory.density_split.power_spectrum import pqm_mu, pqg_mu
from drift.utils.cosmology import get_cosmology
from drift.utils.multipoles import compute_multipoles


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture
def ds_bin():
    return DSSplitBin(label="DS3", bq=0.5, cq=0.0)


def test_pqm_shape(cosmo, ds_bin):
    k = np.logspace(-2, -0.5, 30)
    mu = np.linspace(-1, 1, 20)
    result = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, R=10.0)
    assert result.shape == (30, 20)


def test_pqm_finite(cosmo, ds_bin):
    k = np.logspace(-2, -0.5, 30)
    mu = np.linspace(-1, 1, 20)
    result = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, R=10.0)
    assert np.all(np.isfinite(result))


def test_pqg_shape(cosmo, ds_bin):
    k = np.logspace(-2, -0.5, 30)
    mu = np.linspace(-1, 1, 20)
    result = pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, tracer_bias=1.5, R=10.0)
    assert result.shape == (30, 20)


def test_pqg_reduces_to_pqm_when_b1_is_1(cosmo, ds_bin):
    """pqg with b1=1 should equal pqm (both use [1 + f*mu^2])."""
    k = np.logspace(-2, -0.5, 30)
    mu = np.linspace(-1, 1, 20)
    qm = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, R=10.0, ds_model="baseline")
    qg = pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, tracer_bias=1.0, R=10.0, ds_model="baseline")
    np.testing.assert_allclose(qm, qg, rtol=1e-12)


def test_monopole_sign(cosmo):
    """Bins with bq < 0 should have negative monopole; bq > 0 positive."""
    k = np.logspace(-1.5, -0.5, 20)
    z = 0.5
    R = 10.0

    for bq, expected_sign in [(-1.5, -1), (1.5, +1)]:
        ds = DSSplitBin(label="test", bq=bq)

        def model(kk, mu, _ds=ds):
            return pqm_mu(kk, mu, z, cosmo, _ds, R)

        p0 = compute_multipoles(k, model, ells=(0,))[0]
        assert np.all(np.sign(p0) == expected_sign), (
            f"Expected {expected_sign} monopole for bq={bq}, got {p0}"
        )


def test_nabla_bias_effect(cosmo):
    """Non-zero cq should modify the spectrum at high k."""
    k = np.logspace(-2, -0.5, 30)
    mu = np.array([0.0])

    ds_no_nabla = DSSplitBin(label="A", bq=1.0, cq=0.0)
    ds_nabla = DSSplitBin(label="B", bq=1.0, cq=0.5)

    p_no = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_no_nabla, R=10.0)
    p_nab = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_nabla, R=10.0)

    # They should differ at high k
    assert not np.allclose(p_no[-5:, 0], p_nab[-5:, 0])


def test_baseline_hexadecapole_vanishes(cosmo, ds_bin):
    """P4 must be zero (analytically) for the baseline model."""
    k = np.logspace(-2, -0.5, 30)

    def model(kk, mu):
        return pqg_mu(kk, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin,
                      tracer_bias=1.5, R=10.0, ds_model="baseline")

    p4 = compute_multipoles(k, model, ells=(4,))[4]
    np.testing.assert_allclose(p4, 0.0, atol=1e-8)


def test_rsd_selection_hexadecapole_nonzero(cosmo, ds_bin):
    """P4 must be nonzero for rsd_selection (has mu^4 terms)."""
    k = np.logspace(-2, -0.5, 30)

    def model(kk, mu):
        return pqg_mu(kk, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin,
                      tracer_bias=1.5, R=10.0, ds_model="rsd_selection")

    p4 = compute_multipoles(k, model, ells=(4,))[4]
    assert np.any(np.abs(p4) > 1e-10)


def test_pheno_beta0_matches_baseline(cosmo, ds_bin):
    """phenomenological with beta_q=0 must reproduce baseline exactly."""
    k = np.logspace(-2, -0.5, 30)
    mu = np.linspace(-1, 1, 20)

    ds_bin_pheno = DSSplitBin(label="X", bq=ds_bin.bq, cq=ds_bin.cq, beta_q=0.0)

    baseline = pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin,
                      tracer_bias=1.5, R=10.0, ds_model="baseline")
    pheno = pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin_pheno,
                   tracer_bias=1.5, R=10.0, ds_model="phenomenological")

    np.testing.assert_allclose(pheno, baseline, rtol=1e-12)


def test_pheno_beta_bq_shape(cosmo):
    """With beta_q = bq and cq=0, DS factor is bq*(1 + f*mu^2).

    Numerically verify P0 and P2 via Gauss-Legendre integration.
    """
    from drift.utils.cosmology import get_growth_rate
    from drift.utils.kernels import gaussian_kernel
    from drift.utils.cosmology import get_linear_power

    bq = 1.0
    b1 = 2.0
    R = 10.0
    z = 0.5
    ds_bin_pheno = DSSplitBin(label="Y", bq=bq, cq=0.0, beta_q=bq)

    k = np.logspace(-2, -0.5, 30)

    def model(kk, mu):
        return pqg_mu(kk, mu, z=z, cosmo=cosmo, ds_params=ds_bin_pheno,
                      tracer_bias=b1, R=R, ds_model="phenomenological")

    poles = compute_multipoles(k, model, ells=(0, 2))
    p0_num = poles[0]
    p2_num = poles[2]

    # Analytic expectation: DS factor = bq*(1 + f*mu^2) * (b1 + f*mu^2)
    # = bq * [b1 + f*(1+b1)*mu^2 + f^2*mu^4]
    # P0 = bq * A(k) * (b1 + f*(1+b1)/3 + f^2/5)
    # P2 = bq * A(k) * (2*f*(1+b1)/3 + 4*f^2/7)  [times 5/2 from legendre norm, included in compute_multipoles]
    f = get_growth_rate(cosmo, z)
    wk = gaussian_kernel(k, R)
    plin = get_linear_power(cosmo, k, z)
    Ak = bq * wk * plin

    p0_analytic = Ak * (b1 + f * (1 + b1) / 3 + f**2 / 5)
    p2_analytic = Ak * (2 * f * (1 + b1) / 3 + 4 * f**2 / 7)

    np.testing.assert_allclose(p0_num, p0_analytic, rtol=1e-4)
    np.testing.assert_allclose(p2_num, p2_analytic, rtol=1e-4)


def test_rsd_selection_multiplicative_formula(cosmo):
    """pqg_mu rsd_selection must equal bq_eff*(1+f*mu^2)*(b1+f*mu^2)*W_R*P_lin."""
    from drift.utils.cosmology import get_linear_power, get_growth_rate
    from drift.utils.kernels import gaussian_kernel

    k = np.logspace(-2, -0.5, 20)
    mu = np.linspace(-1, 1, 15)
    z = 0.5
    R = 10.0
    b1 = 1.5
    bq = 0.8
    cq = 0.3
    ds = DSSplitBin(label="T", bq=bq, cq=cq)

    result = pqg_mu(k, mu, z=z, cosmo=cosmo, ds_params=ds, tracer_bias=b1,
                    R=R, ds_model="rsd_selection")

    plin = get_linear_power(cosmo, k, z)
    wk = gaussian_kernel(k, R)
    f = get_growth_rate(cosmo, z)
    bq_eff = bq + cq * (k * R) ** 2
    expected = (
        (bq_eff * plin * wk)[:, np.newaxis]
        * (1.0 + f * mu[np.newaxis, :] ** 2)
        * (b1 + f * mu[np.newaxis, :] ** 2)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_invalid_ds_model_raises(cosmo, ds_bin):
    k = np.logspace(-2, -0.5, 10)
    mu = np.linspace(-1, 1, 5)
    with pytest.raises(ValueError):
        pqg_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin,
               tracer_bias=1.5, R=10.0, ds_model="bad_name")
