"""Unit tests for one-loop SPT integrals."""

import numpy as np
import pytest

from drift.cosmology import get_cosmology, get_linear_power
from drift.one_loop import compute_P22, compute_P13, compute_one_loop_matter


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology()


@pytest.fixture(scope="module")
def plin_func(cosmo):
    def _f(k):
        return get_linear_power(cosmo, np.asarray(k, dtype=float), 0.5)
    return _f


@pytest.fixture(scope="module")
def k_arr():
    return np.logspace(-2, 0, 20)


@pytest.fixture(scope="module")
def loop_dict(k_arr, plin_func):
    # Use small grids for speed in tests
    return compute_one_loop_matter(
        k_arr, plin_func,
        q_min=1e-4, q_max=10.0,
        n_q_22=64, n_mu_22=64, n_q_13=128,
    )


def test_P22_finite(k_arr, plin_func):
    p22 = compute_P22(k_arr, plin_func, n_q=64, n_mu=64)
    assert p22.shape == (len(k_arr),)
    assert np.all(np.isfinite(p22))


def test_P22_positive(k_arr, plin_func):
    p22 = compute_P22(k_arr, plin_func, n_q=64, n_mu=64)
    assert np.all(p22 >= 0.0), f"P22 has negative values: {p22[p22 < 0]}"


def test_P13_finite(k_arr, plin_func):
    p13 = compute_P13(k_arr, plin_func, n_q=128)
    assert p13.shape == (len(k_arr),)
    assert np.all(np.isfinite(p13))


def test_P13_negative_at_intermediate_k(plin_func):
    k_mid = np.logspace(np.log10(0.05), np.log10(0.5), 10)
    p13 = compute_P13(k_mid, plin_func, n_q=128)
    assert np.all(p13 < 0), f"2P13 should be negative at intermediate k; got {p13}"


def test_P22_subdominant_at_low_k(loop_dict, k_arr):
    """P22 << Plin at small k because P22 ~ Plin^2 (quadratic in amplitude).

    Note: P13 is NOT small at low k — it has a UV-sensitive contribution
    proportional to Plin (known SPT behaviour). Only P22 is guaranteed
    small in the linear regime.
    """
    k_low_mask = k_arr < 0.05
    if not np.any(k_low_mask):
        pytest.skip("No k < 0.05 in k_arr")
    ratio = loop_dict["p22"][k_low_mask] / loop_dict["plin"][k_low_mask]
    assert np.all(ratio < 0.1), \
        f"P22/Plin > 10% at low k: ratio = {ratio}"


def test_one_loop_dict_keys(loop_dict):
    assert set(loop_dict.keys()) == {"plin", "p22", "p13", "p1loop"}


def test_P22_scales_as_Plin_squared(k_arr, plin_func):
    """Doubling P_lin should quadruple P22 (P22 ~ Plin^2)."""
    p22_ref = compute_P22(k_arr, plin_func, n_q=32, n_mu=32)
    p22_2x = compute_P22(k_arr, lambda k: 2.0 * plin_func(k), n_q=32, n_mu=32)
    ratio = p22_2x / p22_ref
    assert np.allclose(ratio, 4.0, rtol=1e-6), \
        f"P22 scaling ratio expected 4, got {ratio}"


def test_P13_not_over_normalized(loop_dict, k_arr):
    """|2P13 / Plin| at low k must not be inflated by a factor-of-2 normalisation bug.

    SPT P13 has a UV-sensitive piece proportional to Plin(k)*int q^2 Plin(q) dq
    that is large at low k regardless of k (expected SPT behaviour). With the
    correct 4*pi^2 prefactor the ratio is ~13-15 for this cosmology/grid; the
    former buggy 2*pi^2 prefactor doubles it to ~27-30. A threshold of 20
    distinguishes the two without being sensitive to cosmological parameters.
    """
    k_low_mask = k_arr < 0.05
    if not np.any(k_low_mask):
        pytest.skip("No k < 0.05 in k_arr")
    ratio = np.abs(loop_dict["p13"][k_low_mask] / loop_dict["plin"][k_low_mask])
    assert np.all(ratio < 20.0), (
        f"|2P13/Plin| > 20 at low k — P13 may have a factor-of-2 over-normalisation: ratio = {ratio}"
    )


def test_P22_not_over_normalized(loop_dict, k_arr):
    """P22/Plin at intermediate k must be in a physically reasonable range.

    For a Planck-like cosmology at z=0.5, P22/Plin ~ 0.31–0.37 at k ~ 0.09–0.11 h/Mpc
    (measured with n_q=64, n_mu=64). The bounds are set so that a factor-of-2
    normalization error in either direction would trip this assertion:
      - lower bound = observed/3 ~ 0.10  (flags drastic under-normalization)
      - upper bound = observed*2  ~ 0.75  (flags factor-of-2 over-normalization)
    """
    mask = (k_arr >= 0.08) & (k_arr <= 0.12)
    if not np.any(mask):
        pytest.skip("No k in [0.08, 0.12] in k_arr")
    ratio = loop_dict["p22"][mask] / loop_dict["plin"][mask]
    assert np.all(ratio > 0.10), (
        f"P22/Plin < 0.10 at k~0.09-0.11 — possible under-normalization: ratio = {ratio}"
    )
    assert np.all(ratio < 0.75), (
        f"P22/Plin > 0.75 at k~0.09-0.11 — possible factor-of-2 over-normalization: ratio = {ratio}"
    )


def test_P13_scales_linearly_with_Plin(k_arr, plin_func):
    """P13 ~ Plin * int Plin dq, so 2x Plin -> 4x P13."""
    p13_ref = compute_P13(k_arr, plin_func, n_q=64)
    p13_2x = compute_P13(k_arr, lambda k: 2.0 * plin_func(k), n_q=64)
    ratio = p13_2x / p13_ref
    assert np.allclose(ratio, 4.0, rtol=1e-6), \
        f"P13 scaling ratio expected 4, got {ratio}"
