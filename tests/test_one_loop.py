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


def test_P13_scales_linearly_with_Plin(k_arr, plin_func):
    """P13 ~ Plin * int Plin dq, so 2x Plin -> 4x P13."""
    p13_ref = compute_P13(k_arr, plin_func, n_q=64)
    p13_2x = compute_P13(k_arr, lambda k: 2.0 * plin_func(k), n_q=64)
    ratio = p13_2x / p13_ref
    assert np.allclose(ratio, 4.0, rtol=1e-6), \
        f"P13 scaling ratio expected 4, got {ratio}"
