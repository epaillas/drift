"""Tests for analytic density-split pair covariance."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import analytic_pqq_covariance


PAIR_ORDER = (("DS1", "DS1"), ("DS1", "DS2"), ("DS2", "DS2"))
ELLS = (0, 2)


def _fiducial_pair_poles(k):
    return {
        ("DS1", "DS1"): {
            0: 1200.0 / (1.0 + 8.0 * k),
            2: 150.0 / (1.0 + 6.0 * k),
        },
        ("DS1", "DS2"): {
            0: -600.0 / (1.0 + 7.0 * k),
            2: -80.0 / (1.0 + 5.0 * k),
        },
        ("DS2", "DS2"): {
            0: 900.0 / (1.0 + 9.0 * k),
            2: 120.0 / (1.0 + 4.0 * k),
        },
    }


def _pair_shot_noise():
    return {
        ("DS1", "DS1"): 250.0,
        ("DS1", "DS2"): 40.0,
        ("DS2", "DS2"): 220.0,
    }


def test_analytic_pqq_covariance_shapes_and_symmetry():
    k = np.array([0.05, 0.10, 0.15])
    cov, precision = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=1.0e9,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
    )

    expected = len(PAIR_ORDER) * len(ELLS) * len(k)
    assert cov.shape == (expected, expected)
    assert precision.shape == (expected, expected)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.diag(cov) > 0.0)


def test_analytic_pqq_covariance_is_block_diagonal_in_k():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
    )
    nk = len(k)
    first_block = cov[:nk, :nk]
    cross_block = cov[:nk, nk:2 * nk]
    assert np.allclose(first_block - np.diag(np.diag(first_block)), 0.0)
    assert np.allclose(cross_block - np.diag(np.diag(cross_block)), 0.0)
    assert np.any(np.abs(np.diag(cross_block)) > 0.0)


def test_analytic_pqq_covariance_has_cross_pair_coupling():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
    )
    nk = len(k)
    block_size = len(ELLS) * nk
    pair00_to_pair01 = cov[:block_size, block_size:2 * block_size]
    assert np.any(np.abs(np.diag(pair00_to_pair01)) > 0.0)


def test_analytic_pqq_covariance_accepts_reversed_pair_keys():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_pair_poles(k)
    poles[("DS2", "DS1")] = poles.pop(("DS1", "DS2"))
    shot = _pair_shot_noise()
    shot[("DS2", "DS1")] = shot.pop(("DS1", "DS2"))

    cov_a, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=8.0e8, pair_order=PAIR_ORDER, shot_noise=shot
    )
    cov_b, _ = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
    )
    np.testing.assert_allclose(cov_a, cov_b)


def test_analytic_pqq_covariance_mask_and_rescale_match_expectation():
    k = np.array([0.05, 0.10, 0.15])
    full_cov, _ = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=6.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
    )
    mask = np.ones(full_cov.shape[0], dtype=bool)
    mask[1::4] = False
    masked_cov, _ = analytic_pqq_covariance(
        k,
        _fiducial_pair_poles(k),
        ells=ELLS,
        volume=6.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
        mask=mask,
        rescale=5.0,
    )
    np.testing.assert_allclose(masked_cov, full_cov[np.ix_(mask, mask)] / 5.0)


def test_analytic_pqq_covariance_scales_with_volume_and_shot_noise():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_pair_poles(k)
    shot_lo = _pair_shot_noise()
    shot_hi = {pair: value * 2.0 for pair, value in shot_lo.items()}
    cov_small_vol, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=5.0e8, pair_order=PAIR_ORDER, shot_noise=shot_lo
    )
    cov_large_vol, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=1.0e9, pair_order=PAIR_ORDER, shot_noise=shot_lo
    )
    cov_low_noise, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=5.0e8, pair_order=PAIR_ORDER, shot_noise=shot_lo
    )
    cov_high_noise, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=5.0e8, pair_order=PAIR_ORDER, shot_noise=shot_hi
    )

    assert np.all(np.diag(cov_large_vol) < np.diag(cov_small_vol))
    assert np.all(np.diag(cov_high_noise) > np.diag(cov_low_noise))


def test_analytic_pqq_effective_cng_adds_off_diagonal_mode_coupling():
    k = np.array([0.05, 0.10, 0.15, 0.20])
    poles = _fiducial_pair_poles(k)
    cov_gauss, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=8.0e8, pair_order=PAIR_ORDER, shot_noise=_pair_shot_noise(),
    )
    cov_cng, _ = analytic_pqq_covariance(
        k,
        poles,
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
        terms="gaussian+effective_cng",
        cng_amplitude=0.25,
        cng_coherence=0.30,
    )
    nk = len(k)
    block_00 = cov_cng[:nk, :nk]
    block_02 = cov_cng[:nk, nk:2 * nk]
    assert np.any(np.abs(block_00 - np.diag(np.diag(block_00))) > 0.0)
    assert np.any(np.abs(block_02 - np.diag(np.diag(block_02))) > 0.0)
    assert np.all(np.diag(cov_cng) > np.diag(cov_gauss))


def test_analytic_pqq_effective_cng_zero_amplitude_matches_gaussian():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_pair_poles(k)
    cov_gauss, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=8.0e8, pair_order=PAIR_ORDER, shot_noise=_pair_shot_noise(),
    )
    cov_zero, _ = analytic_pqq_covariance(
        k,
        poles,
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
        terms="gaussian+effective_cng",
        cng_amplitude=0.0,
    )
    np.testing.assert_allclose(cov_zero, cov_gauss)


def test_analytic_pqq_ssc_adds_off_diagonal_mode_coupling():
    k = np.array([0.05, 0.10, 0.15, 0.20])
    poles = _fiducial_pair_poles(k)
    cov_gauss, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=8.0e8, pair_order=PAIR_ORDER, shot_noise=_pair_shot_noise(),
    )
    cov_ssc, _ = analytic_pqq_covariance(
        k,
        poles,
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
        terms="gaussian+ssc",
        ssc_sigma_b2=1.0e-4,
    )
    nk = len(k)
    block_00 = cov_ssc[:nk, :nk]
    block_02 = cov_ssc[:nk, nk:2 * nk]
    assert np.any(np.abs(block_00 - np.diag(np.diag(block_00))) > 0.0)
    assert np.any(np.abs(block_02 - np.diag(np.diag(block_02))) > 0.0)
    assert np.any(np.abs(cov_ssc - cov_gauss) > 0.0)


def test_analytic_pqq_ssc_zero_variance_matches_gaussian():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_pair_poles(k)
    cov_gauss, _ = analytic_pqq_covariance(
        k, poles, ells=ELLS, volume=8.0e8, pair_order=PAIR_ORDER, shot_noise=_pair_shot_noise(),
    )
    cov_zero, _ = analytic_pqq_covariance(
        k,
        poles,
        ells=ELLS,
        volume=8.0e8,
        pair_order=PAIR_ORDER,
        shot_noise=_pair_shot_noise(),
        terms="gaussian+ssc",
        ssc_sigma_b2=0.0,
    )
    np.testing.assert_allclose(cov_zero, cov_gauss)


def test_analytic_pqq_covariance_validates_missing_pairs():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_pair_poles(k)
    poles.pop(("DS1", "DS2"))
    try:
        analytic_pqq_covariance(
            k,
            poles,
            ells=ELLS,
            volume=8.0e8,
            pair_order=PAIR_ORDER,
            shot_noise=_pair_shot_noise(),
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for missing DS-pair fiducial poles.")
