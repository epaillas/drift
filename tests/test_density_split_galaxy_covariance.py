"""Tests for analytic density-split-galaxy covariance."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import analytic_pqg_covariance


DS_LABELS = ("DS1", "DS2")
ELLS = (0, 2)


def _fiducial_pqg_poles(k):
    return {
        "DS1": {
            0: -700.0 / (1.0 + 8.0 * k),
            2: -120.0 / (1.0 + 5.0 * k),
        },
        "DS2": {
            0: 900.0 / (1.0 + 7.0 * k),
            2: 160.0 / (1.0 + 4.0 * k),
        },
    }


def _fiducial_pqq_poles(k):
    return {
        ("DS1", "DS1"): {
            0: 1100.0 / (1.0 + 8.0 * k),
            2: 140.0 / (1.0 + 5.0 * k),
        },
        ("DS1", "DS2"): {
            0: -500.0 / (1.0 + 7.0 * k),
            2: -60.0 / (1.0 + 4.0 * k),
        },
        ("DS2", "DS2"): {
            0: 850.0 / (1.0 + 9.0 * k),
            2: 120.0 / (1.0 + 6.0 * k),
        },
    }


def _fiducial_pgg_poles(k):
    return {
        0: 1500.0 / (1.0 + 10.0 * k),
        2: 500.0 / (1.0 + 8.0 * k),
    }


def _pair_shot_noise():
    return {
        ("DS1", "DS1"): 250.0,
        ("DS1", "DS2"): 40.0,
        ("DS2", "DS2"): 220.0,
    }


def _cross_shot_noise():
    return {
        "DS1": 0.0,
        "DS2": 0.0,
    }


def test_analytic_pqg_covariance_shapes_and_symmetry():
    k = np.array([0.05, 0.10, 0.15])
    cov, precision = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=1.0e9,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=350.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )

    expected = len(DS_LABELS) * len(ELLS) * len(k)
    assert cov.shape == (expected, expected)
    assert precision.shape == cov.shape
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.diag(cov) > 0.0)


def test_analytic_pqg_covariance_is_block_diagonal_in_k():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=8.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=300.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    nk = len(k)
    first_block = cov[:nk, :nk]
    cross_ell_block = cov[:nk, nk:2 * nk]
    assert np.allclose(first_block - np.diag(np.diag(first_block)), 0.0)
    assert np.allclose(cross_ell_block - np.diag(np.diag(cross_ell_block)), 0.0)
    assert np.any(np.abs(np.diag(cross_ell_block)) > 0.0)


def test_analytic_pqg_covariance_has_cross_quantile_coupling():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=8.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=300.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    nk = len(k)
    block_size = len(ELLS) * nk
    q1_to_q2 = cov[:block_size, block_size:2 * block_size]
    assert np.any(np.abs(np.diag(q1_to_q2)) > 0.0)


def test_analytic_pqg_covariance_mask_and_rescale_match_expectation():
    k = np.array([0.05, 0.10, 0.15])
    full_cov, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=6.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=250.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    mask = np.ones(full_cov.shape[0], dtype=bool)
    mask[1::3] = False
    masked_cov, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=6.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=250.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
        mask=mask,
        rescale=5.0,
    )
    np.testing.assert_allclose(masked_cov, full_cov[np.ix_(mask, mask)] / 5.0)


def test_analytic_pqg_covariance_scales_with_volume_and_noise():
    k = np.array([0.05, 0.10, 0.15])
    cov_small_vol, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=5.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=150.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    cov_large_vol, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=1.0e9,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=150.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    cov_low_noise, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=5.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=100.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )
    cov_high_noise, _ = analytic_pqg_covariance(
        k,
        _fiducial_pqg_poles(k),
        _fiducial_pqq_poles(k),
        _fiducial_pgg_poles(k),
        ells=ELLS,
        volume=5.0e8,
        ds_labels=DS_LABELS,
        galaxy_shot_noise=500.0,
        ds_pair_shot_noise=_pair_shot_noise(),
        ds_cross_shot_noise=_cross_shot_noise(),
    )

    assert np.all(np.diag(cov_large_vol) < np.diag(cov_small_vol))
    assert np.all(np.diag(cov_high_noise) > np.diag(cov_low_noise))


def test_analytic_pqg_covariance_rejects_non_gaussian_terms():
    k = np.array([0.05, 0.10, 0.15])
    try:
        analytic_pqg_covariance(
            k,
            _fiducial_pqg_poles(k),
            _fiducial_pqq_poles(k),
            _fiducial_pgg_poles(k),
            ells=ELLS,
            volume=8.0e8,
            ds_labels=DS_LABELS,
            galaxy_shot_noise=200.0,
            ds_pair_shot_noise=_pair_shot_noise(),
            ds_cross_shot_noise=_cross_shot_noise(),
            terms="gaussian+effective_cng",
        )
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError for beyond-Gaussian DSG covariance.")
