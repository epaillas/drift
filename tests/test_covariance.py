"""Tests for analytic P_gg covariance and plotting helpers."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import analytic_pgg_covariance, correlation_matrix, plot_correlation_matrix


def _fiducial_poles(k):
    return {
        0: 1500.0 / (1.0 + 10.0 * k),
        2: 600.0 / (1.0 + 8.0 * k),
        4: 150.0 / (1.0 + 6.0 * k),
    }


def test_analytic_covariance_shapes_and_symmetry():
    k = np.array([0.05, 0.10, 0.15])
    cov, precision = analytic_pgg_covariance(
        k,
        _fiducial_poles(k),
        ells=(0, 2, 4),
        volume=1.0e9,
        shot_noise=500.0,
    )

    assert cov.shape == (9, 9)
    assert precision.shape == (9, 9)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.diag(cov) > 0.0)


def test_analytic_covariance_is_block_diagonal_in_k():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pgg_covariance(
        k,
        _fiducial_poles(k),
        ells=(0, 2),
        volume=8.0e8,
        shot_noise=300.0,
    )
    nk = len(k)
    block_00 = cov[:nk, :nk]
    block_02 = cov[:nk, nk:]
    assert np.allclose(block_00 - np.diag(np.diag(block_00)), 0.0)
    assert np.allclose(block_02 - np.diag(np.diag(block_02)), 0.0)
    assert np.any(np.abs(np.diag(block_02)) > 0.0)


def test_effective_cng_adds_off_diagonal_mode_coupling():
    k = np.array([0.05, 0.10, 0.15, 0.20])
    poles = _fiducial_poles(k)
    cov_gauss, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=8.0e8, shot_noise=300.0,
    )
    cov_cng, _ = analytic_pgg_covariance(
        k,
        poles,
        ells=(0, 2),
        volume=8.0e8,
        shot_noise=300.0,
        terms="gaussian+effective_cng",
        cng_amplitude=0.25,
        cng_coherence=0.30,
    )
    nk = len(k)
    block_00 = cov_cng[:nk, :nk]
    block_02 = cov_cng[:nk, nk:]
    assert np.any(np.abs(block_00 - np.diag(np.diag(block_00))) > 0.0)
    assert np.any(np.abs(block_02 - np.diag(np.diag(block_02))) > 0.0)
    assert np.all(np.diag(cov_cng) > np.diag(cov_gauss))


def test_effective_cng_zero_amplitude_matches_gaussian():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_poles(k)
    cov_gauss, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=8.0e8, shot_noise=300.0,
    )
    cov_zero, _ = analytic_pgg_covariance(
        k,
        poles,
        ells=(0, 2),
        volume=8.0e8,
        shot_noise=300.0,
        terms="gaussian+effective_cng",
        cng_amplitude=0.0,
    )
    np.testing.assert_allclose(cov_zero, cov_gauss)


def test_analytic_covariance_scales_with_volume_and_shot_noise():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_poles(k)
    cov_small_vol, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=5.0e8, shot_noise=300.0,
    )
    cov_large_vol, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=1.0e9, shot_noise=300.0,
    )
    cov_low_noise, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=5.0e8, shot_noise=100.0,
    )
    cov_high_noise, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=5.0e8, shot_noise=500.0,
    )

    assert np.all(np.diag(cov_large_vol) < np.diag(cov_small_vol))
    assert np.all(np.diag(cov_high_noise) > np.diag(cov_low_noise))


def test_analytic_covariance_mask_and_rescale_match_expectation():
    k = np.array([0.05, 0.10, 0.15])
    poles = _fiducial_poles(k)
    mask = np.array([True, False, True, True, False, True], dtype=bool)
    full_cov, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=6.0e8, shot_noise=200.0,
    )
    masked_cov, _ = analytic_pgg_covariance(
        k, poles, ells=(0, 2), volume=6.0e8, shot_noise=200.0, mask=mask, rescale=4.0,
    )

    np.testing.assert_allclose(masked_cov, full_cov[np.ix_(mask, mask)] / 4.0)


def test_correlation_plot_helper_returns_axes():
    k = np.array([0.05, 0.10, 0.15])
    cov, _ = analytic_pgg_covariance(
        k, _fiducial_poles(k), ells=(0, 2), volume=7.0e8, shot_noise=250.0,
    )
    corr = correlation_matrix(cov)
    np.testing.assert_allclose(np.diag(corr), 1.0)

    fig, ax = plot_correlation_matrix(cov, k=k, ells=(0, 2), title="test")
    assert ax.get_title() == "test"
    assert len(fig.axes) >= 1
