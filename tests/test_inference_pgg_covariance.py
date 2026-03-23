"""Tests for P_gg covariance selection helpers."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference_pgg import _resolve_pgg_covariance


def _args(**overrides):
    base = dict(
        synthetic=False,
        diag_cov=False,
        analytic_cov=False,
        cov_rescale=64.0,
        box_volume=None,
        number_density=None,
        shot_noise=None,
        rebin=13,
        analytic_cov_terms="gaussian",
        cng_amplitude=0.0,
        cng_coherence=0.35,
    )
    base.update(overrides)
    return Namespace(**base)


def test_resolve_pgg_covariance_analytic_matches_masked_shape():
    k = np.array([0.05, 0.10, 0.15])
    poles = {
        0: np.array([1200.0, 900.0, 700.0]),
        2: np.array([400.0, 300.0, 200.0]),
        4: np.array([80.0, 60.0, 40.0]),
    }
    mask = np.array([True, False, True, True, False, True, True, False, True], dtype=bool)
    data = np.concatenate([poles[ell] for ell in (0, 2, 4)])[mask]

    cov, precision = _resolve_pgg_covariance(
        _args(analytic_cov=True, box_volume=1.0e9, shot_noise=200.0),
        k,
        data,
        mask,
        fiducial_poles=poles,
    )

    assert cov.shape == (mask.sum(), mask.sum())
    assert precision.shape == cov.shape


def test_resolve_pgg_covariance_forwards_effective_cng_terms():
    k = np.array([0.05, 0.10, 0.15])
    poles = {
        0: np.array([1200.0, 900.0, 700.0]),
        2: np.array([400.0, 300.0, 200.0]),
        4: np.array([80.0, 60.0, 40.0]),
    }
    mask = np.ones(9, dtype=bool)
    data = np.concatenate([poles[ell] for ell in (0, 2, 4)])

    cov_gauss, _ = _resolve_pgg_covariance(
        _args(analytic_cov=True, box_volume=1.0e9, shot_noise=200.0),
        k,
        data,
        mask,
        fiducial_poles=poles,
    )
    cov_cng, _ = _resolve_pgg_covariance(
        _args(
            analytic_cov=True,
            box_volume=1.0e9,
            shot_noise=200.0,
            analytic_cov_terms="gaussian+effective_cng",
            cng_amplitude=0.2,
            cng_coherence=0.3,
        ),
        k,
        data,
        mask,
        fiducial_poles=poles,
    )

    assert np.any(np.abs(cov_cng - cov_gauss) > 0.0)
    assert np.any(np.abs(cov_cng[:3, :3] - np.diag(np.diag(cov_cng[:3, :3]))) > 0.0)


def test_resolve_pgg_covariance_prefers_diagonal_for_synthetic_without_analytic():
    k = np.array([0.05, 0.10])
    mask = np.array([True, True, True, True, True, True], dtype=bool)
    data = np.array([1.0, 2.0, 0.5, 0.4, 0.1, 0.1])

    cov, _ = _resolve_pgg_covariance(
        _args(synthetic=True),
        k,
        data,
        mask,
    )

    assert np.allclose(cov, np.diag(np.diag(cov)))
