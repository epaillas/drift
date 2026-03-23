"""Tests for DSG covariance selection helpers."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference_dsg import _resolve_dsg_covariance


def _args(**overrides):
    base = dict(
        synthetic=False,
        diag_cov=False,
        analytic_cov=False,
        cov_rescale=64.0,
        box_volume=None,
        galaxy_number_density=None,
        galaxy_shot_noise=None,
        ds_pair_auto_shot_noise=250.0,
        ds_pair_cross_shot_noise=40.0,
        ds_cross_shot_noise=0.0,
        analytic_cov_terms="gaussian",
    )
    base.update(overrides)
    return Namespace(**base)


def _fiducials():
    k = np.array([0.05, 0.10, 0.15])
    return k, {
        "labels": ("DS1", "DS2"),
        "pqg_poles": {
            "DS1": {0: np.array([-600.0, -500.0, -400.0]), 2: np.array([-90.0, -70.0, -50.0])},
            "DS2": {0: np.array([800.0, 650.0, 500.0]), 2: np.array([130.0, 100.0, 75.0])},
        },
        "pqq_poles": {
            ("DS1", "DS1"): {0: np.array([1000.0, 850.0, 700.0]), 2: np.array([120.0, 100.0, 80.0])},
            ("DS1", "DS2"): {0: np.array([-450.0, -360.0, -280.0]), 2: np.array([-55.0, -42.0, -30.0])},
            ("DS2", "DS2"): {0: np.array([900.0, 740.0, 620.0]), 2: np.array([110.0, 90.0, 70.0])},
        },
        "pgg_poles": {
            0: np.array([1400.0, 1150.0, 920.0]),
            2: np.array([420.0, 330.0, 250.0]),
        },
    }


def test_resolve_dsg_covariance_analytic_matches_masked_shape():
    k, fiducials = _fiducials()
    mask = np.array([True, False, True, True, False, True, True, False, True], dtype=bool)
    data = np.arange(mask.sum(), dtype=float) + 1.0

    cov, precision = _resolve_dsg_covariance(
        _args(analytic_cov=True, box_volume=1.0e9, galaxy_shot_noise=300.0),
        k,
        data,
        mask,
        (1, 2),
        fiducial_blocks=fiducials,
    )

    assert cov.shape == (mask.sum(), mask.sum())
    assert precision.shape == cov.shape


def test_resolve_dsg_covariance_rejects_beyond_gaussian_terms():
    k, fiducials = _fiducials()
    mask = np.ones(12, dtype=bool)
    data = np.arange(mask.sum(), dtype=float) + 1.0

    try:
        _resolve_dsg_covariance(
            _args(
                analytic_cov=True,
                box_volume=1.0e9,
                galaxy_shot_noise=300.0,
                analytic_cov_terms="gaussian+effective_cng",
            ),
            k,
            data,
            mask,
            (1, 2),
            fiducial_blocks=fiducials,
        )
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError for beyond-Gaussian DSG covariance.")


def test_resolve_dsg_covariance_prefers_diagonal_for_synthetic_without_analytic():
    k = np.array([0.05, 0.10])
    mask = np.array([True, True, True, True], dtype=bool)
    data = np.array([1.0, 2.0, 0.5, 0.4])

    cov, _ = _resolve_dsg_covariance(
        _args(synthetic=True),
        k,
        data,
        mask,
        (1,),
    )

    assert np.allclose(cov, np.diag(np.diag(cov)))
