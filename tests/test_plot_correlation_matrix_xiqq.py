"""Tests for xi_qq correlation-matrix covariance selection."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_correlation_matrix_xiqq import _resolve_xiqq_covariance


def _args(**overrides):
    base = dict(
        diag_cov=False,
        analytic_cov=False,
        autos_only=False,
        cov_rescale=64.0,
        box_volume=1.0e9,
        analytic_cov_terms="gaussian",
        cng_amplitude=0.0,
        cng_coherence=0.35,
        ssc_sigma_b2=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_resolve_xiqq_covariance_propagates_mock_covariance(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_xiqq.estimate_mock_covariance",
        lambda *args, **kwargs: np.eye(8),
    )

    k = np.array([0.05, 0.10])
    s = np.array([20.0, 40.0])
    flat = np.zeros(8)
    mask = np.ones(8, dtype=bool)
    quantiles = (1, 3)
    pair_order = (("DS1", "DS1"), ("DS3", "DS3"))
    cov, precision = _resolve_xiqq_covariance(
        _args(autos_only=True),
        k,
        s,
        flat,
        mask,
        quantiles,
        pair_order,
        shot_noise={("DS1", "DS1"): 1.0, ("DS3", "DS3"): 1.0},
        fiducial_poles={},
        mock_cfg={"rebin": 5, "kmin": 0.02, "kmax": 0.3},
        ells=(0, 2),
    )

    assert cov.shape == (8, 8)
    assert precision is None


def test_resolve_xiqq_covariance_requires_autos_for_mock_path():
    k = np.array([0.05, 0.10])
    s = np.array([20.0, 40.0])
    flat = np.zeros(12)
    mask = np.ones(12, dtype=bool)
    quantiles = (1, 2)
    pair_order = (("DS1", "DS1"), ("DS1", "DS2"), ("DS2", "DS2"))

    with pytest.raises(ValueError, match="Mock xi_qq covariance is only available for auto pairs"):
        _resolve_xiqq_covariance(
            _args(),
            k,
            s,
            flat,
            mask,
            quantiles,
            pair_order,
            shot_noise={("DS1", "DS1"): 1.0, ("DS1", "DS2"): 1.0, ("DS2", "DS2"): 1.0},
            fiducial_poles={},
            mock_cfg={"rebin": 5, "kmin": 0.02, "kmax": 0.3},
            ells=(0, 2),
        )
