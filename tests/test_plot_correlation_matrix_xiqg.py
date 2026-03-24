"""Tests for xi_qg correlation-matrix covariance selection."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_correlation_matrix_xiqg import _resolve_xiqg_covariance


def _args(**overrides):
    base = dict(
        diag_cov=False,
        analytic_cov=False,
        cov_rescale=64.0,
        box_volume=1.0e9,
        galaxy_shot_noise=400.0,
        ds_pair_auto_shot_noise=250.0,
        ds_pair_cross_shot_noise=40.0,
        ds_cross_shot_noise=0.0,
        analytic_cov_terms="gaussian",
        cng_amplitude=0.0,
        cng_coherence=0.35,
        ssc_sigma_b2=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_resolve_xiqg_covariance_propagates_mock_covariance(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_xiqg.mock_covariance_matrix",
        lambda *args, **kwargs: np.eye(8),
    )

    k = np.array([0.05, 0.10])
    s = np.array([20.0, 40.0])
    flat = np.zeros(8)
    mask = np.ones(8, dtype=bool)
    quantiles = (1, 3)
    labels = ("DS1", "DS3")
    cov, precision = _resolve_xiqg_covariance(
        _args(),
        k,
        s,
        flat,
        mask,
        quantiles,
        labels,
        fiducials=None,
        mock_cfg={"rebin": 5, "kmin": 0.02, "kmax": 0.3},
    )

    assert cov.shape == (8, 8)
    assert precision is None
