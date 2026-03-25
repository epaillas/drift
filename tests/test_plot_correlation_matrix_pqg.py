"""Tests for P_qg correlation-matrix covariance selection."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_correlation_matrix_pqg import (
    _resolve_analytic_settings,
    _resolve_mock_settings,
    _resolve_pqg_covariance,
)


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
        mock_rebin=5,
        mock_kmin=None,
        mock_kmax=None,
        analytic_kmin=None,
        analytic_kmax=None,
        analytic_dk=None,
        rebin=None,
        kmin=None,
        kmax=None,
        nk=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_resolve_mock_settings_prefers_new_mock_flags(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_pqg._load_mock_k",
        lambda quantiles, ells, rebin, kmin=0.0, kmax=np.inf: np.array([0.02, 0.04]),
    )
    cfg = _resolve_mock_settings(_args(mock_rebin=7, mock_kmin=0.02, mock_kmax=0.3), (1, 5), (0, 2))
    assert cfg == {"rebin": 7, "kmin": 0.02, "kmax": 0.3}


def test_resolve_analytic_settings_uses_reference_dk(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_pqg._load_mock_k",
        lambda quantiles, ells, rebin, kmin=0.0, kmax=np.inf: np.array([0.02, 0.05, 0.08]),
    )
    cfg = _resolve_analytic_settings(_args(analytic_cov=True), (1, 5), (0, 2))
    assert cfg["kmin"] == 0.02
    assert cfg["kmax"] == 0.3
    assert cfg["dk"] == 0.03


def test_resolve_pqg_covariance_uses_mock_path_by_default(monkeypatch):
    captured = {}

    def fake_mock_covariance(mock_dir, statistic, ells, **kwargs):
        captured["mock_dir"] = mock_dir
        captured["statistic"] = statistic
        captured["ells"] = ells
        captured["kwargs"] = kwargs
        return np.eye(4)

    monkeypatch.setattr("scripts.plot_correlation_matrix_pqg.estimate_mock_covariance", fake_mock_covariance)

    k = np.array([0.05, 0.10])
    flat = np.zeros(8)
    mask = np.ones(8, dtype=bool)
    quantiles = (1, 3)
    labels = ("DS1", "DS3")
    mock_cfg = {"rebin": 5, "kmin": 0.02, "kmax": 0.3}

    cov, precision = _resolve_pqg_covariance(
        _args(),
        k,
        flat,
        mask,
        quantiles,
        labels,
        fiducials=None,
        mock_cfg=mock_cfg,
        ells=(0, 2),
    )

    assert cov.shape == (4, 4)
    assert precision is None
    assert captured["statistic"] == "pqg"
    assert captured["ells"] == (0, 2)
    assert captured["kwargs"]["k_data"] is k
    assert captured["kwargs"]["quantiles"] == quantiles
    assert captured["kwargs"]["kmin"] == 0.02
    assert captured["kwargs"]["kmax"] == 0.3
    assert captured["kwargs"].get("return_precision", False) is False
