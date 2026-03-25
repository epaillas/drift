"""Tests for P_qq correlation-matrix covariance selection."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_correlation_matrix_pqq import (
    _resolve_analytic_settings,
    _resolve_mock_settings,
    _resolve_pqq_covariance,
)


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
        "scripts.plot_correlation_matrix_pqq._load_mock_k",
        lambda quantiles, ells, rebin, kmin=0.0, kmax=np.inf: np.array([0.02, 0.04]),
    )
    cfg = _resolve_mock_settings(_args(mock_rebin=7, mock_kmin=0.02, mock_kmax=0.3), (1, 2), (0, 2))
    assert cfg == {"rebin": 7, "kmin": 0.02, "kmax": 0.3}


def test_resolve_analytic_settings_uses_reference_dk(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_pqq._load_mock_k",
        lambda quantiles, ells, rebin, kmin=0.0, kmax=np.inf: np.array([0.02, 0.05, 0.08]),
    )
    cfg = _resolve_analytic_settings(_args(analytic_cov=True), (1, 2), (0, 2))
    assert cfg["kmin"] == 0.02
    assert cfg["kmax"] == 0.3
    assert cfg["dk"] == 0.03


def test_resolve_pqq_covariance_uses_mock_path_for_autos_only(monkeypatch):
    captured = {}

    def fake_mock_covariance(mock_dir, statistic, ells, **kwargs):
        captured["mock_dir"] = mock_dir
        captured["statistic"] = statistic
        captured["ells"] = ells
        captured["kwargs"] = kwargs
        return np.eye(4)

    monkeypatch.setattr("scripts.plot_correlation_matrix_pqq.estimate_mock_covariance", fake_mock_covariance)

    k = np.array([0.05, 0.10])
    flat = np.arange(8.0)
    mask = np.ones(8, dtype=bool)
    quantiles = (1, 3)
    pair_order = (("DS1", "DS1"), ("DS3", "DS3"))
    poles = {
        ("DS1", "DS1"): {0: np.array([1.0, 2.0]), 2: np.array([3.0, 4.0])},
        ("DS3", "DS3"): {0: np.array([5.0, 6.0]), 2: np.array([7.0, 8.0])},
    }
    shot_noise = {("DS1", "DS1"): 1.0, ("DS3", "DS3"): 1.0}
    mock_cfg = {"rebin": 5, "kmin": 0.02, "kmax": 0.3}

    cov, precision = _resolve_pqq_covariance(
        _args(autos_only=True),
        k,
        flat,
        mask,
        quantiles,
        pair_order,
        shot_noise,
        poles,
        mock_cfg,
        (0, 2),
    )

    assert cov.shape == (4, 4)
    assert precision is None
    assert captured["statistic"] == "pqq_auto"
    assert captured["ells"] == (0, 2)
    assert captured["kwargs"]["k_data"] is k
    assert captured["kwargs"]["kmin"] == 0.02
    assert captured["kwargs"]["kmax"] == 0.3
    assert captured["kwargs"].get("return_precision", False) is False


def test_resolve_pqq_covariance_requires_autos_or_analytic_for_default_path():
    k = np.array([0.05, 0.10])
    flat = np.arange(12.0)
    mask = np.ones(12, dtype=bool)
    quantiles = (1, 2)
    pair_order = (("DS1", "DS1"), ("DS1", "DS2"), ("DS2", "DS2"))
    poles = {
        ("DS1", "DS1"): {0: np.array([1.0, 2.0]), 2: np.array([3.0, 4.0])},
        ("DS1", "DS2"): {0: np.array([5.0, 6.0]), 2: np.array([7.0, 8.0])},
        ("DS2", "DS2"): {0: np.array([9.0, 10.0]), 2: np.array([11.0, 12.0])},
    }
    shot_noise = {("DS1", "DS1"): 1.0, ("DS1", "DS2"): 1.0, ("DS2", "DS2"): 1.0}

    with pytest.raises(ValueError, match="Mock P_qq covariance is only available for auto pairs"):
        _resolve_pqq_covariance(
            _args(),
            k,
            flat,
            mask,
            quantiles,
            pair_order,
            shot_noise,
            poles,
            {"rebin": 5, "kmin": 0.02, "kmax": 0.3},
            (0, 2),
        )
