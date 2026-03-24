"""Tests for xi_gg correlation-matrix helpers."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_correlation_matrix_xi_common import (
    apply_reciprocal_analytic_k_limits,
    build_s_grid,
    implied_k_limits,
)
from scripts.plot_correlation_matrix_xigg import _resolve_xigg_covariance
from scripts.plot_correlation_matrix_pgg import _resolve_analytic_settings


def _args(**overrides):
    base = dict(
        diag_cov=False,
        analytic_cov=False,
        cov_rescale=64.0,
        box_volume=1.0e9,
        number_density=None,
        shot_noise=250.0,
        analytic_cov_terms="gaussian",
        cng_amplitude=0.0,
        cng_coherence=0.35,
        ssc_sigma_b2=None,
        analytic_kmin=None,
        analytic_kmax=None,
        analytic_dk=None,
        smin=None,
        smax=None,
        rebin=None,
        kmin=None,
        kmax=None,
        nk=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_build_s_grid_defaults_to_reliable_window():
    k = np.array([0.02, 0.05, 0.10])
    s = build_s_grid(k)
    assert np.isclose(s[0], 30.0)
    assert np.isclose(s[-1], 35.0)


def test_implied_k_limits_match_reciprocal_heuristic():
    kmin, kmax = implied_k_limits(smin=20.0, smax=200.0)
    assert np.isclose(kmin, 0.0035)
    assert np.isclose(kmax, 0.15)


def test_apply_reciprocal_analytic_k_limits_only_fills_missing_values():
    args = _args(analytic_kmin=None, analytic_kmax=None, smin=20.0, smax=200.0)
    analytic_kmin, analytic_kmax = apply_reciprocal_analytic_k_limits(args)
    assert np.isclose(analytic_kmin, 0.0035)
    assert np.isclose(analytic_kmax, 0.15)

    args_override = _args(analytic_kmin=0.01, analytic_kmax=0.2, smin=20.0, smax=200.0)
    analytic_kmin, analytic_kmax = apply_reciprocal_analytic_k_limits(args_override)
    assert np.isclose(analytic_kmin, 0.01)
    assert np.isclose(analytic_kmax, 0.2)


def test_resolve_analytic_settings_can_use_implied_k_limits(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_pgg.load_pgg_measurements",
        lambda path, ells, rebin, kmin=0.0, kmax=np.inf: (
            np.array([0.01, 0.03, 0.05, 0.07]),
            {0: np.ones(4), 2: np.ones(4), 4: np.ones(4)},
        ),
    )
    args = _args(analytic_cov=True, analytic_kmin=None, analytic_kmax=None, analytic_dk=None, smin=20.0, smax=200.0)
    args.analytic_kmin, args.analytic_kmax = apply_reciprocal_analytic_k_limits(args)
    cfg = _resolve_analytic_settings(args)
    assert np.isclose(cfg["kmin"], 0.0035)
    assert np.isclose(cfg["kmax"], 0.15)


def test_resolve_xigg_covariance_propagates_mock_covariance(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_xigg.mock_covariance_matrix",
        lambda *args, **kwargs: np.eye(6),
    )

    k = np.array([0.05, 0.10])
    s = np.array([20.0, 40.0])
    flat = np.zeros(6)
    mask = np.ones(6, dtype=bool)
    cov, precision = _resolve_xigg_covariance(
        _args(),
        k,
        s,
        flat,
        mask,
        poles=None,
        mock_cfg={"rebin": 13, "kmin": 0.02, "kmax": 0.3},
    )

    assert cov.shape == (6, 6)
    assert precision is None
