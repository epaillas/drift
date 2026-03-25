"""Tests for P_gg correlation-matrix binning resolution."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_correlation_matrix_pgg import _resolve_analytic_settings, _resolve_mock_settings


def _args(**overrides):
    base = dict(
        analytic_cov=False,
        diag_cov=False,
        mock_rebin=13,
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


def test_resolve_mock_settings_uses_new_mock_flags():
    cfg = _resolve_mock_settings(_args(mock_rebin=7, mock_kmin=0.02, mock_kmax=0.3), (0, 2, 4))
    assert cfg == {"rebin": 7, "kmin": 0.02, "kmax": 0.3}


def test_resolve_analytic_settings_uses_default_dk(monkeypatch):
    monkeypatch.setattr(
        "scripts.plot_correlation_matrix_pgg.load_observable_measurements",
        lambda path, observable, ells, rebin, kmin=0.0, kmax=np.inf: (
            np.array([0.01, 0.03, 0.05, 0.07]),
            {0: np.ones(4), 2: np.ones(4), 4: np.ones(4)},
        ),
    )

    cfg = _resolve_analytic_settings(_args(analytic_cov=True), (0, 2, 4))

    assert cfg["kmin"] == 0.01
    assert cfg["kmax"] == 0.3
    assert np.isclose(cfg["dk"], 0.02)
    assert np.isclose(cfg["k"][1] - cfg["k"][0], 0.02)
