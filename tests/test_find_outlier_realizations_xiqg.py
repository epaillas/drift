"""Tests for xi_qg catastrophic outlier finding."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.find_outlier_realizations_xiqg import (
    find_outlier_realizations,
    write_outlier_csv,
)


def _synthetic_realizations():
    s = np.array([10.0, 20.0, 30.0])
    phases = ("ph0001", "ph0002", "ph0003", "ph0004")
    realizations = {
        "DS1": {
            0: np.array(
                [
                    [1.00, 1.05, 0.95],
                    [1.02, 1.00, 0.98],
                    [0.98, 1.01, 1.00],
                    [1.01, 0.99, 1.02],
                ]
            ),
            2: np.array(
                [
                    [0.20, 0.18, 0.22],
                    [0.21, 0.19, 0.20],
                    [0.19, 0.21, 0.18],
                    [0.20, 0.20, 0.21],
                ]
            ),
        },
        "DS3": {
            0: np.array(
                [
                    [0.60, 0.62, 0.61],
                    [0.61, 0.59, 0.60],
                    [0.59, 0.60, 0.58],
                    [0.60, 0.61, 0.59],
                ]
            ),
            2: np.array(
                [
                    [0.10, 0.11, 0.10],
                    [0.11, 0.10, 0.09],
                    [0.09, 0.10, 0.11],
                    [0.10, 9.00, 0.10],
                ]
            ),
        },
    }
    return s, realizations, phases


def test_find_outlier_realizations_flags_single_catastrophic_phase():
    s, realizations, phases = _synthetic_realizations()

    results = find_outlier_realizations(
        s,
        realizations,
        phases,
        quantiles=(1, 3),
        ells=(0, 2),
        threshold=12.0,
    )

    assert results[0]["phase"] == "ph0004"
    assert results[0]["flagged"] is True
    assert results[0]["worst_quantile"] == "DS3"
    assert results[0]["worst_ell"] == 2
    assert np.isclose(results[0]["worst_s"], 20.0)
    assert results[0]["max_panel_score"] > 12.0
    assert [result["phase"] for result in results if result["flagged"]] == ["ph0004"]


def test_find_outlier_realizations_does_not_flag_ordinary_scatter():
    s = np.array([10.0, 20.0, 30.0])
    phases = ("ph0001", "ph0002", "ph0003", "ph0004")
    realizations = {
        "DS1": {
            0: np.array(
                [
                    [1.00, 1.02, 0.99],
                    [1.03, 0.98, 1.01],
                    [0.97, 1.01, 1.02],
                    [1.01, 0.99, 1.00],
                ]
            ),
            2: np.array(
                [
                    [0.30, 0.31, 0.29],
                    [0.29, 0.30, 0.31],
                    [0.31, 0.29, 0.30],
                    [0.30, 0.30, 0.30],
                ]
            ),
        },
    }

    results = find_outlier_realizations(
        s,
        realizations,
        phases,
        quantiles=(1,),
        ells=(0, 2),
        threshold=12.0,
    )

    assert all(result["flagged"] is False for result in results)
    assert max(result["max_panel_score"] for result in results) < 12.0


def test_write_outlier_csv_sorts_and_filters_flagged_rows(tmp_path):
    rows = [
        {
            "phase": "ph0003",
            "flagged": False,
            "max_panel_score": 5.0,
            "worst_quantile": "DS1",
            "worst_ell": 0,
            "worst_s": 30.0,
            "worst_value": 1.0,
            "worst_median": 0.8,
            "worst_residual": 0.2,
        },
        {
            "phase": "ph0002",
            "flagged": True,
            "max_panel_score": 18.0,
            "worst_quantile": "DS3",
            "worst_ell": 2,
            "worst_s": 20.0,
            "worst_value": 3600.0,
            "worst_median": 40.0,
            "worst_residual": 3560.0,
        },
    ]
    path = tmp_path / "outliers.csv"

    write_outlier_csv(path, rows, flagged_only=True)

    with path.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))

    assert len(loaded) == 1
    assert loaded[0]["phase"] == "ph0002"
    assert loaded[0]["flagged"] == "True"
    assert loaded[0]["worst_quantile"] == "DS3"
