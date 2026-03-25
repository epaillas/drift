"""Tests for DS auto-pair mock covariance loading."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import _load_pqq_auto_mocks


def test_load_pqq_auto_mocks_reads_expected_pattern_and_order(tmp_path, monkeypatch):
    for name in ("dsc_pkqq_poles_ph3001.h5", "dsc_pkqq_poles_ph3000.h5", "ignore_me.h5"):
        (tmp_path / name).touch()

    calls = []

    def fake_load_observable_measurements(path, observable, nquantiles, ells, rebin, kmin=0.0, kmax=np.inf):
        assert observable == "pqg"
        calls.append(Path(path).name)
        k = np.array([0.05, 0.1, 0.2, 0.3])
        k = k[(k >= kmin) & (k <= kmax)]
        phase = int(Path(path).stem.split("ph")[-1])
        meas = {
            "DS1": {
                0: np.arange(len(k), dtype=float) + phase + 1.0,
                2: np.arange(len(k), dtype=float) + phase + 3.0,
            },
            "DS3": {
                0: np.arange(len(k), dtype=float) + phase + 5.0,
                2: np.arange(len(k), dtype=float) + phase + 7.0,
            },
        }
        return k, meas

    monkeypatch.setattr("drift.io.load_observable_measurements", fake_load_observable_measurements)

    k, mock_matrix = _load_pqq_auto_mocks(
        tmp_path,
        nquantiles=3,
        quantiles=(1, 3),
        ells=(0, 2),
        rebin=5,
    )

    np.testing.assert_allclose(k, [0.05, 0.1, 0.2, 0.3])
    assert calls == ["dsc_pkqq_poles_ph3000.h5", "dsc_pkqq_poles_ph3001.h5"]
    np.testing.assert_allclose(
        mock_matrix,
        [
            [3001.0, 3002.0, 3003.0, 3004.0, 3003.0, 3004.0, 3005.0, 3006.0, 3005.0, 3006.0, 3007.0, 3008.0, 3007.0, 3008.0, 3009.0, 3010.0],
            [3002.0, 3003.0, 3004.0, 3005.0, 3004.0, 3005.0, 3006.0, 3007.0, 3006.0, 3007.0, 3008.0, 3009.0, 3008.0, 3009.0, 3010.0, 3011.0],
        ],
    )


def test_load_pqq_auto_mocks_applies_k_cuts(tmp_path, monkeypatch):
    (tmp_path / "dsc_pkqq_poles_ph3000.h5").touch()

    def fake_load_observable_measurements(path, observable, nquantiles, ells, rebin, kmin=0.0, kmax=np.inf):
        assert observable == "pqg"
        k = np.array([0.05, 0.1, 0.2, 0.3])
        k = k[(k >= kmin) & (k <= kmax)]
        meas = {"DS1": {0: np.arange(len(k), dtype=float), 2: np.arange(len(k), dtype=float) + 10.0}}
        return k, meas

    monkeypatch.setattr("drift.io.load_observable_measurements", fake_load_observable_measurements)

    k, mock_matrix = _load_pqq_auto_mocks(
        tmp_path, nquantiles=1, quantiles=(1,), ells=(0, 2), rebin=5, kmin=0.1, kmax=0.2
    )

    np.testing.assert_allclose(k, [0.1, 0.2])
    assert mock_matrix.shape == (1, 4)
