"""Tests for direct xi_qg mock covariance loading."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import (
    _load_mock_matrix,
    _load_xiqg_mocks,
    _mock_cache_key,
    estimate_mock_covariance,
    load_observable_measurements,
)


def test_load_observable_measurements_projects_requested_quantiles_and_ells(monkeypatch):
    class FakePole:
        def __init__(self, s, values):
            self.s = np.asarray(s)
            self._values = np.asarray(values)

        def value(self):
            return self._values

    class FakePoles:
        def __init__(self, s, values_by_ell):
            self._s = np.asarray(s)
            self._values_by_ell = values_by_ell

        def get(self, ell):
            return FakePole(self._s, self._values_by_ell[ell])

    class FakeCorrelation:
        def __init__(self, qid, s):
            self._qid = qid
            self._s = np.asarray(s)

        def select(self, s):
            if isinstance(s, slice):
                return FakeCorrelation(self._qid, self._s[s])
            smin, smax = s
            mask = (self._s >= smin) & (self._s <= smax)
            return FakeCorrelation(self._qid, self._s[mask])

        def project(self, ells):
            values_by_ell = {
                ell: np.arange(self._s.size, dtype=float) + 100.0 * self._qid + ell
                for ell in ells
            }
            return FakePoles(self._s, values_by_ell)

    class FakeTree:
        def __init__(self, s):
            self._s = np.asarray(s)

        def select(self, s):
            smin, smax = s
            mask = (self._s >= smin) & (self._s <= smax)
            return FakeTree(self._s[mask])

        def get(self, quantiles):
            return FakeCorrelation(quantiles, self._s)

    monkeypatch.setattr("lsstypes.read", lambda path: FakeTree([5.0, 15.0, 25.0, 35.0]))

    s, meas = load_observable_measurements(
        "ignored.h5",
        "xiqg",
        nquantiles=5,
        quantiles=(1, 3),
        ells=(0, 4),
        smin=10.0,
        smax=30.0,
    )

    np.testing.assert_allclose(s, [15.0, 25.0])
    assert tuple(meas) == ("DS1", "DS3")
    np.testing.assert_allclose(meas["DS1"][0], [0.0, 1.0])
    np.testing.assert_allclose(meas["DS1"][4], [4.0, 5.0])
    np.testing.assert_allclose(meas["DS3"][0], [200.0, 201.0])
    np.testing.assert_allclose(meas["DS3"][4], [204.0, 205.0])


def test_load_observable_measurements_xiqg_applies_rebin_after_s_cut(monkeypatch):
    class FakePole:
        def __init__(self, s, values):
            self.s = np.asarray(s)
            self._values = np.asarray(values)

        def value(self):
            return self._values

    class FakePoles:
        def __init__(self, s, ells):
            self._s = np.asarray(s)
            self._ells = tuple(ells)

        def get(self, ell):
            return FakePole(self._s, np.arange(self._s.size, dtype=float) + ell)

    class FakeCorrelation:
        def __init__(self, s):
            self._s = np.asarray(s)

        def select(self, s):
            if isinstance(s, slice):
                return FakeCorrelation(self._s[s])
            smin, smax = s
            mask = (self._s >= smin) & (self._s <= smax)
            return FakeCorrelation(self._s[mask])

        def project(self, ells):
            return FakePoles(self._s, ells)

    class FakeTree:
        def __init__(self, s):
            self._s = np.asarray(s)

        def get(self, quantiles):
            return FakeCorrelation(self._s)

    monkeypatch.setattr("lsstypes.read", lambda path: FakeTree(np.arange(0.5, 10.5, 1.0)))

    s, meas = load_observable_measurements(
        "ignored.h5",
        "xiqg",
        nquantiles=1,
        quantiles=(1,),
        ells=(0, 2),
        rebin=3,
        smin=2.0,
        smax=8.0,
    )

    np.testing.assert_allclose(s, [2.5, 5.5])
    np.testing.assert_allclose(meas["DS1"][0], [0.0, 1.0])
    np.testing.assert_allclose(meas["DS1"][2], [2.0, 3.0])


def test_load_xiqg_mocks_reads_expected_pattern_and_order(tmp_path, monkeypatch):
    xiqg_dir = tmp_path / "dsc_xiqg"
    xiqg_dir.mkdir()
    for name in ("dsc_xiqg_poles_ph3001.h5", "dsc_xiqg_poles_ph3000.h5", "ignore_me.h5"):
        (xiqg_dir / name).touch()

    calls = []

    def fake_load_observable_measurements(path, observable, nquantiles, quantiles, ells, rebin=1, smin=0.0, smax=np.inf):
        assert observable == "xiqg"
        assert rebin == 1
        calls.append(Path(path).name)
        s = np.array([10.0, 20.0, 30.0, 40.0])
        s = s[(s >= smin) & (s <= smax)]
        phase = int(Path(path).stem.split("ph")[-1])
        meas = {
            "DS1": {
                0: np.arange(len(s), dtype=float) + phase + 1.0,
                2: np.arange(len(s), dtype=float) + phase + 3.0,
            },
            "DS3": {
                0: np.arange(len(s), dtype=float) + phase + 5.0,
                2: np.arange(len(s), dtype=float) + phase + 7.0,
            },
        }
        return s, meas

    monkeypatch.setattr("drift.io.load_observable_measurements", fake_load_observable_measurements)

    s, mock_matrix = _load_xiqg_mocks(
        tmp_path,
        nquantiles=3,
        quantiles=(1, 3),
        ells=(0, 2),
    )

    np.testing.assert_allclose(s, [10.0, 20.0, 30.0, 40.0])
    assert calls == ["dsc_xiqg_poles_ph3000.h5", "dsc_xiqg_poles_ph3001.h5"]
    np.testing.assert_allclose(
        mock_matrix,
        [
            [3001.0, 3002.0, 3003.0, 3004.0, 3003.0, 3004.0, 3005.0, 3006.0, 3005.0, 3006.0, 3007.0, 3008.0, 3007.0, 3008.0, 3009.0, 3010.0],
            [3002.0, 3003.0, 3004.0, 3005.0, 3004.0, 3005.0, 3006.0, 3007.0, 3006.0, 3007.0, 3008.0, 3009.0, 3008.0, 3009.0, 3010.0, 3011.0],
        ],
    )


def test_load_xiqg_mocks_forwards_rebin(tmp_path, monkeypatch):
    xiqg_dir = tmp_path / "dsc_xiqg"
    xiqg_dir.mkdir()
    (xiqg_dir / "dsc_xiqg_poles_ph3000.h5").touch()

    captured = {}

    def fake_load_observable_measurements(path, observable, nquantiles, quantiles, ells, rebin=1, smin=0.0, smax=np.inf):
        captured["observable"] = observable
        captured["rebin"] = rebin
        return np.array([10.0, 20.0]), {"DS1": {0: np.array([1.0, 2.0])}}

    monkeypatch.setattr("drift.io.load_observable_measurements", fake_load_observable_measurements)

    s, mock_matrix = _load_xiqg_mocks(tmp_path, nquantiles=1, quantiles=(1,), ells=(0,), rebin=4)

    np.testing.assert_allclose(s, [10.0, 20.0])
    np.testing.assert_allclose(mock_matrix, [[1.0, 2.0]])
    assert captured == {"observable": "xiqg", "rebin": 4}


def test_load_mock_matrix_interpolates_xiqg_to_requested_s_grid(tmp_path, monkeypatch):
    xiqg_dir = tmp_path / "dsc_xiqg"
    xiqg_dir.mkdir()
    (xiqg_dir / "dsc_xiqg_poles_ph3000.h5").touch()

    def fake_load_xiqg_mocks(directory, nquantiles, quantiles, ells, rebin=1, smin=0.0, smax=np.inf):
        assert rebin == 13
        s = np.array([10.0, 20.0, 30.0])
        row = np.array([1.0, 3.0, 5.0, 10.0, 30.0, 50.0])
        return s, row[None, :]

    monkeypatch.setattr("drift.io._load_xiqg_mocks", fake_load_xiqg_mocks)

    mock_matrix = _load_mock_matrix(
        tmp_path,
        "xiqg",
        (0, 2),
        s_data=np.array([15.0, 25.0]),
        nquantiles=1,
        quantiles=(1,),
    )

    np.testing.assert_allclose(mock_matrix, [[2.0, 4.0, 20.0, 40.0]])


def test_mock_cache_key_for_xiqg_depends_on_s_range():
    key_a = _mock_cache_key("xiqg", (0, 2), 13, nquantiles=2, quantiles=(1, 3), smin=10.0, smax=80.0)
    key_b = _mock_cache_key("xiqg", (0, 2), 13, nquantiles=2, quantiles=(1, 3), smin=20.0, smax=80.0)

    assert key_a != key_b


def test_estimate_mock_covariance_optionally_returns_precision(monkeypatch):
    monkeypatch.setattr(
        "drift.io._load_mock_matrix",
        lambda *args, **kwargs: np.array(
            [
                [1.0, 2.0],
                [2.0, 1.2],
                [3.1, 0.4],
                [4.3, -1.1],
                [5.2, -1.7],
            ]
        ),
    )

    cov = estimate_mock_covariance("ignored", "pgg", (0,), return_precision=False)
    assert cov.shape == (2, 2)

    cov2, precision = estimate_mock_covariance("ignored", "pgg", (0,), return_precision=True)
    np.testing.assert_allclose(cov, cov2)
    assert precision.shape == (2, 2)
