"""Tests for xi_qg mock-realization inspection plots."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_mock_realizations_xiqg import (
    DEFAULT_ELLS,
    DEFAULT_QUANTILES,
    discover_mock_paths,
    load_mock_realizations,
    make_figure,
)


def test_discover_mock_paths_sorts_and_limits(tmp_path):
    for name in ("dsc_xiqg_poles_ph3002.h5", "dsc_xiqg_poles_ph3000.h5", "dsc_xiqg_poles_ph3001.h5"):
        (tmp_path / name).touch()

    paths = discover_mock_paths(tmp_path, max_realizations=2)

    assert [path.name for path in paths] == [
        "dsc_xiqg_poles_ph3000.h5",
        "dsc_xiqg_poles_ph3001.h5",
    ]


def test_load_mock_realizations_stacks_selected_quantiles_and_means(monkeypatch, tmp_path):
    for name in ("dsc_xiqg_poles_ph3001.h5", "dsc_xiqg_poles_ph3000.h5"):
        (tmp_path / name).touch()

    def fake_load_observable_measurements(path, observable, nquantiles, quantiles, ells, rebin=1, smin=0.0, smax=np.inf):
        assert observable == "xiqg"
        phase = int(Path(path).stem.split("ph")[-1])
        s = np.array([10.0, 20.0, 30.0])
        values = {
            f"DS{quantile}": {
                ell: np.full(3, phase + 10.0 * quantile + ell, dtype=float)
                for ell in ells
            }
            for quantile in quantiles
        }
        return s, values

    monkeypatch.setattr(
        "scripts.plot_mock_realizations_xiqg.load_observable_measurements",
        fake_load_observable_measurements,
    )

    s, realizations, means, phases = load_mock_realizations(
        tmp_path,
        quantiles=(1, 3),
        ells=(0, 2),
    )

    np.testing.assert_allclose(s, [10.0, 20.0, 30.0])
    assert phases == ("ph3000", "ph3001")
    assert realizations["DS1"][0].shape == (2, 3)
    np.testing.assert_allclose(realizations["DS1"][0][0], [3010.0, 3010.0, 3010.0])
    np.testing.assert_allclose(realizations["DS1"][0][1], [3011.0, 3011.0, 3011.0])
    np.testing.assert_allclose(means["DS3"][2], [3032.5, 3032.5, 3032.5])


def test_make_figure_returns_quantile_by_ell_grid_with_mean_on_top():
    s = np.array([10.0, 20.0, 30.0])
    realizations = {
        "DS1": {
            0: np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
            2: np.array([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5]]),
        },
        "DS2": {
            0: np.array([[0.5, 1.0, 1.5], [0.75, 1.25, 1.75]]),
            2: np.array([[2.0, 2.5, 3.0], [2.25, 2.75, 3.25]]),
        },
    }
    means = {
        "DS1": {
            0: np.array([1.25, 2.25, 3.25]),
            2: np.array([4.25, 5.25, 6.25]),
        },
        "DS2": {
            0: np.array([0.625, 1.125, 1.625]),
            2: np.array([2.125, 2.625, 3.125]),
        },
    }

    fig, axes = make_figure(
        s,
        realizations,
        means,
        quantiles=(1, 2),
        ells=(0, 2),
        phases=("ph0001", "ph0002"),
    )

    assert axes.shape == (2, 2)
    lines = axes[0, 0].get_lines()
    data_lines = [line for line in lines if line.get_linestyle() != "--"]
    assert len(data_lines) == 3
    assert data_lines[0].get_color() == "0.7"
    assert np.isclose(data_lines[0].get_linewidth(), 0.7)
    assert data_lines[-1].get_color() == "k"
    assert np.isclose(data_lines[-1].get_linewidth(), 1.8)
    np.testing.assert_allclose(data_lines[-1].get_ydata(), s**2 * means["DS1"][0])
    assert fig._suptitle.get_text() == "DRIFT: xi_{qg} mocks (2 realizations; mean + realizations)"


def test_make_figure_show_mean_only_omits_grey_realizations():
    s = np.array([10.0, 20.0])
    realizations = {"DS1": {0: np.array([[1.0, 2.0], [1.5, 2.5]])}}
    means = {"DS1": {0: np.array([1.25, 2.25])}}

    fig, axes = make_figure(
        s,
        realizations,
        means,
        quantiles=(1,),
        ells=(0,),
        phases=("ph0001", "ph0002"),
        show_mean_only=True,
    )

    lines = axes[0, 0].get_lines()
    data_lines = [line for line in lines if line.get_linestyle() != "--"]
    assert len(data_lines) == 1
    assert data_lines[-1].get_color() == "k"
    assert DEFAULT_QUANTILES == (1, 2, 3, 4, 5)
    assert DEFAULT_ELLS == (0, 2)
    assert fig._suptitle.get_text() == "DRIFT: xi_{qg} mocks (2 realizations; mean only)"
