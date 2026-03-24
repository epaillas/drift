"""Tests for DS×galaxy correlation bundle and plot helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_dsg_correlation import load_ds_galaxy_multipoles, make_figure
from scripts.run_dsg_correlation import save_ds_galaxy_multipoles


def test_ds_galaxy_bundle_roundtrip(tmp_path):
    path = tmp_path / "dsg_bundle.npz"
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {"DS1": {0: np.ones(16), 2: 2.0 * np.ones(16)}, "DS5": {0: 3.0 * np.ones(16), 2: 4.0 * np.ones(16)}}
    xi = {"DS1": {0: -np.ones(16), 2: -2.0 * np.ones(16)}, "DS5": {0: -3.0 * np.ones(16), 2: -4.0 * np.ones(16)}}

    save_ds_galaxy_multipoles(
        path,
        k=k,
        s=s,
        poles_by_label=poles,
        xi_by_label=xi,
        metadata={"observable": "pqg", "mode": "tree", "space": "redshift", "ds_model": "baseline", "z": 0.5, "q": 1.0},
    )

    k_out, s_out, poles_out, xi_out, metadata = load_ds_galaxy_multipoles(path)
    np.testing.assert_allclose(k_out, k)
    np.testing.assert_allclose(s_out, s)
    assert metadata["observable"] == "pqg"
    np.testing.assert_allclose(poles_out["DS1"][0], poles["DS1"][0])
    np.testing.assert_allclose(xi_out["DS5"][2], xi["DS5"][2])


def test_dsg_make_figure_returns_expected_grid():
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {"DS1": {0: np.ones(16), 2: 2.0 * np.ones(16)}, "DS5": {0: 3.0 * np.ones(16), 2: 4.0 * np.ones(16)}}
    xi = {"DS1": {0: -np.ones(16), 2: -2.0 * np.ones(16)}, "DS5": {0: -3.0 * np.ones(16), 2: -4.0 * np.ones(16)}}

    fig, axes = make_figure(k, s, poles, xi, {"mode": "tree", "space": "redshift", "ds_model": "baseline"})

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_xscale() == "linear"
    assert axes[0, 1].get_xscale() == "linear"
    fig.savefig(Path("/tmp") / "dsg_corr_test.png", dpi=50)
