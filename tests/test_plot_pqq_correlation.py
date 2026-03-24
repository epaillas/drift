"""Tests for DS-pair correlation bundle and plot helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_pqq_correlation import default_auto_pairs, load_dspair_multipoles, make_figure, parse_selected_pairs
from scripts.run_pqq_correlation import key_to_pair, pair_to_key, save_dspair_multipoles


def test_pair_token_roundtrip():
    pair = ("DS1", "DS3")
    assert key_to_pair(pair_to_key(pair)) == pair


def test_default_auto_pairs_and_selection():
    available = (("DS1", "DS1"), ("DS1", "DS2"), ("DS2", "DS2"))
    assert default_auto_pairs(available) == (("DS1", "DS1"), ("DS2", "DS2"))
    assert parse_selected_pairs(None, available) == (("DS1", "DS1"), ("DS2", "DS2"))
    assert parse_selected_pairs(["DS1-DS2"], available) == (("DS1", "DS2"),)


def test_dspair_bundle_roundtrip(tmp_path):
    path = tmp_path / "pqq_bundle.npz"
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {("DS1", "DS1"): {0: np.ones(16), 2: 2.0 * np.ones(16)}, ("DS1", "DS2"): {0: 3.0 * np.ones(16), 2: 4.0 * np.ones(16)}}
    xi = {("DS1", "DS1"): {0: -np.ones(16), 2: -2.0 * np.ones(16)}, ("DS1", "DS2"): {0: -3.0 * np.ones(16), 2: -4.0 * np.ones(16)}}

    save_dspair_multipoles(
        path,
        k=k,
        s=s,
        poles_by_pair=poles,
        xi_by_pair=xi,
        metadata={"observable": "pqq", "mode": "tree", "space": "redshift", "ds_model": "baseline", "z": 0.5, "q": 1.0},
    )

    k_out, s_out, poles_out, xi_out, metadata = load_dspair_multipoles(path)
    np.testing.assert_allclose(k_out, k)
    np.testing.assert_allclose(s_out, s)
    assert metadata["observable"] == "pqq"
    np.testing.assert_allclose(poles_out[("DS1", "DS1")][0], poles[("DS1", "DS1")][0])
    np.testing.assert_allclose(xi_out[("DS1", "DS2")][2], xi[("DS1", "DS2")][2])


def test_pqq_make_figure_returns_expected_grid():
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {("DS1", "DS1"): {0: np.ones(16), 2: 2.0 * np.ones(16)}, ("DS2", "DS2"): {0: 3.0 * np.ones(16), 2: 4.0 * np.ones(16)}}
    xi = {("DS1", "DS1"): {0: -np.ones(16), 2: -2.0 * np.ones(16)}, ("DS2", "DS2"): {0: -3.0 * np.ones(16), 2: -4.0 * np.ones(16)}}

    fig, axes = make_figure(k, s, poles, xi, {"mode": "tree", "space": "redshift", "ds_model": "baseline"})

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_xscale() == "linear"
    fig.savefig(Path("/tmp") / "pqq_corr_test.png", dpi=50)
