"""Tests for galaxy P_gg to xi_gg run/plot scripts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.plot_pgg_correlation import (
    load_galaxy_multipoles,
    make_figure,
    matching_k_mask,
    reliable_s_mask,
)
from scripts.run_pgg_correlation import save_galaxy_multipoles


def test_save_and_load_galaxy_multipoles_roundtrip(tmp_path):
    path = tmp_path / "galaxy_bundle.npz"
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {0: np.ones(16), 2: 2.0 * np.ones(16), 4: 3.0 * np.ones(16)}
    xi_poles = {0: -np.ones(16), 2: -2.0 * np.ones(16), 4: -3.0 * np.ones(16)}

    save_galaxy_multipoles(
        path,
        k=k,
        s=s,
        poles=poles,
        xi_poles=xi_poles,
        metadata={"mode": "eft", "space": "redshift", "z": 0.5, "q": 1.0},
    )

    k_out, s_out, poles_out, xi_out, metadata = load_galaxy_multipoles(path)

    np.testing.assert_allclose(k_out, k)
    np.testing.assert_allclose(s_out, s)
    assert metadata == {"mode": "eft", "space": "redshift", "z": 0.5, "q": 1.0}
    for ell in (0, 2, 4):
        np.testing.assert_allclose(poles_out[ell], poles[ell])
        np.testing.assert_allclose(xi_out[ell], xi_poles[ell])


def test_make_figure_and_save(tmp_path):
    k = np.logspace(-2, -0.3, 16)
    s = np.logspace(0.2, 1.7, 16)
    poles = {0: np.linspace(1.0, 2.0, 16), 2: np.linspace(0.5, 1.5, 16)}
    xi_poles = {0: np.linspace(-0.1, 0.1, 16), 2: np.linspace(-0.05, 0.05, 16)}

    mask = reliable_s_mask(k, s, smin=2.0, smax=30.0)
    k_mask = matching_k_mask(k, smin=2.0, smax=30.0)
    fig, axes = make_figure(
        k,
        s,
        poles,
        xi_poles,
        {"mode": "tree", "space": "real", "z": 0.5, "q": 1.0},
        s_mask=mask,
        k_mask=k_mask,
    )

    assert len(axes) == 2
    assert axes[0].get_xscale() == "linear"
    assert axes[1].get_xscale() == "linear"
    xdata = axes[1].lines[0].get_xdata()
    assert np.all(xdata >= 2.0)
    assert np.all(xdata <= 30.0)
    kdata = axes[0].lines[0].get_xdata()
    assert np.all(kdata >= 0.7 / 30.0)
    assert np.all(kdata <= 3.0 / 2.0)

    out_path = tmp_path / "figure.png"
    fig.savefig(out_path, dpi=80)
    assert out_path.exists()


def test_reliable_s_mask_uses_k_range():
    k = np.logspace(-4, 0, 32)
    s = np.logspace(-1, 5, 64)

    mask = reliable_s_mask(k, s)

    kept = s[mask]
    assert kept[0] >= 3.0 / k.max()
    assert kept[-1] <= 0.7 / k.min()


def test_matching_k_mask_uses_s_range():
    k = np.logspace(-3, 0, 64)

    mask = matching_k_mask(k, smin=5.0, smax=40.0)

    kept = k[mask]
    assert kept[0] >= 0.7 / 40.0
    assert kept[-1] <= 3.0 / 5.0
