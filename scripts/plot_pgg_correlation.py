"""Plot galaxy power-spectrum and correlation-function multipoles side by side."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pgg_correlation import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "pgg_xi_multipoles_eft_redshift.npz"


def reliable_s_mask(k, s, smin=None, smax=None):
    """Return an s-range mask that avoids the FFTLog edge-dominated region."""
    if smin is None:
        smin = 3.0 / float(np.max(k))
    if smax is None:
        smax = 0.7 / float(np.min(k))
    return (s >= smin) & (s <= smax)


def matching_k_mask(k, smin=None, smax=None):
    """Map an s-range to the rough reciprocal k-range used by the xi mask."""
    kmin = None if smax is None else 0.7 / float(smax)
    kmax = None if smin is None else 3.0 / float(smin)
    mask = np.ones_like(k, dtype=bool)
    if kmin is not None:
        mask &= k >= kmin
    if kmax is not None:
        mask &= k <= kmax
    return mask


def load_galaxy_multipoles(path):
    """Load a galaxy P_ell/XI_ell prediction bundle."""
    data = np.load(path, allow_pickle=True)
    ells = tuple(int(ell) for ell in np.asarray(data["ells"], dtype=int))
    poles = {ell: np.asarray(data[f"P{ell}"], dtype=float) for ell in ells}
    xi_poles = {ell: np.asarray(data[f"XI{ell}"], dtype=float) for ell in ells}
    metadata = {
        "mode": str(data["mode"]),
        "space": str(data["space"]),
        "z": float(data["z"]),
        "q": float(data["q"]) if "q" in data else 1.0,
    }
    return np.asarray(data["k"], dtype=float), np.asarray(data["s"], dtype=float), poles, xi_poles, metadata


def make_figure(k, s, poles, xi_poles, metadata, s_mask=None, k_mask=None, logx=False):
    """Build the two-panel galaxy multipole figure."""
    colors = {0: "C0", 2: "C1", 4: "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    if s_mask is None:
        s_mask = np.ones_like(s, dtype=bool)
    if k_mask is None:
        k_mask = np.ones_like(k, dtype=bool)

    for ell, pk in poles.items():
        axes[0].plot(
            k[k_mask],
            k[k_mask] * pk[k_mask],
            color=colors.get(ell, None),
            label=rf"$\ell={ell}$",
        )

    for ell, xi in xi_poles.items():
        axes[1].plot(
            s[s_mask],
            s[s_mask] ** 2 * xi[s_mask],
            color=colors.get(ell, None),
            label=rf"$\ell={ell}$",
        )

    axes[0].set_xlabel(r"$k$ [$h$/Mpc]")
    axes[0].set_ylabel(r"$k\,P_\ell(k)$ [$({\rm Mpc}/h)^2$]")
    axes[0].axhline(0.0, color="k", lw=0.5, ls="--")
    axes[0].legend()

    axes[1].set_xlabel(r"$s$ [Mpc$/h$]")
    axes[1].set_ylabel(r"$s^2\,\xi_\ell(s)$ [$(\mathrm{Mpc}/h)^2$]")
    axes[1].axhline(0.0, color="k", lw=0.5, ls="--")
    axes[1].legend()

    if logx:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    fig.suptitle(
        rf"DRIFT: galaxy multipoles ({metadata['mode']}, {metadata['space']}, z={metadata['z']:.2f})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, metavar="PATH")
    parser.add_argument("--output", type=Path, default=None, metavar="PATH")
    parser.add_argument("--smin", type=float, default=None, metavar="VALUE")
    parser.add_argument("--smax", type=float, default=None, metavar="VALUE")
    parser.add_argument("--logx", action="store_true")
    args = parser.parse_args()

    k, s, poles, xi_poles, metadata = load_galaxy_multipoles(args.input)
    s_mask = reliable_s_mask(k, s, smin=args.smin, smax=args.smax)
    k_mask = matching_k_mask(k, smin=args.smin, smax=args.smax)
    fig, _ = make_figure(
        k,
        s,
        poles,
        xi_poles,
        metadata,
        s_mask=s_mask,
        k_mask=k_mask,
        logx=args.logx,
    )

    out_path = args.output
    if out_path is None:
        out_path = args.input.with_suffix(".png")

    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
