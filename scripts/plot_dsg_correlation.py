"""Plot density-split×galaxy power-spectrum and correlation-function multipoles."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_pgg_correlation import matching_k_mask, reliable_s_mask
from run_dsg_correlation import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "dsg_xi_multipoles_tree_redshift_baseline.npz"


def load_ds_galaxy_multipoles(path):
    """Load a DS×galaxy P_ell/XI_ell prediction bundle."""
    data = np.load(path, allow_pickle=True)
    labels = tuple(str(label) for label in data["labels"])
    ells = tuple(int(ell) for ell in np.asarray(data["ells"], dtype=int))
    poles_by_label = {
        label: {ell: np.asarray(data[f"{label}_P{ell}"], dtype=float) for ell in ells}
        for label in labels
    }
    xi_by_label = {
        label: {ell: np.asarray(data[f"{label}_XI{ell}"], dtype=float) for ell in ells}
        for label in labels
    }
    metadata = {
        "observable": str(data["observable"]),
        "mode": str(data["mode"]),
        "space": str(data["space"]),
        "ds_model": str(data["ds_model"]),
        "z": float(data["z"]),
        "q": float(data["q"]) if "q" in data else 1.0,
    }
    return np.asarray(data["k"], dtype=float), np.asarray(data["s"], dtype=float), poles_by_label, xi_by_label, metadata


def make_figure(k, s, poles_by_label, xi_by_label, metadata, s_mask=None, k_mask=None, logx=False):
    """Build a two-column figure with one row per multipole."""
    labels = tuple(sorted(poles_by_label))
    ells = tuple(sorted(next(iter(poles_by_label.values())).keys()))
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(labels)))
    fig, axes = plt.subplots(len(ells), 2, figsize=(12, 3.5 * len(ells)), squeeze=False)

    if s_mask is None:
        s_mask = np.ones_like(s, dtype=bool)
    if k_mask is None:
        k_mask = np.ones_like(k, dtype=bool)

    for row, ell in enumerate(ells):
        ax_pk, ax_xi = axes[row]
        for label, color in zip(labels, colors):
            ax_pk.plot(k[k_mask], k[k_mask] * poles_by_label[label][ell][k_mask], color=color, label=label)
            ax_xi.plot(s[s_mask], s[s_mask] ** 2 * xi_by_label[label][ell][s_mask], color=color, label=label)

        ax_pk.set_ylabel(rf"$k\,P_{ell}(k)$ [$({{\rm Mpc}}/h)^2$]")
        ax_xi.set_ylabel(rf"$s^2\,\xi_{ell}(s)$ [$(\mathrm{{Mpc}}/h)^2$]")
        ax_pk.axhline(0.0, color="k", lw=0.5, ls="--")
        ax_xi.axhline(0.0, color="k", lw=0.5, ls="--")
        if row == 0:
            ax_pk.legend(fontsize=8)
            ax_xi.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel(r"$k$ [$h$/Mpc]" if ax is axes[-1, 0] else r"$s$ [Mpc$/h$]")

    if logx:
        for ax in axes[:, 0]:
            ax.set_xscale("log")
        for ax in axes[:, 1]:
            ax.set_xscale("log")

    fig.suptitle(
        rf"DRIFT: DS$\times$galaxy multipoles ({metadata['mode']}, {metadata['space']}, {metadata['ds_model']})",
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

    k, s, poles_by_label, xi_by_label, metadata = load_ds_galaxy_multipoles(args.input)
    s_mask = reliable_s_mask(k, s, smin=args.smin, smax=args.smax)
    k_mask = matching_k_mask(k, smin=args.smin, smax=args.smax)
    fig, _ = make_figure(k, s, poles_by_label, xi_by_label, metadata, s_mask=s_mask, k_mask=k_mask, logx=args.logx)

    out_path = args.output
    if out_path is None:
        out_path = args.input.with_suffix(".png")

    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
