"""Plot density-split × matter power spectrum multipoles.

Reads outputs/dsm_multipoles.npz (produced by run_dsm_multipoles.py) and
saves a two-panel figure (P_0 and P_2) to outputs/dsm_multipoles.png.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_predictions

INPUT_PATH = Path(__file__).parents[1] / "outputs" / "dsm_multipoles.npz"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def main():
    k, multipoles_per_bin = load_predictions(INPUT_PATH)

    labels = sorted(multipoles_per_bin.keys())
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, color in zip(labels, colors):
        poles = multipoles_per_bin[label]
        axes[0].plot(k, k * poles[0], color=color, label=label)
        axes[1].plot(k, k * poles[2], color=color, label=label)

    for ax, title in zip(axes, [r"$k\,P_0(k)$", r"$k\,P_2(k)$"]):
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(title + r" [$({\rm Mpc}/h)^2$]")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=8)

    fig.suptitle(r"DRIFT: Density-split $\times$ matter multipoles (tree-level)", fontsize=13)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "dsm_multipoles.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
