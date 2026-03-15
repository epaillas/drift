"""Plot density-split × galaxy power spectrum multipoles.

Reads outputs/dsg_multipoles.npz (produced by run_dsg_multipoles.py) and
saves a two-panel figure (P_0 and P_2) to outputs/dsg_multipoles.png.
If outputs/dsg_measured.hdf5 is present, measurements are overlaid as points.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_predictions, load_measurements

INPUT_PATH = Path(__file__).parents[1] / "outputs" / "dsg_multipoles.npz"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def main():
    k_th, theory = load_predictions(INPUT_PATH)

    meas_path = OUTPUT_DIR / "dsg_measured.hdf5"
    measured = None
    if meas_path.exists():
        k_m, measured = load_measurements(meas_path)

    labels = sorted(theory.keys())
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, color in zip(labels, colors):
        poles = theory[label]
        axes[0].plot(k_th, k_th * poles[0], color=color, label=label)
        axes[1].plot(k_th, k_th * poles[2], color=color, label=label)

        if measured is not None and label in measured:
            mpoles = measured[label]
            axes[0].scatter(k_m, k_m * mpoles[0], color=color, s=8, zorder=3)
            axes[1].scatter(k_m, k_m * mpoles[2], color=color, s=8, zorder=3)

    for ax, title in zip(axes, [r"$k\,P_0(k)$", r"$k\,P_2(k)$"]):
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(title + r" [$({\rm Mpc}/h)^2$]")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=8)

    fig.suptitle(r"DRIFT: Density-split $\times$ galaxy multipoles (tree-level)", fontsize=13)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "dsg_multipoles.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
