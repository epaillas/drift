"""Compare measured DSG multipoles between real space and redshift space.

Reads outputs/dsg_measured.hdf5 and outputs/dsg_measured_real.hdf5 and
produces a panel plot with P0 and P2 for each DS bin, overlaying both spaces.

Saves to outputs/rsd_comparison.png.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_observable_measurements

OUTPUT_DIR  = Path(__file__).parents[1] / "outputs"
RSD_PATH    = OUTPUT_DIR / "dsg_measured.hdf5"
REAL_PATH   = OUTPUT_DIR / "dsg_measured_real.hdf5"

NQUANTILES = 5
ELLS       = (0, 2)


def main():
    missing = [p for p in (RSD_PATH, REAL_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        raise SystemExit(1)

    k_rsd,  rsd_poles  = load_observable_measurements(RSD_PATH, "pqg", nquantiles=NQUANTILES, ells=ELLS)
    k_real, real_poles = load_observable_measurements(REAL_PATH, "pqg", nquantiles=NQUANTILES, ells=ELLS)

    labels = [f"DS{i+1}" for i in range(NQUANTILES)]
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, NQUANTILES))

    fig, axes = plt.subplots(NQUANTILES, 2, figsize=(10, 3 * NQUANTILES), sharex=True)

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax0, ax2 = axes[i]

        ax0.plot(k_rsd,  k_rsd  * rsd_poles[label][0],  color=color, ls="-",  marker="o", ms=3, label="redshift space")
        ax0.plot(k_real, k_real * real_poles[label][0], color=color, ls="--", marker="^", ms=3, label="real space")

        ax2.plot(k_rsd,  k_rsd  * rsd_poles[label][2],  color=color, ls="-",  marker="o", ms=3, label="redshift space")
        ax2.plot(k_real, k_real * real_poles[label][2], color=color, ls="--", marker="^", ms=3, label="real space")

        for ax in (ax0, ax2):
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xscale("log")
            ax.set_ylabel(rf"$k\,P_\ell$ $[(\rm{{Mpc}}/h)^2]$")

        ax0.set_title(f"{label} — $P_0$", fontsize=9)
        ax2.set_title(f"{label} — $P_2$", fontsize=9)

    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel(r"$k$ [$h$/Mpc]")

    fig.suptitle("DRIFT: real vs. redshift space DSG multipoles", fontsize=13)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "rsd_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
