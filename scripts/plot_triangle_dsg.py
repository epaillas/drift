"""Triangle (corner) plot of the DSG posterior from PocoMC.

Reads outputs/inference_dsg/chains.npz and saves a filled triangle plot
to outputs/inference_dsg/triangle.png.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getdist import MCSamples, plots

CHAINS_PATH = Path(__file__).parents[1] / "outputs" / "inference_dsg" / "chains.npz"
OUTPUT_DIR = CHAINS_PATH.parent

LABELS = [r"b_1", r"b_{q1}", r"b_{q2}", r"b_{q3}", r"b_{q4}", r"b_{q5}"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    d = np.load(CHAINS_PATH, allow_pickle=True)
    samples = d["samples"]
    weights = d["weights"]
    param_names = list(d["param_names"])

    mcs = MCSamples(
        samples=samples,
        weights=weights,
        names=param_names,
        labels=LABELS,
        label="PocoMC posterior",
    )

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.legend_fontsize = 11
    g.triangle_plot([mcs], filled=True, title_limit=1)

    # Mark posterior mean on each 1-D marginal
    means = np.average(samples, weights=weights, axis=0)
    for i, ax in enumerate(g.subplots[i, i] for i in range(len(param_names))):
        ax.axvline(means[i], color="C1", ls="--", lw=1.0)

    out_path = OUTPUT_DIR / "triangle.png"
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
