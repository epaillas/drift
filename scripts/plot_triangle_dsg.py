"""Triangle (corner) plot of the DSG posterior from PocoMC.

Reads outputs/inference_dsg/<space>/<ds_model>/<mode>/chains.npz and saves
a filled triangle plot to the same directory as triangle.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getdist import MCSamples, plots

SPACE      = "redshift"         # "redshift" | "real"
DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft_full"         # "tree_only" | "eft_lite" | "eft_full"


def _labels_from_names(param_names):
    """Derive getdist-compatible LaTeX labels from parameter names."""
    _map = {
        "sigma8": r"\sigma_8",
        "Omega_m": r"\Omega_m",
        "b1":      r"b_1",
        "c0":      r"c_0",
        "s0":      r"s_0",
    }
    labels = []
    for name in param_names:
        if name in _map:
            labels.append(_map[name])
        elif name.startswith("bq1_"):
            q = name.split("_")[-1]
            labels.append(rf"b_{{q1,{q}}}")
        elif name.startswith("beta_q_"):
            q = name.split("_")[-1]
            labels.append(rf"\beta_{{q,{q}}}")
        elif name.startswith("bq_nabla2_"):
            q = name.split("_")[-1]
            labels.append(rf"b_{{q\nabla^2,{q}}}")
        else:
            labels.append(name)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subset of parameter names to include in the triangle plot.",
    )
    args = parser.parse_args()

    chains_path = (
        Path(__file__).parents[1] / "outputs" / "inference_dsg"
        / SPACE / DS_MODEL / MODEL_MODE / "chains.npz"
    )
    output_dir = chains_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(chains_path, allow_pickle=True)
    samples    = d["samples"]
    weights    = d["weights"]
    param_names = list(d["param_names"])

    # Optionally restrict to a subset of parameters
    if args.params is not None:
        indices     = [param_names.index(p) for p in args.params]
        param_names = [param_names[i] for i in indices]
        samples     = samples[:, indices]

    labels = _labels_from_names(param_names)

    mcs = MCSamples(
        samples=samples,
        weights=weights,
        names=param_names,
        labels=labels,
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

    out_path = output_dir / "triangle.png"
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
