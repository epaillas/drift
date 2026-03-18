"""Triangle (corner) plot from any PocoMC chains.npz file.

Usage
-----
    python plot_triangle.py chains.npz
    python plot_triangle.py chains.npz --params sigma8 Omega_m b1
    python plot_triangle.py chains.npz -o my_triangle.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getdist import MCSamples, plots

# Known parameter -> LaTeX mappings
_LATEX = {
    "sigma8":  r"\sigma_8",
    "Omega_m": r"\Omega_m",
    "b1":      r"b_1",
    "b2":      r"b_2",
    "bs2":     r"b_{s^2}",
    "c0":      r"c_0",
    "c2":      r"c_2",
    "c4":      r"c_4",
    "s0":      r"s_0",
}

# Patterns: prefix -> latex template (the trailing quantile index is appended)
_LATEX_PATTERNS = [
    ("bq1_",       r"b_{{q1,{q}}}"),
    ("beta_q_",    r"\beta_{{q,{q}}}"),
    ("bq_nabla2_", r"b_{{q\nabla^2,{q}}}"),
]


def _labels_from_names(param_names):
    """Derive getdist-compatible LaTeX labels from parameter names."""
    labels = []
    for name in param_names:
        if name in _LATEX:
            labels.append(_LATEX[name])
            continue
        matched = False
        for prefix, template in _LATEX_PATTERNS:
            if name.startswith(prefix):
                q = name.split("_")[-1]
                labels.append(template.format(q=q))
                matched = True
                break
        if not matched:
            labels.append(name)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chains", type=Path,
                        help="Path to a PocoMC chains.npz file.")
    parser.add_argument("--params", nargs="+", default=None, metavar="NAME",
                        help="Subset of parameter names to plot.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output image path (default: triangle.png next to chains file).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    d = np.load(args.chains, allow_pickle=True)
    samples     = d["samples"]
    weights     = d["weights"]
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

    out_path = args.output or (args.chains.parent / "triangle.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
