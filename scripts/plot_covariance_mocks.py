"""Plot the mean of all covariance mock measurements per quantile.

Monopole (ell=0) and quadrupole (ell=2) are shown in separate panels,
with 1-sigma scatter from the mock ensemble shown as shaded bands.

Saves to outputs/hods/for_covariance/mean_multipoles.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_covariance_mocks

COV_DIR   = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
ELLS      = (0, 2)
QUANTILES = (1, 2, 3, 4, 5)
NQUANTILES = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebin",
        type=int,
        default=5,
        metavar="N",
        help="Keep every Nth k-bin (default: 5).",
    )
    args = parser.parse_args()

    print(f"Loading mocks from {COV_DIR} ...")
    k, mock_matrix = load_covariance_mocks(
        COV_DIR, nquantiles=NQUANTILES, quantiles=QUANTILES, ells=ELLS, rebin=args.rebin,
    )
    n_mocks = mock_matrix.shape[0]
    nk = len(k)
    print(f"  {n_mocks} mocks, {nk} k-bins")

    # Reshape mock_matrix (n_mocks, nq*nell*nk) -> (n_mocks, nq, nell, nk)
    nell = len(ELLS)
    nq   = len(QUANTILES)
    mocks = mock_matrix.reshape(n_mocks, nq, nell, nk)

    mean  = mocks.mean(axis=0)   # (nq, nell, nk)
    sigma = mocks.std(axis=0)    # (nq, nell, nk)

    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, nq))

    fig, axes = plt.subplots(1, nell, figsize=(6 * nell, 5), sharey=False)
    axes = np.atleast_1d(axes)

    for qi, (q, color) in enumerate(zip(QUANTILES, colors)):
        label = f"DS{q}"
        for ai, ell in enumerate(ELLS):
            ax = axes[ai]
            mu  = mean[qi, ai]
            sig = sigma[qi, ai]
            ax.plot(k, k * mu, color=color, label=label)
            ax.fill_between(k, k * (mu - sig), k * (mu + sig),
                            color=color, alpha=0.2, linewidth=0)

    for ax, ell in zip(axes, ELLS):
        ax.set_title(rf"$\ell = {ell}$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(rf"$k\,P_{{{ell}}}(k)$" + r" [$({\rm Mpc}/h)^2$]")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, ls="--")

    axes[-1].legend(fontsize=8, title="quantile")

    fig.suptitle(
        f"Covariance mocks: mean ± 1σ  ({n_mocks} realizations)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = COV_DIR / "mean_multipoles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
