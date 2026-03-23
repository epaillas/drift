"""Visualize the P_qg correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift.covariance import plot_correlation_matrix
from drift.io import diagonal_covariance
from inference_dsg import (
    CONFIG_PATH,
    ELLS,
    _build_analytic_dsg_fiducials,
    _build_data_mask,
    _default_ds_cross_shot_noise,
    _default_ds_pair_shot_noise,
    _parse_kmax,
    _quantile_labels,
)
from drift.io import analytic_pqg_covariance

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def _parse_quantiles(values):
    if values is None:
        return (1, 5)
    return tuple(int(value) for value in values)


def _pair_block_edges(labels, block_sizes):
    """Return cumulative matrix edges separating DS-label blocks."""
    if len(labels) * len(ELLS) != len(block_sizes):
        raise ValueError("block_sizes must contain one entry per label-ell block.")

    edges = []
    offset = 0
    for idx in range(len(labels)):
        label_size = sum(block_sizes[idx * len(ELLS):(idx + 1) * len(ELLS)])
        offset += label_size
        edges.append(offset)
    return edges[:-1]


def _ell_block_edges(block_sizes):
    """Return cumulative matrix edges separating ell sub-blocks."""
    edges = []
    offset = 0
    for size in block_sizes:
        offset += size
        edges.append(offset)
    return edges[:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q")
    parser.add_argument("--kmin", type=float, default=0.01, metavar="VALUE")
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE")
    parser.add_argument("--nk", type=int, default=80, metavar="N")
    parser.add_argument("--diag-cov", action="store_true",
                        help="Plot diagonal covariance instead of the analytic source.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Plot analytic cubic-box covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms. Only 'gaussian' is currently supported.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR")
    parser.add_argument("--box-volume", type=float, default=1.0e9, metavar="V")
    parser.add_argument("--galaxy-shot-noise", type=float, default=400.0, metavar="P0")
    parser.add_argument("--ds-pair-auto-shot-noise", type=float, default=250.0, metavar="PQQ0")
    parser.add_argument("--ds-pair-cross-shot-noise", type=float, default=40.0, metavar="PQQX")
    parser.add_argument("--ds-cross-shot-noise", type=float, default=0.0, metavar="PQG0")
    args = parser.parse_args()

    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    quantiles = _parse_quantiles(args.quantiles)
    labels = _quantile_labels(quantiles)
    k = np.logspace(np.log10(max(args.kmin, 0.005)), np.log10(0.3), args.nk)
    fiducials = _build_analytic_dsg_fiducials(args, k, quantiles)

    kmax_dict = _parse_kmax(args.kmax, ELLS)
    full_mask = _build_data_mask(k, ELLS, quantiles, kmax_dict, kmin=args.kmin)
    block_sizes = [
        int(((k >= args.kmin) & (k <= kmax_dict[ell])).sum())
        for _label in labels
        for ell in ELLS
    ]

    flat = np.concatenate([
        fiducials["pqg_poles"][label][ell]
        for label in labels
        for ell in ELLS
    ])

    if args.diag_cov:
        cov, _ = diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
    else:
        cov, _ = analytic_pqg_covariance(
            k,
            fiducials["pqg_poles"],
            fiducials["pqq_poles"],
            fiducials["pgg_poles"],
            ELLS,
            volume=args.box_volume,
            ds_labels=labels,
            galaxy_shot_noise=args.galaxy_shot_noise,
            ds_pair_shot_noise=_default_ds_pair_shot_noise(
                labels,
                args.ds_pair_auto_shot_noise,
                args.ds_pair_cross_shot_noise,
            ),
            ds_cross_shot_noise=_default_ds_cross_shot_noise(
                labels,
                args.ds_cross_shot_noise,
            ),
            mask=full_mask,
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
        )

    label_str = ", ".join(labels)
    fig, ax = plot_correlation_matrix(
        cov,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=rf"$P_{{qg}}$ correlation ({label_str})",
    )

    for edge in _ell_block_edges(block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.35, alpha=0.25)
        ax.axhline(edge - 0.5, color="k", lw=0.35, alpha=0.25)

    for edge in _pair_block_edges(labels, block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.8)
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = OUTPUT_DIR / "correlation_matrix_pqg.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
