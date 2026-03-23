"""Visualize the P_gg correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import plot_correlation_matrix
from drift.io import diagonal_covariance, load_pgg_measurements
from inference_pgg import ELLS, MEAS_PATH, MODEL_MODE, OUTPUT_DIR, SPACE, _resolve_pgg_covariance


def _parse_kmax(values):
    kmax = {ell: 0.5 for ell in ELLS}
    if values is None:
        return kmax
    if len(values) == 1 and ":" not in values[0]:
        val = float(values[0])
        return {ell: val for ell in ELLS}
    for item in values:
        ell_str, val_str = item.split(":")
        kmax[int(ell_str)] = float(val_str)
    return kmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebin", type=int, default=13, metavar="N")
    parser.add_argument("--kmin", type=float, default=0.01, metavar="VALUE")
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE")
    parser.add_argument("--diag-cov", action="store_true",
                        help="Plot diagonal covariance instead of the analytic source.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Plot analytic cubic-box covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms: 'gaussian', 'gaussian+effective_cng', 'gaussian+ssc', or 'gaussian+effective_cng+ssc'.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR")
    parser.add_argument("--box-volume", type=float, default=None, metavar="V")
    parser.add_argument("--number-density", type=float, default=None, metavar="N")
    parser.add_argument("--shot-noise", type=float, default=None, metavar="P0")
    parser.add_argument(
        "--cng-amplitude",
        type=float,
        default=None,
        metavar="A",
        help=(
            "Amplitude of the effective connected covariance term. "
            "Defaults to the shared nonzero value when effective_cng is enabled."
        ),
    )
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR")
    args = parser.parse_args()

    k, poles = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=args.rebin, kmin=args.kmin)
    flat = np.concatenate([poles[ell] for ell in ELLS])
    kmax_dict = _parse_kmax(args.kmax)
    per_ell_masks = [((k >= args.kmin) & (k <= kmax_dict[ell])) for ell in ELLS]
    full_mask = np.concatenate(per_ell_masks)
    block_sizes = [int(mask.sum()) for mask in per_ell_masks]

    if args.diag_cov:
        cov, _ = diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
        plot_k = np.arange(full_mask.sum())
        plot_ells = None
    else:
        cov, _ = _resolve_pgg_covariance(
            args, k, flat[full_mask], full_mask, fiducial_poles=poles,
        )
        plot_k = k
        plot_ells = ELLS

    fig, ax = plot_correlation_matrix(
        cov,
        ells=plot_ells,
        block_sizes=block_sizes if plot_ells is not None else None,
        cmap="RdBu_r",
        title=rf"$P_{{gg}}$ correlation ({SPACE}, {MODEL_MODE})",
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "correlation_matrix_pgg.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
