"""Find catastrophic xi_qg mock realizations using robust outlier scores."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_mock_realizations_xiqg import (
    DEFAULT_ELLS,
    DEFAULT_MOCK_DIR,
    DEFAULT_QUANTILES,
    _parse_ells,
    _parse_quantiles,
    load_mock_realizations,
)

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
DEFAULT_CSV_PATH = OUTPUT_DIR / "outlier_realizations_xiqg.csv"
DEFAULT_THRESHOLD = 5.0
MAD_TO_SIGMA = 1.4826


def _panel_robust_scores(values, scale_floor=1.0):
    """Return panel median, residuals, scale, and max-|z| scores per realization."""
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=0)
    residuals = values - median
    mad = np.median(np.abs(residuals), axis=0)
    scale = MAD_TO_SIGMA * mad
    positive_scale = scale[scale > 0.0]
    if positive_scale.size:
        panel_floor = max(float(np.median(positive_scale)), float(scale_floor))
    else:
        panel_floor = float(scale_floor)
    scale = np.maximum(scale, panel_floor)
    zscores = np.abs(residuals) / scale
    scores = np.max(zscores, axis=1)
    worst_bins = np.argmax(zscores, axis=1)
    return median, residuals, scale, zscores, scores, worst_bins


def find_outlier_realizations(
    s,
    realizations,
    phases,
    *,
    quantiles=DEFAULT_QUANTILES,
    ells=DEFAULT_ELLS,
    threshold=DEFAULT_THRESHOLD,
    scale_floor=1.0,
):
    """Score and flag xi_qg mock realizations."""
    labels = tuple(f"DS{quantile}" for quantile in quantiles)
    yscale = np.asarray(s, dtype=float) ** 2

    results = [
        {
            "phase": phase,
            "flagged": False,
            "max_panel_score": -np.inf,
            "worst_quantile": None,
            "worst_ell": None,
            "worst_s": None,
            "worst_value": None,
            "worst_median": None,
            "worst_residual": None,
        }
        for phase in phases
    ]

    for label in labels:
        for ell in ells:
            panel_values = yscale[None, :] * np.asarray(realizations[label][ell], dtype=float)
            median, residuals, _scale, _zscores, scores, worst_bins = _panel_robust_scores(
                panel_values,
                scale_floor=scale_floor,
            )
            for idx, score in enumerate(scores):
                if score <= results[idx]["max_panel_score"]:
                    continue
                bin_idx = int(worst_bins[idx])
                results[idx].update(
                    max_panel_score=float(score),
                    worst_quantile=label,
                    worst_ell=int(ell),
                    worst_s=float(s[bin_idx]),
                    worst_value=float(panel_values[idx, bin_idx]),
                    worst_median=float(median[bin_idx]),
                    worst_residual=float(residuals[idx, bin_idx]),
                )

    for result in results:
        result["flagged"] = bool(result["max_panel_score"] >= threshold)

    results.sort(key=lambda item: item["max_panel_score"], reverse=True)
    return results


def write_outlier_csv(path, results, *, flagged_only=False):
    """Write per-phase outlier scores to CSV."""
    fieldnames = [
        "phase",
        "flagged",
        "max_panel_score",
        "worst_quantile",
        "worst_ell",
        "worst_s",
        "worst_value",
        "worst_median",
        "worst_residual",
    ]
    rows = results if not flagged_only else [row for row in results if row["flagged"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-dir", type=Path, default=DEFAULT_MOCK_DIR, metavar="PATH",
                        help="Directory containing dsc_xiqg_poles_ph*.h5 covariance mocks.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to scan. Defaults to 1 2 3 4 5.")
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to scan. Defaults to 0 2.")
    parser.add_argument("--smin", type=float, default=0.0, metavar="VALUE",
                        help="Minimum separation retained from the xi_qg mocks, in Mpc/h.")
    parser.add_argument("--smax", type=float, default=200.0, metavar="VALUE",
                        help="Maximum separation retained from the xi_qg mocks, in Mpc/h.")
    parser.add_argument("--rebin", type=int, default=1, metavar="N",
                        help="Keep every Nth separation bin when loading the xi_qg mocks.")
    parser.add_argument("--max-realizations", type=int, default=None, metavar="N",
                        help="Only load the first N sorted realizations.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, metavar="VALUE",
                        help="Flag a phase when its max robust MAD score reaches this threshold.")
    parser.add_argument("--top", type=int, default=None, metavar="N",
                        help="Also print the top N worst phases, even if they are below threshold.")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Write only flagged phases to the CSV output.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, metavar="PATH",
                        help="CSV path for per-phase outlier scores.")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    ells = _parse_ells(args.ells)
    s, realizations, _means, phases = load_mock_realizations(
        args.mock_dir,
        quantiles=quantiles,
        ells=ells,
        rebin=args.rebin,
        smin=args.smin,
        smax=args.smax,
        max_realizations=args.max_realizations,
    )
    results = find_outlier_realizations(
        s,
        realizations,
        phases,
        quantiles=quantiles,
        ells=ells,
        threshold=args.threshold,
    )
    write_outlier_csv(args.csv, results, flagged_only=args.flagged_only)

    flagged = [result for result in results if result["flagged"]]
    print("Scanning xi_qg mock realizations for catastrophic outliers")
    print(f"  mock directory: {args.mock_dir}")
    print(f"  realizations scanned: {len(results)}")
    print(f"  quantiles: {tuple(f'DS{quantile}' for quantile in quantiles)}")
    print(f"  ells: {ells}")
    print(f"  threshold: {args.threshold}")
    print(f"  flagged phases: {len(flagged)}")
    if flagged:
        print("  rerun candidates:")
        for result in flagged:
            print(
                "    "
                f"{result['phase']} score={result['max_panel_score']:.2f} "
                f"panel={result['worst_quantile']} ell={result['worst_ell']} "
                f"s={result['worst_s']:.1f}"
            )
    else:
        print("  rerun candidates: none")
    if args.top is not None:
        print(f"  top {min(args.top, len(results))} phases:")
        for result in results[:args.top]:
            print(
                "    "
                f"{result['phase']} score={result['max_panel_score']:.2f} "
                f"panel={result['worst_quantile']} ell={result['worst_ell']} "
                f"s={result['worst_s']:.1f} flagged={result['flagged']}"
            )
    print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
