"""Visualize the xi_gg correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift.covariance import (
    analytic_xigg_covariance,
    plot_correlation_matrix,
    propagate_covariance_to_correlation,
)
from drift.io import build_diagonal_covariance, estimate_mock_covariance
from plot_correlation_matrix_pgg import (
    COV_DIR,
    DEFAULT_ELLS,
    MEAS_PATH,
    OUTPUT_DIR,
    _parse_ells,
    _covariance_source_label,
    _interpolate_measurement_poles,
    _print_covariance_summary as _print_pgg_covariance_summary,
    _resolve_analytic_settings,
    _resolve_cng_amplitude,
    _resolve_mock_settings,
    _resolve_ssc_sigma_b2,
    load_observable_measurements,
)
from plot_correlation_matrix_xi_common import apply_reciprocal_analytic_k_limits, build_s_grid


def _print_covariance_summary(args, k, s, block_sizes, binning_summary, ells):
    _print_pgg_covariance_summary(args, k, block_sizes, binning_summary, ells)
    print(f"  ns: {len(s)}")
    print(f"  smin: {float(s.min())}")
    print(f"  smax: {float(s.max())}")


def _resolve_xigg_covariance(args, k, s, flat, full_mask, poles, mock_cfg, ells):
    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    if args.diag_cov:
        cov_k, _ = build_diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
        return propagate_covariance_to_correlation(cov_k, k, s, ells=ells), None

    if args.analytic_cov:
        return analytic_xigg_covariance(
            k,
            s,
            poles,
            ells=ells,
            volume=args.box_volume,
            number_density=args.number_density,
            shot_noise=args.shot_noise,
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )

    cov_k = estimate_mock_covariance(
        COV_DIR,
        "pgg",
        ells,
        k_data=k,
        mask=full_mask,
        rescale=args.cov_rescale,
        rebin=mock_cfg["rebin"],
        kmin=mock_cfg["kmin"],
        kmax=mock_cfg["kmax"],
    )
    return propagate_covariance_to_correlation(cov_k, k, s, ells=ells), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to plot, for example --ells 0 2 4.")
    parser.add_argument("--mock-rebin", type=int, default=13, metavar="N",
                        help="Rebin factor used when loading the mock or measurement P_gg multipoles that seed the xi_gg covariance.")
    parser.add_argument("--mock-kmin", type=float, default=None, metavar="VALUE",
                        help="Minimum Fourier-space k retained on the mock covariance path, in h/Mpc.")
    parser.add_argument("--mock-kmax", type=float, default=None, metavar="VALUE",
                        help="Maximum Fourier-space k retained on the mock covariance path, in h/Mpc.")
    parser.add_argument("--analytic-kmin", type=float, default=None, metavar="VALUE",
                        help="Minimum Fourier-space k of the analytic grid, in h/Mpc. If omitted, the script infers a rough reciprocal cut from --smax.")
    parser.add_argument("--analytic-kmax", type=float, default=None, metavar="VALUE",
                        help="Maximum Fourier-space k of the analytic grid, in h/Mpc. If omitted, the script infers a rough reciprocal cut from --smin.")
    parser.add_argument("--analytic-dk", type=float, default=None, metavar="VALUE",
                        help="Spacing of the analytic Fourier-space k grid, in h/Mpc.")
    parser.add_argument("--smin", type=float, default=None, metavar="VALUE",
                        help="Minimum separation shown in the xi_gg matrix, in Mpc/h.")
    parser.add_argument("--smax", type=float, default=None, metavar="VALUE",
                        help="Maximum separation shown in the xi_gg matrix, in Mpc/h.")
    parser.add_argument("--ds", type=float, default=None, metavar="VALUE",
                        help="Spacing of the output separation grid, in Mpc/h. Provide at most one of --ds or --ns.")
    parser.add_argument("--ns", type=int, default=None, metavar="N",
                        help="Number of separation bins in the output xi_gg grid. Provide at most one of --ds or --ns.")
    parser.add_argument("--rebin", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--kmin", type=float, default=None, metavar="VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--nk", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--diag-cov", action="store_true",
                        help="Use a diagonal covariance built from the fiducial P_gg data vector before propagating it to xi_gg.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_gg covariance before propagating it to xi_gg.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms to include before propagation: gaussian, gaussian+effective_cng, gaussian+ssc, or gaussian+effective_cng+ssc.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR",
                        help="Divide the covariance matrix by this factor before plotting.")
    parser.add_argument("--box-volume", type=float, default=None, metavar="V",
                        help="Survey or box volume in (Mpc/h)^3 for the analytic covariance path.")
    parser.add_argument("--number-density", type=float, default=None, metavar="N",
                        help="Galaxy number density in (h/Mpc)^3 for the analytic covariance path. Provide exactly one of --number-density or --shot-noise.")
    parser.add_argument("--shot-noise", type=float, default=None, metavar="P0",
                        help="Constant galaxy shot-noise power in (Mpc/h)^3 for the analytic covariance path. Provide exactly one of --number-density or --shot-noise.")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Amplitude of the effective connected non-Gaussian covariance term before propagation to xi_gg.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected covariance term.")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR",
                        help="Long-mode density variance used by the SSC term on the analytic covariance path.")
    args = parser.parse_args()
    ells = _parse_ells(args.ells)

    mock_cfg = _resolve_mock_settings(args, ells)
    if args.analytic_cov or args.diag_cov:
        implied_kmin, implied_kmax = apply_reciprocal_analytic_k_limits(args)
        if args.analytic_kmin is None:
            args.analytic_kmin = implied_kmin
        if args.analytic_kmax is None:
            args.analytic_kmax = implied_kmax
        analytic_cfg = _resolve_analytic_settings(args, ells)
        k = analytic_cfg["k"]
        poles = _interpolate_measurement_poles(k, ells)
        flat = np.concatenate([poles[ell] for ell in ells])
        full_mask = np.ones(len(ells) * len(k), dtype=bool)
        binning_summary = [
            "binning source: analytic grid",
            f"analytic kmin: {analytic_cfg['kmin']}",
            f"analytic kmax: {analytic_cfg['kmax']}",
            f"analytic dk: {analytic_cfg['dk']}",
            f"analytic nk: {len(k)}",
        ]
    else:
        k, _ = load_observable_measurements(
            MEAS_PATH, "pgg", ells=ells, rebin=mock_cfg["rebin"], kmin=mock_cfg["kmin"], kmax=mock_cfg["kmax"]
        )
        poles = None
        flat = np.zeros(len(ells) * len(k), dtype=float)
        full_mask = np.ones(len(ells) * len(k), dtype=bool)
        binning_summary = [
            "binning source: mock/measurement I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock kmin: {mock_cfg['kmin']}",
            f"mock kmax: {mock_cfg['kmax']}",
        ]

    s = build_s_grid(k, smin=args.smin, smax=args.smax, ds=args.ds, ns=args.ns)
    block_sizes = [len(s)] * len(ells)
    _print_covariance_summary(args, k, s, block_sizes, binning_summary, ells)

    cov, _ = _resolve_xigg_covariance(args, k, s, flat, full_mask, poles, mock_cfg, ells)
    print(f"  covariance matrix shape: {cov.shape}")

    fig, _ = plot_correlation_matrix(
        cov,
        ells=ells,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=r"$\xi_{gg}$ correlation",
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"correlation_matrix_xigg_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
