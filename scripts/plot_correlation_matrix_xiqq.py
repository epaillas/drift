"""Visualize the xi_qiqj correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift.covariance import (
    analytic_xiqq_covariance,
    plot_correlation_matrix,
    propagate_covariance_to_correlation,
)
from drift.io import build_diagonal_covariance, estimate_mock_covariance
from plot_correlation_matrix_pqq import (
    CONFIG_PATH,
    _parse_ells,
    OUTPUT_DIR,
    COV_DIR,
    _build_default_shot_noise,
    _build_pair_order,
    _compute_fiducial_pair_poles,
    _covariance_source_label,
    _ell_block_edges,
    _filter_auto_pairs,
    _load_mock_k,
    _pair_block_edges,
    _parse_quantiles,
    _print_covariance_summary as _print_pqq_covariance_summary,
    _resolve_analytic_settings,
    _resolve_cng_amplitude,
    _resolve_mock_settings,
    _resolve_ssc_sigma_b2,
    get_cosmology,
    load_config,
)
from plot_correlation_matrix_xi_common import apply_reciprocal_analytic_k_limits, build_s_grid


def _print_covariance_summary(args, quantiles, pair_order, block_sizes, k, s, binning_summary, ells):
    _print_pqq_covariance_summary(args, quantiles, pair_order, block_sizes, k, binning_summary, ells)
    print(f"  ns: {len(s)}")
    print(f"  smin: {float(s.min())}")
    print(f"  smax: {float(s.max())}")


def _resolve_xiqq_covariance(
    args,
    k,
    s,
    flat,
    full_mask,
    quantiles,
    pair_order,
    shot_noise,
    fiducial_poles,
    mock_cfg,
    ells,
):
    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    if args.diag_cov:
        cov_k, _ = build_diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
        return propagate_covariance_to_correlation(cov_k, k, s, ells=ells, observable_blocks=len(pair_order)), None

    if args.analytic_cov:
        return analytic_xiqq_covariance(
            k,
            s,
            fiducial_poles,
            ells=ells,
            volume=args.box_volume,
            pair_order=pair_order,
            shot_noise=shot_noise,
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )

    if not args.autos_only:
        raise ValueError(
            "Mock xi_qq covariance is only available for auto pairs. "
            "Pass --autos-only or use --analytic-cov."
        )

    cov_k = estimate_mock_covariance(
        COV_DIR,
        "pqq_auto",
        ells,
        k_data=k,
        mask=full_mask,
        rescale=args.cov_rescale,
        rebin=mock_cfg["rebin"],
        nquantiles=max(quantiles),
        quantiles=quantiles,
        kmin=mock_cfg["kmin"],
        kmax=mock_cfg["kmax"],
    )
    return propagate_covariance_to_correlation(cov_k, k, s, ells=ells, observable_blocks=len(pair_order)), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH",
                        help="Density-split theory configuration used to build analytic fiducial spectra.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to include when building DS-pair xi_qq blocks.")
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to plot, for example --ells 0 2.")
    parser.add_argument("--autos-only", action="store_true",
                        help="Restrict the plotted xi_qq blocks to DS auto pairs only.")
    parser.add_argument("--ds-model", type=str, default="baseline",
                        choices=("baseline", "rsd_selection", "phenomenological"),
                        help="Density-split model used to build the analytic fiducial P_qq spectra before propagation.")
    parser.add_argument("--mock-rebin", type=int, default=5, metavar="N",
                        help="Rebin factor used when loading the mock P_qq multipoles that seed the xi_qq covariance.")
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
                        help="Minimum separation shown in the xi_qq matrix, in Mpc/h.")
    parser.add_argument("--smax", type=float, default=None, metavar="VALUE",
                        help="Maximum separation shown in the xi_qq matrix, in Mpc/h.")
    parser.add_argument("--ds", type=float, default=None, metavar="VALUE",
                        help="Spacing of the output separation grid, in Mpc/h. Provide at most one of --ds or --ns.")
    parser.add_argument("--ns", type=int, default=None, metavar="N",
                        help="Number of separation bins in the output xi_qq grid. Provide at most one of --ds or --ns.")
    parser.add_argument("--rebin", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--kmin", type=float, default=None, metavar="VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--nk", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--diag-cov", action="store_true",
                        help="Use a diagonal covariance built from the fiducial P_qq data vector before propagating it to xi_qq.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_qq covariance before propagating it to xi_qq.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms to include before propagation: gaussian, gaussian+effective_cng, gaussian+ssc, or gaussian+effective_cng+ssc.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR",
                        help="Divide the covariance matrix by this factor before plotting.")
    parser.add_argument("--box-volume", type=float, default=1.0e9, metavar="V",
                        help="Survey or box volume in (Mpc/h)^3 for the analytic covariance path.")
    parser.add_argument("--auto-shot-noise", type=float, default=250.0, metavar="P0",
                        help="Constant shot-noise power assigned to DS auto pairs in the analytic P_qq covariance, in (Mpc/h)^3.")
    parser.add_argument("--cross-shot-noise", type=float, default=40.0, metavar="P0X",
                        help="Constant shot-noise power assigned to DS cross pairs in the analytic P_qq covariance, in (Mpc/h)^3.")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Amplitude of the effective connected non-Gaussian covariance term before propagation to xi_qq.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected covariance term.")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR",
                        help="Long-mode density variance used by the SSC term on the analytic covariance path.")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    ells = _parse_ells(args.ells)
    full_pair_order = _build_pair_order(quantiles)
    pair_order = _filter_auto_pairs(full_pair_order, args.autos_only)
    shot_noise = _build_default_shot_noise(
        full_pair_order, args.auto_shot_noise, args.cross_shot_noise
    )

    cfg = load_config(args.config)
    cosmo = get_cosmology({
        "h": cfg.cosmo.h,
        "Omega_m": cfg.cosmo.Omega_m,
        "Omega_b": cfg.cosmo.Omega_b,
        "sigma8": cfg.cosmo.sigma8,
        "n_s": cfg.cosmo.n_s,
        "engine": cfg.cosmo.engine,
    })

    mock_cfg = _resolve_mock_settings(args, quantiles, ells)
    if args.analytic_cov or args.diag_cov:
        implied_kmin, implied_kmax = apply_reciprocal_analytic_k_limits(args)
        if args.analytic_kmin is None:
            args.analytic_kmin = implied_kmin
        if args.analytic_kmax is None:
            args.analytic_kmax = implied_kmax
        analytic_cfg = _resolve_analytic_settings(args, quantiles, ells)
        k = analytic_cfg["k"]
        poles = _compute_fiducial_pair_poles(cfg, cosmo, k, full_pair_order, args.ds_model, ells)
        flat = np.concatenate([poles[pair][ell] for pair in pair_order for ell in ells])
        full_mask = np.ones(len(pair_order) * len(ells) * len(k), dtype=bool)
        binning_summary = [
            "binning source: analytic grid",
            f"analytic kmin: {analytic_cfg['kmin']}",
            f"analytic kmax: {analytic_cfg['kmax']}",
            f"analytic dk: {analytic_cfg['dk']}",
            f"analytic nk: {len(k)}",
        ]
    else:
        k = _load_mock_k(quantiles, ells, mock_cfg["rebin"], kmin=mock_cfg["kmin"], kmax=mock_cfg["kmax"])
        poles = {}
        flat = np.zeros(len(pair_order) * len(ells) * len(k), dtype=float)
        full_mask = np.ones(len(pair_order) * len(ells) * len(k), dtype=bool)
        binning_summary = [
            "binning source: mock I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock kmin: {mock_cfg['kmin']}",
            f"mock kmax: {mock_cfg['kmax']}",
        ]

    s = build_s_grid(k, smin=args.smin, smax=args.smax, ds=args.ds, ns=args.ns)
    block_sizes = [len(s)] * (len(pair_order) * len(ells))
    _print_covariance_summary(args, quantiles, pair_order, block_sizes, k, s, binning_summary, ells)

    cov, _ = _resolve_xiqq_covariance(
        args,
        k,
        s,
        flat,
        full_mask,
        quantiles,
        pair_order,
        shot_noise,
        poles,
        mock_cfg,
        ells,
    )
    print(f"  covariance matrix shape: {cov.shape}")

    pair_str = ", ".join(f"{a}-{b}" for a, b in pair_order)
    fig, ax = plot_correlation_matrix(
        cov,
        ells=ells,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=rf"$\xi_{{qq}}$ correlation ({args.ds_model}; {pair_str})",
    )

    for edge in _ell_block_edges(block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.35, alpha=0.25)
        ax.axhline(edge - 0.5, color="k", lw=0.35, alpha=0.25)

    for edge in _pair_block_edges(pair_order, block_sizes, ells):
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.8)
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"correlation_matrix_xiqq_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
