"""Visualize the xi_qg correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift.covariance import (
    analytic_xiqg_covariance,
    plot_correlation_matrix,
    propagate_covariance_to_correlation,
)
from drift.io import (
    build_diagonal_covariance,
    estimate_mock_covariance,
    load_observable_measurements,
)
from plot_correlation_matrix_pqg import (
    CONFIG_PATH,
    DEFAULT_ELLS,
    _parse_ells,
    OUTPUT_DIR,
    _build_analytic_dsg_fiducials,
    _covariance_source_label,
    _ell_block_edges,
    _pair_block_edges,
    _parse_quantiles,
    _quantile_labels,
    _resolve_analytic_settings,
    _resolve_cng_amplitude,
    _resolve_ssc_sigma_b2,
    _default_ds_cross_shot_noise,
    _default_ds_pair_shot_noise,
)
from plot_correlation_matrix_xi_common import apply_reciprocal_analytic_k_limits, build_s_grid

XIQG_MOCK_DIR = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"


def _representative_xiqg_mock_path():
    mock_path = next(iter(sorted(XIQG_MOCK_DIR.glob("dsc_xiqg/dsc_xiqg_poles_ph*.h5"))), None)
    if mock_path is None:
        raise FileNotFoundError(f"No dsc_xiqg mock files found in {XIQG_MOCK_DIR / 'dsc_xiqg'}")
    return mock_path


def _load_mock_s(quantiles, ells, rebin=1, smin=0.0, smax=np.inf):
    s, _ = load_observable_measurements(
        _representative_xiqg_mock_path(),
        "xiqg",
        nquantiles=max(quantiles),
        quantiles=quantiles,
        ells=ells,
        rebin=rebin,
        smin=smin,
        smax=smax,
    )
    return np.asarray(s)


def _print_covariance_summary(args, labels, block_sizes, k, s, binning_summary, ells):
    source = _covariance_source_label(args)
    print("Preparing xi_qg correlation matrix")
    print(f"  covariance source: {source}")
    print(f"  quantiles: {labels}")
    print(f"  ells: {ells}")
    if k is not None:
        print(f"  nk: {len(k)}")
    print(f"  ns: {len(s)}")
    print(f"  retained bins per block: {block_sizes}")
    print(f"  covariance rescale: {args.cov_rescale}")
    for line in binning_summary:
        print(f"  {line}")
    print(f"  smin: {float(s.min())}")
    print(f"  smax: {float(s.max())}")
    if source == "analytic":
        print(f"  analytic terms: {args.analytic_cov_terms}")
        print(f"  box volume: {args.box_volume}")
        print(f"  galaxy shot noise: {args.galaxy_shot_noise}")
        print(f"  DS auto shot noise: {args.ds_pair_auto_shot_noise}")
        print(f"  DS cross shot noise: {args.ds_pair_cross_shot_noise}")
        print(f"  DS-galaxy cross shot noise: {args.ds_cross_shot_noise}")
        print(f"  cng amplitude: {args.cng_amplitude}")
        print(f"  cng coherence: {args.cng_coherence}")
        print(f"  ssc sigma_b^2: {args.ssc_sigma_b2}")
    elif source == "mock":
        print(f"  mock directory: {XIQG_MOCK_DIR / 'dsc_xiqg'}")
        print("  mock files: dsc_xiqg_poles_ph*.h5")


def _resolve_xiqg_covariance(args, k, s, flat, full_mask, quantiles, labels, fiducials, mock_cfg, ells):
    """Return the selected xi_qg covariance source."""
    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    if args.diag_cov:
        cov_k, _ = build_diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
        return propagate_covariance_to_correlation(cov_k, k, s, ells=ells, observable_blocks=len(labels)), None

    if args.analytic_cov:
        return analytic_xiqg_covariance(
            k,
            s,
            fiducials["pqg_poles"],
            fiducials["pqq_poles"],
            fiducials["pgg_poles"],
            ells,
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
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )

    cov_s = estimate_mock_covariance(
        XIQG_MOCK_DIR,
        "xiqg",
        ells,
        s_data=s,
        rescale=args.cov_rescale,
        rebin=mock_cfg["rebin"],
        nquantiles=max(quantiles),
        quantiles=quantiles,
        smin=mock_cfg["smin"],
        smax=mock_cfg["smax"],
    )
    return cov_s, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH",
                        help="Density-split theory configuration used to build analytic fiducial spectra.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to include in the xi_qg matrix.")
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to plot, for example --ells 0 2.")
    parser.add_argument("--mock-rebin", type=int, default=1, metavar="N",
                        help="Keep every Nth separation bin when loading direct xi_qg mocks.")
    parser.add_argument("--mock-smin", type=float, default=None, metavar="VALUE",
                        help="Minimum separation retained on the direct xi_qg mock covariance path, in Mpc/h.")
    parser.add_argument("--mock-smax", type=float, default=None, metavar="VALUE",
                        help="Maximum separation retained on the direct xi_qg mock covariance path, in Mpc/h.")
    parser.add_argument("--analytic-kmin", type=float, default=None, metavar="VALUE",
                        help="Minimum Fourier-space k of the analytic grid, in h/Mpc. If omitted, the script infers a rough reciprocal cut from --smax.")
    parser.add_argument("--analytic-kmax", type=float, default=None, metavar="VALUE",
                        help="Maximum Fourier-space k of the analytic grid, in h/Mpc. If omitted, the script infers a rough reciprocal cut from --smin.")
    parser.add_argument("--analytic-dk", type=float, default=None, metavar="VALUE",
                        help="Spacing of the analytic Fourier-space k grid, in h/Mpc.")
    parser.add_argument("--smin", type=float, default=None, metavar="VALUE",
                        help="Minimum separation shown in the xi_qg matrix, in Mpc/h.")
    parser.add_argument("--smax", type=float, default=None, metavar="VALUE",
                        help="Maximum separation shown in the xi_qg matrix, in Mpc/h.")
    parser.add_argument("--ds", type=float, default=None, metavar="VALUE",
                        help="Spacing of the output separation grid, in Mpc/h. Provide at most one of --ds or --ns.")
    parser.add_argument("--ns", type=int, default=None, metavar="N",
                        help="Number of separation bins in the output xi_qg grid. Provide at most one of --ds or --ns.")
    parser.add_argument("--rebin", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--kmin", type=float, default=None, metavar="VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--nk", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--diag-cov", action="store_true",
                        help="Use a diagonal covariance built from the fiducial P_qg data vector before propagating it to xi_qg.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_qg covariance before propagating it to xi_qg.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms to include before propagation: gaussian, gaussian+effective_cng, gaussian+ssc, or gaussian+effective_cng+ssc.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR",
                        help="Divide the covariance matrix by this factor before plotting.")
    parser.add_argument("--box-volume", type=float, default=1.0e9, metavar="V",
                        help="Survey or box volume in (Mpc/h)^3 for the analytic covariance path.")
    parser.add_argument("--galaxy-shot-noise", type=float, default=400.0, metavar="P0",
                        help="Constant galaxy shot-noise power entering the analytic P_qg covariance, in (Mpc/h)^3.")
    parser.add_argument("--ds-pair-auto-shot-noise", type=float, default=250.0, metavar="PQQ0",
                        help="Constant DS auto-pair shot noise entering the analytic P_qg covariance, in (Mpc/h)^3.")
    parser.add_argument("--ds-pair-cross-shot-noise", type=float, default=40.0, metavar="PQQX",
                        help="Constant DS cross-pair shot noise entering the analytic P_qg covariance, in (Mpc/h)^3.")
    parser.add_argument("--ds-cross-shot-noise", type=float, default=0.0, metavar="PQG0",
                        help="Constant DS-galaxy cross shot-noise term entering the analytic P_qg covariance, in (Mpc/h)^3.")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Amplitude of the effective connected non-Gaussian covariance term before propagation to xi_qg.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected covariance term.")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR",
                        help="Long-mode density variance used by the SSC term on the analytic covariance path.")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    ells = _parse_ells(args.ells)
    labels = _quantile_labels(quantiles)
    if args.analytic_cov or args.diag_cov:
        mock_cfg = None
        implied_kmin, implied_kmax = apply_reciprocal_analytic_k_limits(args)
        if args.analytic_kmin is None:
            args.analytic_kmin = implied_kmin
        if args.analytic_kmax is None:
            args.analytic_kmax = implied_kmax
        analytic_cfg = _resolve_analytic_settings(args, quantiles, ells)
        k = analytic_cfg["k"]
        full_mask = np.ones(len(labels) * len(ells) * len(k), dtype=bool)
        binning_summary = [
            "binning source: analytic grid",
            f"analytic kmin: {analytic_cfg['kmin']}",
            f"analytic kmax: {analytic_cfg['kmax']}",
            f"analytic dk: {analytic_cfg['dk']}",
            f"analytic nk: {len(k)}",
        ]
        s = build_s_grid(k, smin=args.smin, smax=args.smax, ds=args.ds, ns=args.ns)
    else:
        k = None
        full_mask = None
        rebin = args.mock_rebin
        if args.rebin is not None:
            rebin = args.rebin
        smin = args.mock_smin if args.mock_smin is not None else (args.smin if args.smin is not None else 0.0)
        smax = args.mock_smax if args.mock_smax is not None else (args.smax if args.smax is not None else np.inf)
        s = _load_mock_s(quantiles, ells, rebin=rebin, smin=smin, smax=smax)
        mock_cfg = {"rebin": int(rebin), "smin": float(smin), "smax": float(smax)}
        binning_summary = [
            "binning source: mock I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock smin: {mock_cfg['smin']}",
            f"mock smax: {mock_cfg['smax']}",
        ]
    block_sizes = [len(s)] * (len(labels) * len(ells))
    _print_covariance_summary(args, labels, block_sizes, k, s, binning_summary, ells)

    fiducials = None
    flat = np.zeros(0, dtype=float)
    if args.diag_cov or args.analytic_cov:
        fiducials = _build_analytic_dsg_fiducials(args, k, quantiles, ells=ells)
        flat = np.concatenate([
            fiducials["pqg_poles"][label][ell]
            for label in labels
            for ell in ells
        ])

    cov, _ = _resolve_xiqg_covariance(
        args, k, s, flat, full_mask, quantiles, labels, fiducials, mock_cfg, ells
    )
    print(f"  covariance matrix shape: {cov.shape}")

    label_str = ", ".join(labels)
    fig, ax = plot_correlation_matrix(
        cov,
        ells=ells,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=rf"$\xi_{{qg}}$ correlation ({label_str})",
    )

    for edge in _ell_block_edges(block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.35, alpha=0.25)
        ax.axhline(edge - 0.5, color="k", lw=0.35, alpha=0.25)

    for edge in _pair_block_edges(labels, block_sizes, ells):
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.8)
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"correlation_matrix_xiqg_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
