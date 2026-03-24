"""Visualize the P_gg correlation matrix for the selected covariance source."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import estimate_ssc_sigma_b2, plot_correlation_matrix
from drift.io import (
    analytic_pgg_covariance,
    diagonal_covariance,
    load_pgg_measurements,
    mock_covariance_matrix,
)
from inference_pgg import ELLS, MEAS_PATH, MODEL_MODE, OUTPUT_DIR, SPACE

COV_DIR = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
DEFAULT_EFFECTIVE_CNG_AMPLITUDE = 0.2
LEGACY_DEFAULT_KMAX = 0.5
DEFAULT_ANALYTIC_KMAX = 0.3
DEFAULT_REFERENCE_REBIN = 13


def _resolve_cng_amplitude(args):
    amplitude = getattr(args, "cng_amplitude", None)
    if amplitude is not None:
        return float(amplitude)
    if "effective_cng" in str(getattr(args, "analytic_cov_terms", "gaussian")).lower():
        return DEFAULT_EFFECTIVE_CNG_AMPLITUDE
    return 0.0


def _resolve_ssc_sigma_b2(args):
    sigma_b2 = getattr(args, "ssc_sigma_b2", None)
    if sigma_b2 is not None:
        return float(sigma_b2)
    if "ssc" in str(getattr(args, "analytic_cov_terms", "gaussian")).lower():
        return estimate_ssc_sigma_b2(args.box_volume, z=0.5)
    return None


def _parse_legacy_kmax(values):
    if values is None:
        return None
    if len(values) == 1 and ":" not in values[0]:
        return float(values[0])
    raise ValueError("Per-ell legacy --kmax is not supported; use a single scalar cut.")


def _warn_legacy(old_flag, new_flag):
    warnings.warn(
        f"{old_flag} is deprecated; use {new_flag} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _infer_default_analytic_dk():
    k_ref, _ = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=DEFAULT_REFERENCE_REBIN, kmin=0.0)
    diffs = np.diff(k_ref)
    if diffs.size == 0:
        raise ValueError("Cannot infer analytic dk from a single measurement k bin.")
    return float(np.median(diffs))


def _build_linear_k_grid(kmin, kmax, dk):
    if dk <= 0.0:
        raise ValueError("--analytic-dk must be positive.")
    if kmax <= kmin:
        raise ValueError("--analytic-kmax must be larger than --analytic-kmin.")
    n = int(np.floor((kmax - kmin) / dk + 1e-12)) + 1
    grid = kmin + np.arange(n) * dk
    if grid[-1] < kmax - 1e-9:
        grid = np.append(grid, kmax)
    return grid


def _resolve_mock_settings(args):
    rebin = args.mock_rebin
    kmin = args.mock_kmin
    kmax = args.mock_kmax

    if args.rebin is not None:
        _warn_legacy("--rebin", "--mock-rebin")
        rebin = args.rebin
    if not args.analytic_cov and not args.diag_cov:
        if args.kmin is not None:
            _warn_legacy("--kmin", "--mock-kmin")
            kmin = args.kmin
        if args.kmax is not None:
            _warn_legacy("--kmax", "--mock-kmax")
            kmax = _parse_legacy_kmax(args.kmax)

    if kmin is None:
        k0, _ = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=rebin, kmin=0.0)
        kmin = float(k0.min())
    if kmax is None:
        kmax = np.inf
    return {"rebin": rebin, "kmin": float(kmin), "kmax": float(kmax)}


def _resolve_analytic_settings(args):
    k_ref, _ = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=DEFAULT_REFERENCE_REBIN, kmin=0.0)
    kmin = args.analytic_kmin
    kmax = args.analytic_kmax
    dk = args.analytic_dk

    if args.analytic_cov or args.diag_cov:
        if args.kmin is not None:
            _warn_legacy("--kmin", "--analytic-kmin")
            kmin = args.kmin
        if args.kmax is not None:
            _warn_legacy("--kmax", "--analytic-kmax")
            kmax = _parse_legacy_kmax(args.kmax)
        if args.nk is not None:
            _warn_legacy("--nk", "--analytic-dk")
            legacy_kmin = float(kmin if kmin is not None else k_ref.min())
            legacy_kmax = float(kmax if kmax is not None else DEFAULT_ANALYTIC_KMAX)
            if args.nk < 2:
                raise ValueError("Legacy --nk must be at least 2.")
            dk = (legacy_kmax - legacy_kmin) / (args.nk - 1)

    if kmin is None:
        kmin = float(k_ref.min())
    if kmax is None:
        kmax = DEFAULT_ANALYTIC_KMAX
    if dk is None:
        dk = _infer_default_analytic_dk()

    return {
        "kmin": float(kmin),
        "kmax": float(kmax),
        "dk": float(dk),
        "k": _build_linear_k_grid(float(kmin), float(kmax), float(dk)),
    }


def _interpolate_measurement_poles(k_target):
    k_src, poles_src = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=DEFAULT_REFERENCE_REBIN, kmin=0.0)
    return {
        ell: np.interp(k_target, k_src, poles_src[ell])
        for ell in ELLS
    }


def _covariance_source_label(args):
    if args.diag_cov:
        return "diagonal"
    if args.analytic_cov:
        return "analytic"
    return "mock"


def _print_covariance_summary(args, k, block_sizes, binning_summary):
    source = _covariance_source_label(args)
    print("Preparing P_gg correlation matrix")
    print(f"  covariance source: {source}")
    print(f"  measurement file: {MEAS_PATH}")
    print(f"  ells: {ELLS}")
    print(f"  nk: {len(k)}")
    print(f"  retained bins per ell: {dict(zip(ELLS, block_sizes))}")
    print(f"  covariance rescale: {args.cov_rescale}")
    for line in binning_summary:
        print(f"  {line}")
    if source == "analytic":
        print(f"  analytic terms: {args.analytic_cov_terms}")
        print(f"  box volume: {args.box_volume}")
        print(f"  number density: {args.number_density}")
        print(f"  shot noise: {args.shot_noise}")
        print(f"  cng amplitude: {args.cng_amplitude}")
        print(f"  cng coherence: {args.cng_coherence}")
        print(f"  ssc sigma_b^2: {args.ssc_sigma_b2}")
    elif source == "mock":
        print(f"  mock directory: {COV_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-rebin", type=int, default=13, metavar="N",
                        help="Rebin factor used when loading the mock or measurement P_gg multipoles.")
    parser.add_argument("--mock-kmin", type=float, default=None, metavar="VALUE",
                        help="Minimum k retained on the mock covariance path, in h/Mpc.")
    parser.add_argument("--mock-kmax", type=float, default=None, metavar="VALUE",
                        help="Maximum k retained on the mock covariance path, in h/Mpc.")
    parser.add_argument("--analytic-kmin", type=float, default=None, metavar="VALUE",
                        help="Minimum k of the linear analytic grid, in h/Mpc.")
    parser.add_argument("--analytic-kmax", type=float, default=None, metavar="VALUE",
                        help="Maximum k of the linear analytic grid, in h/Mpc.")
    parser.add_argument("--analytic-dk", type=float, default=None, metavar="VALUE",
                        help="Spacing of the analytic k grid, in h/Mpc.")
    parser.add_argument("--rebin", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--kmin", type=float, default=None, metavar="VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--nk", type=int, default=None, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--diag-cov", action="store_true",
                        help="Use a diagonal covariance built from the fiducial data vector instead of mock or analytic covariance.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_gg covariance instead of the mock covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms to include: gaussian, gaussian+effective_cng, gaussian+ssc, or gaussian+effective_cng+ssc.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR",
                        help="Divide the covariance matrix by this factor before plotting.")
    parser.add_argument("--box-volume", type=float, default=None, metavar="V",
                        help="Survey or box volume in (Mpc/h)^3 for the analytic covariance path.")
    parser.add_argument("--number-density", type=float, default=None, metavar="N",
                        help="Galaxy number density in (h/Mpc)^3 for the analytic covariance path. Provide exactly one of --number-density or --shot-noise.")
    parser.add_argument("--shot-noise", type=float, default=None, metavar="P0",
                        help="Constant galaxy shot-noise power in (Mpc/h)^3 for the analytic covariance path. Provide exactly one of --number-density or --shot-noise.")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Amplitude of the effective connected non-Gaussian covariance term.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected covariance term.")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR",
                        help="Long-mode density variance used by the SSC term on the analytic covariance path.")
    args = parser.parse_args()

    mock_cfg = _resolve_mock_settings(args)
    if args.analytic_cov or args.diag_cov:
        analytic_cfg = _resolve_analytic_settings(args)
        k = analytic_cfg["k"]
        poles = _interpolate_measurement_poles(k)
        flat = np.concatenate([poles[ell] for ell in ELLS])
        full_mask = np.ones(len(ELLS) * len(k), dtype=bool)
        block_sizes = [len(k)] * len(ELLS)
        binning_summary = [
            "binning source: analytic grid",
            f"analytic kmin: {analytic_cfg['kmin']}",
            f"analytic kmax: {analytic_cfg['kmax']}",
            f"analytic dk: {analytic_cfg['dk']}",
            f"analytic nk: {len(k)}",
        ]
    else:
        k, poles = load_pgg_measurements(
            MEAS_PATH, ells=ELLS, rebin=mock_cfg["rebin"], kmin=mock_cfg["kmin"], kmax=mock_cfg["kmax"]
        )
        flat = np.concatenate([poles[ell] for ell in ELLS])
        full_mask = np.ones(len(ELLS) * len(k), dtype=bool)
        block_sizes = [len(k)] * len(ELLS)
        binning_summary = [
            "binning source: mock/measurement I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock kmin: {mock_cfg['kmin']}",
            f"mock kmax: {mock_cfg['kmax']}",
        ]

    _print_covariance_summary(args, k, block_sizes, binning_summary)

    if args.diag_cov:
        cov, _ = diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
    elif args.analytic_cov:
        cov, _ = analytic_pgg_covariance(
            k,
            {ell: flat[i * len(k):(i + 1) * len(k)] for i, ell in enumerate(ELLS)},
            ELLS,
            volume=args.box_volume,
            number_density=args.number_density,
            shot_noise=args.shot_noise,
            mask=full_mask,
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )
    else:
        cov = mock_covariance_matrix(
            COV_DIR,
            "pgg",
            ELLS,
            k_data=k,
            mask=full_mask,
            rescale=args.cov_rescale,
            rebin=mock_cfg["rebin"],
            kmin=mock_cfg["kmin"],
            kmax=mock_cfg["kmax"],
        )
    print(f"  covariance matrix shape: {cov.shape}")

    fig, ax = plot_correlation_matrix(
        cov,
        ells=ELLS,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=rf"$P_{{gg}}$ correlation ({SPACE}, {MODEL_MODE})",
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"correlation_matrix_pgg_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
