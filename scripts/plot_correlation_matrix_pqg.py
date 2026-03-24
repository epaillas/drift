"""Visualize the P_qg correlation matrix for the selected covariance source."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift.covariance import estimate_ssc_sigma_b2
from drift.covariance import plot_correlation_matrix
from drift.io import diagonal_covariance, load_measurements, mock_covariance_matrix
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
COV_DIR = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
DEFAULT_EFFECTIVE_CNG_AMPLITUDE = 0.2
LEGACY_DEFAULT_KMAX = 0.3
DEFAULT_ANALYTIC_KMAX = 0.3
DEFAULT_REFERENCE_REBIN = 5


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


def _parse_quantiles(values):
    if values is None:
        return (1, 5)
    return tuple(int(value) for value in values)


def _representative_mock_path():
    mock_path = next(iter(sorted(COV_DIR.glob("dsc_pkqg_poles_ph*.h5"))), None)
    if mock_path is None:
        raise FileNotFoundError(f"No dsc_pkqg mock files found in {COV_DIR}")
    return mock_path


def _load_mock_k(quantiles, rebin, kmin=0.0, kmax=np.inf):
    k, _ = load_measurements(
        _representative_mock_path(),
        nquantiles=max(quantiles),
        ells=ELLS,
        rebin=rebin,
        kmin=kmin,
        kmax=kmax,
    )
    positive = np.asarray(k)[np.asarray(k) > 0.0]
    if positive.size == 0:
        raise ValueError("Native mock k-grid does not contain any positive modes.")
    return positive


def _legacy_scalar_kmax(values):
    if values is None:
        return None
    parsed = _parse_kmax(values, ELLS)
    unique = {float(val) for val in parsed.values() if np.isfinite(val)}
    if len(unique) > 1:
        raise ValueError("Legacy per-ell --kmax is no longer supported here; use a single scalar cut.")
    return None if not unique else unique.pop()


def _warn_legacy(old_flag, new_flag):
    warnings.warn(
        f"{old_flag} is deprecated; use {new_flag} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_mock_settings(args, quantiles):
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
            kmax = _legacy_scalar_kmax(args.kmax)
    if kmin is None:
        kmin = float(_load_mock_k(quantiles, rebin).min())
    if kmax is None:
        kmax = np.inf
    return {"rebin": rebin, "kmin": float(kmin), "kmax": float(kmax)}


def _infer_default_analytic_dk(quantiles):
    k_ref = _load_mock_k(quantiles, DEFAULT_REFERENCE_REBIN)
    diffs = np.diff(k_ref)
    if diffs.size == 0:
        raise ValueError("Cannot infer analytic dk from a single mock k bin.")
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


def _resolve_analytic_settings(args, quantiles):
    k_ref = _load_mock_k(quantiles, DEFAULT_REFERENCE_REBIN)
    kmin = args.analytic_kmin
    kmax = args.analytic_kmax
    dk = args.analytic_dk

    if args.analytic_cov or args.diag_cov:
        if args.kmin is not None:
            _warn_legacy("--kmin", "--analytic-kmin")
            kmin = args.kmin
        if args.kmax is not None:
            _warn_legacy("--kmax", "--analytic-kmax")
            kmax = _legacy_scalar_kmax(args.kmax)
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
        dk = _infer_default_analytic_dk(quantiles)

    return {
        "kmin": float(kmin),
        "kmax": float(kmax),
        "dk": float(dk),
        "k": _build_linear_k_grid(float(kmin), float(kmax), float(dk)),
    }


def _covariance_source_label(args):
    if args.diag_cov:
        return "diagonal"
    if args.analytic_cov:
        return "analytic"
    return "mock"


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


def _print_covariance_summary(args, labels, block_sizes, k, binning_summary):
    source = _covariance_source_label(args)

    print("Preparing P_qg correlation matrix")
    print(f"  covariance source: {source}")
    print(f"  quantiles: {labels}")
    print(f"  ells: {ELLS}")
    print(f"  nk: {len(k)}")
    print(f"  retained bins per block: {block_sizes}")
    print(f"  covariance rescale: {args.cov_rescale}")
    for line in binning_summary:
        print(f"  {line}")

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
        print(f"  mock directory: {COV_DIR}")
        print("  mock files: dsc_pkqg_poles_ph*.h5")


def _resolve_pqg_covariance(args, k, flat, full_mask, quantiles, labels, fiducials, mock_cfg):
    """Return the selected P_qg covariance source."""
    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    if args.diag_cov:
        return diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)

    if args.analytic_cov:
        return analytic_pqg_covariance(
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
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )

    cov = mock_covariance_matrix(
        COV_DIR,
        "ds",
        ELLS,
        k_data=k,
        mask=full_mask,
        rescale=args.cov_rescale,
        rebin=mock_cfg["rebin"],
        nquantiles=max(quantiles),
        quantiles=quantiles,
        kmin=mock_cfg["kmin"],
        kmax=mock_cfg["kmax"],
    )
    return cov, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH",
                        help="Density-split theory configuration used to build analytic fiducial spectra.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to include, for example --quantiles 1 5.")
    parser.add_argument("--mock-rebin", type=int, default=5, metavar="N",
                        help="Rebin factor used when loading the mock P_qg multipoles.")
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
                        help="Use a diagonal covariance built from the fiducial P_qg data vector instead of mock or analytic covariance.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_qg covariance instead of the mock covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms: 'gaussian', 'gaussian+effective_cng', 'gaussian+ssc', or 'gaussian+effective_cng+ssc'.")
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
                        help="Defaults to 0.2 when effective_cng is enabled, otherwise 0.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected covariance term.")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR",
                        help="Long-mode density variance used by the SSC term on the analytic covariance path.")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    labels = _quantile_labels(quantiles)
    mock_cfg = _resolve_mock_settings(args, quantiles)
    if args.analytic_cov or args.diag_cov:
        analytic_cfg = _resolve_analytic_settings(args, quantiles)
        k = analytic_cfg["k"]
        full_mask = np.ones(len(labels) * len(ELLS) * len(k), dtype=bool)
        block_sizes = [len(k)] * (len(labels) * len(ELLS))
        binning_summary = [
            "binning source: analytic grid",
            f"analytic kmin: {analytic_cfg['kmin']}",
            f"analytic kmax: {analytic_cfg['kmax']}",
            f"analytic dk: {analytic_cfg['dk']}",
            f"analytic nk: {len(k)}",
        ]
    else:
        k = _load_mock_k(quantiles, mock_cfg["rebin"], kmin=mock_cfg["kmin"], kmax=mock_cfg["kmax"])
        full_mask = np.ones(len(labels) * len(ELLS) * len(k), dtype=bool)
        block_sizes = [len(k)] * (len(labels) * len(ELLS))
        binning_summary = [
            "binning source: mock I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock kmin: {mock_cfg['kmin']}",
            f"mock kmax: {mock_cfg['kmax']}",
        ]
    _print_covariance_summary(args, labels, block_sizes, k, binning_summary)

    fiducials = None
    flat = np.zeros(full_mask.shape[0], dtype=float)
    if args.diag_cov or args.analytic_cov:
        fiducials = _build_analytic_dsg_fiducials(args, k, quantiles)
        flat = np.concatenate([
            fiducials["pqg_poles"][label][ell]
            for label in labels
            for ell in ELLS
        ])

    cov, _ = _resolve_pqg_covariance(
        args, k, flat, full_mask, quantiles, labels, fiducials, mock_cfg
    )
    print(f"  covariance matrix shape: {cov.shape}")

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

    out_path = OUTPUT_DIR / f"correlation_matrix_pqg_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
