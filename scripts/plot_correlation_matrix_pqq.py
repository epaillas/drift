"""Visualize the P_qiqj correlation matrix for the selected covariance source."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import estimate_ssc_sigma_b2
from drift.covariance import plot_correlation_matrix
from drift.io import (
    analytic_pqq_covariance,
    build_diagonal_covariance,
    estimate_mock_covariance,
    load_observable_measurements,
)
from drift.theory.density_split.config import load_config
from drift.theory.density_split.power_spectrum import pqq_mu
from drift.utils.cosmology import get_cosmology
from drift.utils.multipoles import compute_multipoles

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
COV_DIR = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
DEFAULT_ELLS = (0, 2)
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
        return (1, 2)
    return tuple(int(value) for value in values)


def _parse_ells(values):
    if values is None:
        return DEFAULT_ELLS
    ells = tuple(int(value) for value in values)
    if not ells:
        raise ValueError("--ells must contain at least one multipole.")
    if len(set(ells)) != len(ells):
        raise ValueError("--ells must not contain duplicates.")
    return ells


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


def _representative_mock_path():
    mock_path = next(iter(sorted(COV_DIR.glob("dsc_pkqq_poles_ph*.h5"))), None)
    if mock_path is None:
        raise FileNotFoundError(f"No dsc_pkqq mock files found in {COV_DIR}")
    return mock_path


def _load_mock_k(quantiles, ells, rebin, kmin=0.0, kmax=np.inf):
    k, _ = load_observable_measurements(
        _representative_mock_path(),
        "pqg",
        nquantiles=max(quantiles),
        ells=ells,
        rebin=rebin,
        kmin=kmin,
        kmax=kmax,
    )
    positive = np.asarray(k)[np.asarray(k) > 0.0]
    if positive.size == 0:
        raise ValueError("Mock k-grid does not contain any positive modes.")
    return positive


def _infer_default_analytic_dk(quantiles, ells):
    k_ref = _load_mock_k(quantiles, ells, DEFAULT_REFERENCE_REBIN)
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


def _resolve_mock_settings(args, quantiles, ells):
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
        kmin = float(_load_mock_k(quantiles, ells, rebin).min())
    if kmax is None:
        kmax = np.inf
    return {"rebin": rebin, "kmin": float(kmin), "kmax": float(kmax)}


def _resolve_analytic_settings(args, quantiles, ells):
    k_ref = _load_mock_k(quantiles, ells, DEFAULT_REFERENCE_REBIN)
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
        dk = _infer_default_analytic_dk(quantiles, ells)

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


def _build_pair_order(quantiles):
    pair_order = []
    for i, q1 in enumerate(quantiles):
        for q2 in quantiles[i:]:
            pair_order.append((f"DS{q1}", f"DS{q2}"))
    return tuple(pair_order)


def _filter_auto_pairs(pair_order, autos_only):
    if not autos_only:
        return pair_order
    return tuple(pair for pair in pair_order if pair[0] == pair[1])


def _build_default_shot_noise(pair_order, auto_shot, cross_shot):
    shot = {}
    for pair in pair_order:
        shot[pair] = auto_shot if pair[0] == pair[1] else cross_shot
    return shot


def _pair_block_edges(pair_order, block_sizes, ells):
    if len(pair_order) * len(ells) != len(block_sizes):
        raise ValueError("block_sizes must contain one entry per pair-ell block.")
    edges = []
    offset = 0
    for idx in range(len(pair_order)):
        pair_size = sum(block_sizes[idx * len(ells):(idx + 1) * len(ells)])
        offset += pair_size
        edges.append(offset)
    return edges[:-1]


def _ell_block_edges(block_sizes):
    edges = []
    offset = 0
    for size in block_sizes:
        offset += size
        edges.append(offset)
    return edges[:-1]


def _compute_fiducial_pair_poles(cfg, cosmo, k, pair_order, ds_model, ells):
    bins_by_label = {ds_bin.label: ds_bin for ds_bin in cfg.split_bins}
    poles = {}
    for label_a, label_b in pair_order:
        ds_a = bins_by_label[label_a]
        ds_b = bins_by_label[label_b]

        def model(kk, mu, _a=ds_a, _b=ds_b):
            return pqq_mu(
                kk, mu, cfg.z, cosmo, _a, _b, cfg.R,
                kernel=cfg.kernel, ds_model=ds_model,
            )

        poles[(label_a, label_b)] = compute_multipoles(k, model, ells=ells)
    return poles


def _resolve_pqq_covariance(
    args,
    k,
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
        return build_diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)

    if args.analytic_cov:
        return analytic_pqq_covariance(
            k,
            fiducial_poles,
            ells=ells,
            volume=args.box_volume,
            pair_order=pair_order,
            shot_noise=shot_noise,
            mask=full_mask,
            rescale=args.cov_rescale,
            terms=args.analytic_cov_terms,
            cng_amplitude=_resolve_cng_amplitude(args),
            cng_coherence=args.cng_coherence,
            ssc_sigma_b2=_resolve_ssc_sigma_b2(args),
        )

    if not args.autos_only:
        raise ValueError(
            "Mock P_qq covariance is only available for auto pairs. "
            "Pass --autos-only or use --analytic-cov."
        )

    cov = estimate_mock_covariance(
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
    return cov, None


def _print_covariance_summary(args, quantiles, pair_order, block_sizes, k, binning_summary, ells):
    source = _covariance_source_label(args)

    print("Preparing P_qq correlation matrix")
    print(f"  covariance source: {source}")
    print(f"  quantiles: {quantiles}")
    print(f"  pair order: {pair_order}")
    print(f"  autos only: {args.autos_only}")
    print(f"  ds model: {args.ds_model}")
    print(f"  ells: {ells}")
    print(f"  nk: {len(k)}")
    print(f"  retained bins per block: {block_sizes}")
    print(f"  covariance rescale: {args.cov_rescale}")
    for line in binning_summary:
        print(f"  {line}")

    if source == "analytic":
        print(f"  analytic terms: {args.analytic_cov_terms}")
        print(f"  box volume: {args.box_volume}")
        print(f"  auto shot noise: {args.auto_shot_noise}")
        print(f"  cross shot noise: {args.cross_shot_noise}")
        print(f"  cng amplitude: {args.cng_amplitude}")
        print(f"  cng coherence: {args.cng_coherence}")
        print(f"  ssc sigma_b^2: {args.ssc_sigma_b2}")
    elif source == "mock":
        print(f"  mock directory: {COV_DIR}")
        print("  mock files: dsc_pkqq_poles_ph*.h5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH",
                        help="Density-split theory configuration used to build analytic fiducial spectra.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to include when building DS-pair blocks.")
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to plot, for example --ells 0 2.")
    parser.add_argument("--autos-only", action="store_true",
                        help="Plot only auto-spectrum DS blocks (e.g. DS1-DS1, DS5-DS5).")
    parser.add_argument("--ds-model", type=str, default="baseline",
                        choices=("baseline", "rsd_selection", "phenomenological"),
                        help="Density-split model used to build the analytic fiducial P_qq spectra.")
    parser.add_argument("--mock-rebin", type=int, default=5, metavar="N",
                        help="Rebin factor used when loading the mock P_qq multipoles.")
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
                        help="Use a diagonal covariance built from the fiducial P_qq data vector instead of mock or analytic covariance.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use the analytic cubic-box P_qq covariance instead of the mock covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms: 'gaussian', 'gaussian+effective_cng', 'gaussian+ssc', or 'gaussian+effective_cng+ssc'.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR",
                        help="Divide the covariance matrix by this factor before plotting.")
    parser.add_argument("--box-volume", type=float, default=1.0e9, metavar="V",
                        help="Survey or box volume in (Mpc/h)^3 for the analytic covariance path.")
    parser.add_argument("--auto-shot-noise", type=float, default=250.0, metavar="P0",
                        help="Constant shot-noise power assigned to DS auto pairs in the analytic P_qq covariance, in (Mpc/h)^3.")
    parser.add_argument("--cross-shot-noise", type=float, default=40.0, metavar="P0X",
                        help="Constant shot-noise power assigned to DS cross pairs in the analytic P_qq covariance, in (Mpc/h)^3.")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Defaults to 0.2 when effective_cng is enabled, otherwise 0.")
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
        analytic_cfg = _resolve_analytic_settings(args, quantiles, ells)
        k = analytic_cfg["k"]
        poles = _compute_fiducial_pair_poles(cfg, cosmo, k, full_pair_order, args.ds_model, ells)
        full_mask = np.ones(len(pair_order) * len(ells) * len(k), dtype=bool)
        flat = np.concatenate([poles[pair][ell] for pair in pair_order for ell in ells])
        block_sizes = [len(k)] * (len(pair_order) * len(ells))
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
        full_mask = np.ones(len(pair_order) * len(ells) * len(k), dtype=bool)
        flat = np.zeros(full_mask.sum(), dtype=float)
        block_sizes = [len(k)] * (len(pair_order) * len(ells))
        binning_summary = [
            "binning source: mock I/O",
            f"mock rebin: {mock_cfg['rebin']}",
            f"mock kmin: {mock_cfg['kmin']}",
            f"mock kmax: {mock_cfg['kmax']}",
        ]

    _print_covariance_summary(args, quantiles, pair_order, block_sizes, k, binning_summary, ells)

    cov, _ = _resolve_pqq_covariance(
        args,
        k,
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
        title=rf"$P_{{qq}}$ correlation ({args.ds_model}; {pair_str})",
    )

    for edge in _ell_block_edges(block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.35, alpha=0.25)
        ax.axhline(edge - 0.5, color="k", lw=0.35, alpha=0.25)

    for edge in _pair_block_edges(pair_order, block_sizes, ells):
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.8)
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"correlation_matrix_pqq_{_covariance_source_label(args)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
