"""Visualize the P_qiqj correlation matrix for the selected covariance source."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.covariance import estimate_ssc_sigma_b2
from drift.covariance import plot_correlation_matrix
from drift.io import diagonal_covariance, analytic_pqq_covariance
from drift.theory.density_split.config import load_config
from drift.theory.density_split.power_spectrum import pqq_mu
from drift.utils.cosmology import get_cosmology
from drift.utils.multipoles import compute_multipoles

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
ELLS = (0, 2)
DEFAULT_EFFECTIVE_CNG_AMPLITUDE = 0.2


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


def _parse_kmax(values):
    kmax = {ell: 0.3 for ell in ELLS}
    if values is None:
        return kmax
    if len(values) == 1 and ":" not in values[0]:
        val = float(values[0])
        return {ell: val for ell in ELLS}
    for item in values:
        ell_str, val_str = item.split(":")
        kmax[int(ell_str)] = float(val_str)
    return kmax


def _build_pair_order(quantiles):
    pair_order = []
    for i, q1 in enumerate(quantiles):
        for q2 in quantiles[i:]:
            pair_order.append((f"DS{q1}", f"DS{q2}"))
    return tuple(pair_order)


def _filter_auto_pairs(pair_order, autos_only):
    """Optionally restrict the plotted DS pairs to auto spectra only."""
    if not autos_only:
        return pair_order
    return tuple(pair for pair in pair_order if pair[0] == pair[1])


def _build_default_shot_noise(pair_order, auto_shot, cross_shot):
    shot = {}
    for pair in pair_order:
        shot[pair] = auto_shot if pair[0] == pair[1] else cross_shot
    return shot


def _pair_block_edges(pair_order, block_sizes):
    """Return cumulative matrix edges separating DS-pair blocks."""
    if len(pair_order) * len(ELLS) != len(block_sizes):
        raise ValueError("block_sizes must contain one entry per pair-ell block.")

    edges = []
    offset = 0
    for idx in range(len(pair_order)):
        pair_size = sum(block_sizes[idx * len(ELLS):(idx + 1) * len(ELLS)])
        offset += pair_size
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


def _compute_fiducial_pair_poles(cfg, cosmo, k, pair_order, ds_model):
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

        poles[(label_a, label_b)] = compute_multipoles(k, model, ells=ELLS)
    return poles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q")
    parser.add_argument("--autos-only", action="store_true",
                        help="Plot only auto-spectrum DS blocks (e.g. DS1-DS1, DS5-DS5).")
    parser.add_argument("--ds-model", type=str, default="baseline",
                        choices=("baseline", "rsd_selection", "phenomenological"))
    parser.add_argument("--rebin", type=int, default=1, metavar="N",
                        help="Unused placeholder for parity with the P_gg plotting script.")
    parser.add_argument("--kmin", type=float, default=0.01, metavar="VALUE")
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE")
    parser.add_argument("--nk", type=int, default=80, metavar="N")
    parser.add_argument("--diag-cov", action="store_true",
                        help="Plot diagonal covariance instead of the analytic source.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Plot analytic cubic-box covariance.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic covariance terms: 'gaussian', 'gaussian+effective_cng', 'gaussian+ssc', or 'gaussian+effective_cng+ssc'.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR")
    parser.add_argument("--box-volume", type=float, default=1.0e9, metavar="V")
    parser.add_argument("--auto-shot-noise", type=float, default=250.0, metavar="P0")
    parser.add_argument("--cross-shot-noise", type=float, default=40.0, metavar="P0X")
    parser.add_argument("--cng-amplitude", type=float, default=None, metavar="A",
                        help="Defaults to 0.2 when effective_cng is enabled, otherwise 0.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA")
    parser.add_argument("--ssc-sigma-b2", type=float, default=None, metavar="VAR")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
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

    k = np.logspace(np.log10(max(args.kmin, 0.005)), np.log10(0.3), args.nk)
    poles = _compute_fiducial_pair_poles(cfg, cosmo, k, full_pair_order, args.ds_model)

    kmax_dict = _parse_kmax(args.kmax)
    per_pair_masks = []
    flat_blocks = []
    for pair in pair_order:
        for ell in ELLS:
            ell_mask = (k >= args.kmin) & (k <= kmax_dict[ell])
            per_pair_masks.append(ell_mask)
            flat_blocks.append(poles[pair][ell])

    flat = np.concatenate(flat_blocks)
    full_mask = np.concatenate(per_pair_masks)
    block_sizes = [int(mask.sum()) for mask in per_pair_masks]

    if args.diag_cov and args.analytic_cov:
        raise ValueError("Choose at most one of --diag-cov and --analytic-cov.")

    if args.diag_cov:
        cov, _ = diagonal_covariance(flat[full_mask], rescale=args.cov_rescale)
        plot_k = None
        plot_ells = None
    else:
        cov, _ = analytic_pqq_covariance(
            k,
            poles,
            ells=ELLS,
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
        plot_k = None
        plot_ells = None

    pair_str = ", ".join(f"{a}-{b}" for a, b in pair_order)
    fig, ax = plot_correlation_matrix(
        cov,
        block_sizes=block_sizes,
        cmap="RdBu_r",
        title=rf"$P_{{qq}}$ correlation ({args.ds_model}; {pair_str})",
    )

    for edge in _ell_block_edges(block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.35, alpha=0.25)
        ax.axhline(edge - 0.5, color="k", lw=0.35, alpha=0.25)

    for edge in _pair_block_edges(pair_order, block_sizes):
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.8)
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = OUTPUT_DIR / "correlation_matrix_pqq.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
