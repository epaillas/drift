"""Best-fit theory vs. measurements for the HOD DSG multipoles.

Reads outputs/inference_hod/chains.npz, identifies the best-fit sample
(highest log-likelihood), evaluates the theory model, and overlays it on
the measured multipoles from outputs/hods/dsc_pkqg_poles_c000_hod006.h5.

Saves to outputs/inference_hod/<DS_MODEL>/<MODEL_MODE>/bestfit_multipoles.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.utils.cosmology import get_cosmology
from drift.io import estimate_mock_covariance, load_observable_measurements
from inference_dsg import _parse_kmax, _build_data_mask, make_direct_theory_model, ELLS, Z, R, KERNEL, SPACE
from plot_bestfit_dsg import _build_params

DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft"         # "tree" | "eft_ct" | "eft"

CHAINS_PATH = Path(__file__).parents[1] / "outputs" / "inference_hod" / DS_MODEL / MODEL_MODE / "chains.npz"
MEAS_PATH   = Path(__file__).parents[1] / "outputs" / "hods" / "dsc_pkqg_poles_c000_hod006.h5"
OUTPUT_DIR  = CHAINS_PATH.parent
COV_DIR     = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"

QUANTILES   = (1, 5)

VARY_COSMO = False   # set True to jointly infer sigma8, Omega_m

PARAM_NAMES, LABELS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES, vary_cosmo=VARY_COSMO)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kmax",
        nargs="+",
        default=None,
        metavar="[ELL:]VALUE",
        help=(
            "Maximum k to include in the plot. Either a single float (applied "
            "to all multipoles) or 'ell:value' pairs, e.g. '0:0.3 2:0.2'."
        ),
    )
    parser.add_argument(
        "--kmin",
        type=float,
        default=0.025,
        metavar="VALUE",
        help=(
            "Minimum k to include in the plot (applied to all multipoles). "
            "Default: 0.025, chosen to skip the k=0 bins and the ell=2 FFT-grid "
            "zero at k~0.022 h/Mpc where only line-of-sight modes exist."
        ),
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Chains include sigma8 and Omega_m as free parameters.",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=5,
        metavar="N",
        help="Keep every Nth k-bin when loading measurements (default: 5).",
    )
    parser.add_argument(
        "--cov-rescale",
        type=float,
        default=64.0,
        metavar="FACTOR",
        help="Divide the covariance matrix by this factor (default: 64).",
    )
    args = parser.parse_args()
    vary_cosmo = args.vary_cosmo

    if vary_cosmo != VARY_COSMO:
        global PARAM_NAMES, LABELS
        PARAM_NAMES, LABELS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES,
                                            vary_cosmo=vary_cosmo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load chains
    d = np.load(CHAINS_PATH, allow_pickle=True)
    samples = d["samples"]
    logl    = d["logl"]

    theta_bf = samples[np.argmax(logl)]
    print("Best-fit parameters:")
    for name, label, val in zip(PARAM_NAMES, LABELS, theta_bf):
        print(f"  {name}: {val:.4f}")

    # Load measurements
    k, measured = load_observable_measurements(MEAS_PATH, "pqg", nquantiles=5, ells=ELLS, rebin=args.rebin)

    # Build kmax/kmin mask — identical logic to inference_hod.py
    kmax_dict = _parse_kmax(args.kmax, ELLS)
    mask = _build_data_mask(k, ELLS, QUANTILES, kmax_dict)
    kmin_mask = np.tile(k >= args.kmin, len(QUANTILES) * len(ELLS))
    mask = mask & kmin_mask

    # Compute per-bin uncertainties from mock covariance diagonal
    cov_masked = estimate_mock_covariance(
        COV_DIR, "pqg", ELLS, mask=mask, rescale=args.cov_rescale,
        rebin=args.rebin, nquantiles=5, quantiles=QUANTILES,
    )
    err_masked = np.sqrt(np.diag(cov_masked))

    # Scatter masked errors back into full-length arrays so the plot loop can
    # index them with its per-ell kmask; unmasked positions stay zero.
    err_full = np.zeros(len(QUANTILES) * len(ELLS) * len(k))
    err_full[mask] = err_masked

    errors = {}
    idx = 0
    for q in QUANTILES:
        label = f"DS{q}"
        errors[label] = {}
        for ell in ELLS:
            errors[label][ell] = err_full[idx * len(k):(idx + 1) * len(k)]
            idx += 1

    # Evaluate theory at best-fit
    cosmo = get_cosmology()
    theory_fn = make_direct_theory_model(
        cosmo, k, ells=ELLS,
        quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
        vary_cosmo=vary_cosmo,
    )
    pred = theory_fn(theta_bf)

    # Reshape flat vector -> {DS_label: {ell: array}}
    theory_bf = {}
    idx = 0
    for q in QUANTILES:
        label = f"DS{q}"
        theory_bf[label] = {}
        for ell in ELLS:
            theory_bf[label][ell] = pred[idx * len(k):(idx + 1) * len(k)]
            idx += 1

    # Build annotation string
    param_str = "  ".join(
        f"{name}={val:.3f}" for name, val in zip(PARAM_NAMES, theta_bf)
    )

    # Plot
    colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(QUANTILES)))
    fig, axes = plt.subplots(1, len(ELLS), figsize=(6 * len(ELLS), 5))
    axes = np.atleast_1d(axes)

    for q, color in zip(QUANTILES, colors):
        label = f"DS{q}"
        for ax, ell in zip(axes, ELLS):
            kmax_val = kmax_dict.get(ell, np.inf)
            kmask = (k >= args.kmin) & (k <= kmax_val)
            add_label = ell == ELLS[-1]
            ax.errorbar(
                k[kmask], k[kmask] * measured[label][ell][kmask],
                yerr=k[kmask] * errors[label][ell][kmask],
                fmt="o", ms=3, color=color, elinewidth=0.8, capsize=2, zorder=3,
                label=label if add_label else None,
            )
            ax.plot(k[kmask], k[kmask] * theory_bf[label][ell][kmask], color=color)

    for ax, ell in zip(axes, ELLS):
        ax.set_title(rf"$k\,P_{ell}(k)$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(rf"$k\,P_{{{ell}}}$" + r" [$({\rm Mpc}/h)^2$]")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, ls="--")
    axes[-1].legend(fontsize=8)

    fig.suptitle(
        "DRIFT: best-fit HOD DSG multipoles\n" + param_str,
        fontsize=11,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "bestfit_multipoles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
