"""Best-fit theory vs. measurements for the P_gg multipoles.

Reads outputs/inference_pgg/{SPACE}/{MODEL_MODE}/chains.npz, identifies
the best-fit sample (highest log-likelihood), evaluates the theory model,
and overlays it on the measured multipoles.

Saves to outputs/inference_pgg/{SPACE}/{MODEL_MODE}/bestfit_multipoles.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.cosmology import get_cosmology
from drift.io import load_pgg_measurements, mock_covariance, diagonal_covariance
from drift.synthetic import make_synthetic_pgg
from inference_pgg import (
    _build_params, make_direct_theory_model,
    SPACE, MODEL_MODE, MEAS_PATH, COV_DIR, ELLS, Z, VARY_COSMO,
)

CHAINS_PATH = (
    Path(__file__).parents[1] / "outputs" / "inference_pgg" / SPACE / MODEL_MODE / "chains.npz"
)
OUTPUT_DIR = CHAINS_PATH.parent


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
        default=0.01,
        metavar="VALUE",
        help="Minimum k to include in the plot (default: 0.01).",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=13,
        metavar="N",
        help="Keep every Nth k-bin when loading measurements (default: 13).",
    )
    parser.add_argument(
        "--cov-rescale",
        type=float,
        default=64.0,
        metavar="FACTOR",
        help="Divide the covariance matrix by this factor (default: 64).",
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Chains include sigma8 and Omega_m as free parameters.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Plot against synthetic (noiseless) data vector instead of measurements from disk.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["tree_only", "eft_lite", "eft_full", "one_loop", "one_loop_matter_only"],
        default=None,
        metavar="MODE",
        help=(
            "Model mode used to generate synthetic data (default: same as MODEL_MODE). "
            "Must match the mode used during inference."
        ),
    )
    args = parser.parse_args()

    param_names, _ = _build_params(MODEL_MODE, vary_cosmo=args.vary_cosmo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load chains
    d = np.load(CHAINS_PATH, allow_pickle=True)
    samples = d["samples"]
    logl    = d["logl"]
    theta_bf = samples[np.argmax(logl)]
    print("Best-fit parameters:")
    for name, val in zip(param_names, theta_bf):
        print(f"  {name}: {val:.4f}")

    # Load measurements (or generate synthetic data)
    cosmo = get_cosmology()
    if args.synthetic:
        synthetic_mode = args.synthetic_mode or MODEL_MODE
        if synthetic_mode == "tree_only":
            TRUE_PARAMS = {"b1": 2.0}
        elif synthetic_mode == "eft_lite":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0}
        elif synthetic_mode == "eft_full":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "s0": 100.0}
        elif synthetic_mode in ("one_loop", "one_loop_matter_only"):
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "s0": 100.0, "b2": 0.5, "bs2": -0.5}
        else:
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "s0": 100.0, "b2": 0.5, "bs2": -0.5}
        k = np.linspace(0.01, 0.3, 30)
        print(f"Generating synthetic P_gg data vector (mode={synthetic_mode}) ...")
        data_y, _ = make_synthetic_pgg(
            k, ells=ELLS, z=Z, space=SPACE, mode=synthetic_mode,
            true_params=TRUE_PARAMS, cosmo=cosmo,
        )
        nk = len(k)
        poles = {}
        idx = 0
        for ell in ELLS:
            poles[ell] = data_y[idx:idx + nk]
            idx += nk
        print(f"  True parameters: {TRUE_PARAMS}")
    else:
        k, poles = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=args.rebin, kmin=args.kmin)

    # Build kmax mask
    kmax_dict = {ell: 0.5 for ell in ELLS}
    if args.kmax is not None:
        if len(args.kmax) == 1 and ":" not in args.kmax[0]:
            val = float(args.kmax[0])
            kmax_dict = {ell: val for ell in ELLS}
        else:
            for item in args.kmax:
                ell_str, val_str = item.split(":")
                kmax_dict[int(ell_str)] = float(val_str)

    # Covariance for error bars
    mask = np.concatenate([k <= kmax_dict[ell] for ell in ELLS])
    if args.synthetic:
        data_y_masked = data_y[mask]
        cov, _ = diagonal_covariance(data_y_masked, rescale=args.cov_rescale)
    else:
        print(f"Estimating covariance from mocks in {COV_DIR} ...")
        cov, _ = mock_covariance(
            COV_DIR, "pgg", ELLS, k_data=k, mask=mask,
            rescale=args.cov_rescale, rebin=args.rebin,
        )

    # Per-ell error bars from covariance diagonal
    errors = {}
    idx = 0
    for ell in ELLS:
        kmask_ell = k <= kmax_dict[ell]
        nk_ell = kmask_ell.sum()
        errors[ell] = np.sqrt(np.diag(cov)[idx:idx + nk_ell])
        idx += nk_ell
        print(errors[ell])

    # Evaluate theory at best-fit
    theory_fn = make_direct_theory_model(
        cosmo, k, ells=ELLS, mode=MODEL_MODE, vary_cosmo=args.vary_cosmo,
    )
    pred = theory_fn(theta_bf)

    # Reshape flat vector -> {ell: array}
    theory_bf = {}
    idx = 0
    for ell in ELLS:
        theory_bf[ell] = pred[idx:idx + len(k)]
        idx += len(k)

    # Build annotation string
    param_str = "  ".join(
        f"{name}={val:.3f}" for name, val in zip(param_names, theta_bf)
    )

    # Plot
    fig, axes = plt.subplots(1, len(ELLS), figsize=(6 * len(ELLS), 5))
    axes = np.atleast_1d(axes)

    for ax, ell in zip(axes, ELLS):
        kmax_val = kmax_dict.get(ell, np.inf)
        kmask_ell = k <= kmax_val
        k_plot = k[kmask_ell]
        meas_plot = poles[ell][kmask_ell]
        err_plot = errors[ell]

        data_label = "Synthetic" if args.synthetic else "Measurement"
        ax.errorbar(
            k_plot, k_plot * meas_plot,
            yerr=k_plot * err_plot,
            fmt="o", ms=3, color="C0", elinewidth=0.8, capsize=2, zorder=3,
            label=data_label,
        )
        ax.plot(k_plot, k_plot * theory_bf[ell][kmask_ell], color="C1", label="Best-fit")
        ax.set_title(rf"$\ell = {ell}$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(rf"$k\,P_{{{ell}}}$" + r" [$({\rm Mpc}/h)^2$]")
        ax.legend(fontsize=8)

    fig.suptitle(
        "DRIFT: best-fit P_gg multipoles\n" + param_str,
        fontsize=11,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "bestfit_multipoles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
