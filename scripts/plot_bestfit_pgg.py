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

from drift.utils.cosmology import (
    get_cosmology, _DEFAULT_PARAMS, ALL_COSMO_NAMES,
)
from drift.io import build_diagonal_covariance, load_observable_measurements
from drift.synthetic import make_synthetic_pgg
from inference_pgg import (
    SPACE, MODEL_MODE, MEAS_PATH, COV_DIR, ELLS, Z, VARY_COSMO,
    _resolve_pgg_covariance,
    parse_fix_cosmo,
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
        "--diag-cov",
        action="store_true",
        help="Use diagonal covariance for the plotted error bars.",
    )
    parser.add_argument(
        "--analytic-cov",
        action="store_true",
        help="Use analytic cubic-box covariance for the plotted error bars.",
    )
    parser.add_argument(
        "--analytic-cov-terms",
        type=str,
        default="gaussian",
        metavar="TERMS",
        help="Analytic covariance terms: 'gaussian', 'gaussian+effective_cng', 'gaussian+ssc', or 'gaussian+effective_cng+ssc'.",
    )
    parser.add_argument(
        "--box-volume",
        type=float,
        default=None,
        metavar="V",
        help="Box volume in (Mpc/h)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--number-density",
        type=float,
        default=None,
        metavar="N",
        help="Galaxy number density in (h/Mpc)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--shot-noise",
        type=float,
        default=None,
        metavar="P0",
        help="Constant shot-noise power in (Mpc/h)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--cng-amplitude",
        type=float,
        default=None,
        metavar="A",
        help=(
            "Amplitude of the effective connected covariance term. "
            "Defaults to the shared nonzero value when effective_cng is enabled."
        ),
    )
    parser.add_argument(
        "--cng-coherence",
        type=float,
        default=0.35,
        metavar="SIGMA",
        help="Log-k coherence length of the effective connected covariance term.",
    )
    parser.add_argument(
        "--ssc-sigma-b2",
        type=float,
        default=None,
        metavar="VAR",
        help="Long-mode density variance for the SSC term. If omitted and ssc is requested, it is estimated from --box-volume.",
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Chains include cosmological parameters as free parameters.",
    )
    parser.add_argument(
        "--fix-cosmo",
        nargs="+",
        default=None,
        metavar="PARAM[=VALUE]",
        help=(
            "Fixed cosmological parameters (must match inference run). "
            "Accepts 'name=value' or just 'name' (uses Planck 2018 default)."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Plot against synthetic (noiseless) data vector instead of measurements from disk.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["tree", "eft_ct", "eft", "one_loop"],
        default=None,
        metavar="MODE",
        help=(
            "Model mode used to generate synthetic data (default: same as MODEL_MODE). "
            "Must match the mode used during inference."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load chains — read param_names from the file to match the saved samples
    d = np.load(CHAINS_PATH, allow_pickle=True)
    samples = d["samples"]
    logl    = d["logl"]
    param_names = list(d["param_names"])
    best_idx = np.argmax(logl)
    theta_bf = samples[best_idx]

    # Detect marginalized chains and merge linear params
    is_marginalized = "linear_param_names" in d
    if is_marginalized:
        linear_param_names = list(d["linear_param_names"])
        linear_samples = d["linear_samples"]
        linear_bf = linear_samples[best_idx]
        print("Best-fit parameters (nonlinear, sampled):")
        for name, val in zip(param_names, theta_bf):
            print(f"  {name}: {val:.4f}")
        print("Best-fit parameters (linear, recovered):")
        for name, val in zip(linear_param_names, linear_bf):
            print(f"  {name}: {val:.4f}")
    else:
        print("Best-fit parameters:")
        for name, val in zip(param_names, theta_bf):
            print(f"  {name}: {val:.4f}")

    # Detect free cosmo params from chain
    cosmo_param_names = [n for n in param_names if n in ALL_COSMO_NAMES]

    # Build fixed cosmo: from --fix-cosmo CLI + defaults for non-free params
    fixed_cosmo = parse_fix_cosmo(args.fix_cosmo)
    # Also try to read from chains file if available
    if "fixed_cosmo" in d:
        for entry in d["fixed_cosmo"]:
            entry = str(entry)
            if "=" in entry:
                name, val = entry.split("=", 1)
                if name not in fixed_cosmo:
                    fixed_cosmo[name] = float(val)

    # Fill defaults for any param not free and not explicitly fixed
    for name in ALL_COSMO_NAMES:
        if name not in cosmo_param_names and name not in fixed_cosmo:
            fixed_cosmo[name] = _DEFAULT_PARAMS[name]

    # Build full cosmo dict for best-fit
    bf_dict = dict(zip(param_names, theta_bf))
    if is_marginalized:
        bf_dict.update(dict(zip(linear_param_names, linear_bf)))
    cosmo_for_bf = dict(fixed_cosmo)
    for name in cosmo_param_names:
        cosmo_for_bf[name] = bf_dict[name]

    # Load measurements (or generate synthetic data)
    cosmo = get_cosmology()
    if args.synthetic:
        synthetic_mode = args.synthetic_mode or MODEL_MODE
        if synthetic_mode == "tree":
            TRUE_PARAMS = {"b1": 2.0}
        elif synthetic_mode == "eft_ct":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0}
        elif synthetic_mode == "eft":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "s0": 100.0}
        elif synthetic_mode in ("one_loop",):
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "c4": 0.0, "s0": 100.0, "s2": 0.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
        else:
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "c4": 0.0, "s0": 100.0, "s2": 0.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
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
        k, poles = load_observable_measurements(
            MEAS_PATH, "pgg", ells=ELLS, rebin=args.rebin, kmin=args.kmin
        )

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
    data_y_masked = data_y[mask] if args.synthetic else np.concatenate([poles[ell] for ell in ELLS])[mask]
    fiducial_poles = {ell: poles[ell] for ell in ELLS} if not args.synthetic else data_y
    cov, _ = _resolve_pgg_covariance(
        args, k, data_y_masked, mask, fiducial_poles=fiducial_poles,
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

    # Evaluate theory at best-fit using param names from the chains file
    from drift.theory.galaxy.bias import GalaxyEFTParams
    from drift.theory.galaxy.power_spectrum import pgg_eft_mu
    from drift.utils.multipoles import compute_multipoles

    vary_cosmo = len(cosmo_param_names) > 0
    eval_cosmo = get_cosmology(cosmo_for_bf) if vary_cosmo else cosmo
    gal = GalaxyEFTParams(
        b1=bf_dict.get("b1", 1.0),
        c0=bf_dict.get("c0", 0.0),
        c2=bf_dict.get("c2", 0.0),
        c4=bf_dict.get("c4", 0.0),
        s0=bf_dict.get("s0", 0.0),
        s2=bf_dict.get("s2", 0.0),
        b2=bf_dict.get("b2", 0.0),
        bs2=bf_dict.get("bs2", 0.0),
        b3nl=bf_dict.get("b3nl", 0.0),
    )

    # Detect the model mode from the chain's parameter set
    all_param_names = list(param_names)
    if is_marginalized:
        all_param_names += linear_param_names
    chain_mode = MODEL_MODE
    if "b2" in all_param_names or "bs2" in all_param_names or "b3nl" in all_param_names:
        chain_mode = "one_loop"
    elif "s0" in all_param_names and "c2" not in all_param_names:
        chain_mode = "eft"
    elif "c0" in all_param_names and "s0" not in all_param_names:
        chain_mode = "eft_ct"

    def _bf_model(kk, mu):
        return pgg_eft_mu(
            kk, mu, z=Z, cosmo=eval_cosmo,
            gal_params=gal, space=SPACE, mode=chain_mode,
        )

    bf_poles = compute_multipoles(k, _bf_model, ells=ELLS)
    pred = np.concatenate([bf_poles[ell] for ell in ELLS])

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
    if is_marginalized:
        param_str += "  |  " + "  ".join(
            f"{name}={val:.3f}" for name, val in zip(linear_param_names, linear_bf)
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
