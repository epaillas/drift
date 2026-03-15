"""Best-fit theory vs. measurements for the DSG multipoles.

Reads outputs/inference_dsg/chains.npz, identifies the best-fit sample
(highest log-likelihood), evaluates the theory model, and overlays it on
the measured multipoles from outputs/dsg_measured.hdf5.

Saves to outputs/inference_dsg/bestfit_multipoles.png.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.cosmology import get_cosmology, get_linear_power
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.eft_models import pqg_eft_mu
from drift.one_loop import compute_one_loop_matter
from drift.multipoles import compute_multipoles
from drift.io import load_measurements

SPACE      = "redshift"         # "redshift" | "real"
DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft_lite"         # "tree_only" | "eft_lite" | "eft_full"

_suffix     = "_real" if SPACE == "real" else ""
CHAINS_PATH = (
    Path(__file__).parents[1] / "outputs" / "inference_dsg" / SPACE / DS_MODEL / MODEL_MODE / "chains.npz"
)
MEAS_PATH   = Path(__file__).parents[1] / "outputs" / f"dsg_measured{_suffix}.hdf5"
OUTPUT_DIR  = CHAINS_PATH.parent

Z           = 0.5
R           = 10.0
KERNEL      = "gaussian"
NOISE_FRAC  = 0.05
NOISE_FLOOR = 50.0
ELLS      = (0, 2)
QUANTILES = (1, 5)


def _build_params(ds_model, model_mode, quantiles):
    """Return (param_names, labels) for the given model configuration."""
    param_names = ["b1"] + [f"bq1_{q}" for q in quantiles]
    labels      = [r"$b_1$"] + [rf"$b_{{q1,{q}}}$" for q in quantiles]
    if model_mode in ("eft_lite", "eft_full"):
        param_names += ["c0"]
        labels      += [r"$c_0$"]
    if model_mode == "eft_full":
        param_names += ["s0"]
        labels      += [r"$s_0$"]
    if ds_model == "phenomenological":
        param_names += [f"beta_q_{q}" for q in quantiles]
        labels      += [rf"$\beta_{{q,{q}}}$" for q in quantiles]
    return param_names, labels


PARAM_NAMES, LABELS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES)


def make_eft_theory_model(cosmo, k, p1loop, ells, quantiles, ds_model, mode):
    """Return a callable theta -> flat_data_vector.

    p1loop is precomputed outside the MCMC loop (None for tree_only).
    """
    n_bq = len(quantiles)

    def theory(theta):
        b1 = theta[0]
        bq1_vals = theta[1:1 + n_bq]
        idx = 1 + n_bq
        c0, s0 = 0.0, 0.0
        if mode in ("eft_lite", "eft_full"):
            c0 = theta[idx]; idx += 1
        if mode == "eft_full":
            s0 = theta[idx]; idx += 1
        beta_q_vals = theta[idx:] if ds_model == "phenomenological" else [0.0] * n_bq

        gal = GalaxyEFTParams(b1=float(b1), c0=float(c0), s0=float(s0))
        lk = {"p1loop_precomputed": p1loop} if p1loop is not None else {}
        vec = []
        for q, bq1, betaq in zip(quantiles, bq1_vals, beta_q_vals):
            ds_bin = DSSplitBinEFT(label=f"DS{q}", bq1=float(bq1), beta_q=float(betaq))
            poles = compute_multipoles(
                k, pqg_eft_mu,
                z=Z, cosmo=cosmo, ds_params=ds_bin, gal_params=gal,
                R=R, kernel=KERNEL, ells=ells, space=SPACE,
                ds_model=ds_model, mode=mode, loop_kwargs=lk,
            )
            for ell in ells:
                vec.append(poles[ell])
        return np.concatenate(vec)

    return theory


def main():
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
    k, measured = load_measurements(MEAS_PATH, nquantiles=max(QUANTILES), ells=ELLS)

    # Compute per-bin uncertainties from the same diagonal covariance as inference
    errors = {}
    for q in QUANTILES:
        label = f"DS{q}"
        errors[label] = {}
        for ell in ELLS:
            p = measured[label][ell]
            errors[label][ell] = np.sqrt((NOISE_FRAC * np.abs(p)) ** 2 + NOISE_FLOOR ** 2)

    # Evaluate theory at best-fit
    cosmo = get_cosmology()
    p1loop = None
    if MODEL_MODE in ("eft_lite", "eft_full"):
        print("Precomputing one-loop matter power spectrum ...")
        plin_func = lambda kk: get_linear_power(cosmo, kk, Z)
        loop_result = compute_one_loop_matter(k, plin_func)
        p1loop = loop_result["p1loop"]
        print("  done.")
    theory_fn = make_eft_theory_model(
        cosmo, k, p1loop, ells=ELLS,
        quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
    )
    pred      = theory_fn(theta_bf)

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
            add_label = ell == ELLS[-1]
            ax.errorbar(
                k, k * measured[label][ell],
                yerr=k * errors[label][ell],
                fmt="o", ms=3, color=color, elinewidth=0.8, capsize=2, zorder=3,
                label=label if add_label else None,
            )
            ax.plot(k, k * theory_bf[label][ell], color=color)

    for ax, ell in zip(axes, ELLS):
        ax.set_title(rf"$k\,P_{ell}(k)$")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel(rf"$k\,P_{{{ell}}}$" + r" [$({\rm Mpc}/h)^2$]")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, ls="--")
    axes[-1].legend(fontsize=8)

    fig.suptitle(
        "DRIFT: best-fit DSG multipoles\n" + param_str,
        fontsize=11,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "bestfit_multipoles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
