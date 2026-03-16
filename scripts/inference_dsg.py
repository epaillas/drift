"""DSG parameter inference with PocoMC.

Recovers per-bin density biases bq[1..5] and galaxy tracer bias b1
from density-split × galaxy multipole measurements in outputs/dsg_measured.hdf5.
Cosmology is fixed to Planck 2018.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

from drift.cosmology import get_cosmology, LinearPowerGrid
from drift.emulator import TemplateEmulator
from drift.io import load_measurements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACE      = "redshift"         # "redshift" | "real"
DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft_full"         # "tree_only" | "eft_lite" | "eft_full"

_suffix   = "_real" if SPACE == "real" else ""
MEAS_PATH = Path(__file__).parents[1] / "outputs" / f"dsg_measured{_suffix}.hdf5"
OUTPUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "inference_dsg" / SPACE / DS_MODEL / MODEL_MODE
)

Z = 0.5
R = 10.0
KERNEL    = "gaussian"
ELLS      = (0, 2)
QUANTILES = (1, 2, 4, 5)


def _parse_kmax(kmax_args, ells):
    """Parse --kmax CLI values into {ell: float} dict.

    Accepts either a single float (applied to all ells) or
    'ell:value' pairs, e.g. ['0:0.3', '2:0.2'].
    Returns {ell: kmax} for each ell in ells; missing ells get np.inf.
    """
    kmax_dict = {ell: np.inf for ell in ells}
    if kmax_args is None:
        return kmax_dict
    if len(kmax_args) == 1 and ":" not in kmax_args[0]:
        val = float(kmax_args[0])
        return {ell: val for ell in ells}
    for item in kmax_args:
        ell_str, val_str = item.split(":")
        kmax_dict[int(ell_str)] = float(val_str)
    return kmax_dict


def _build_data_mask(k, ells, quantiles, kmax_dict):
    """Build a flat boolean mask matching the data vector ordering.

    Data vector order: for q in quantiles, for ell in ells, k-values.
    kmax_dict: {ell: kmax_value}; missing ells -> no cut (np.inf).
    Returns np.ndarray of bool, shape (n_quantiles * n_ells * nk,).
    """
    segments = []
    for _q in quantiles:
        for ell in ells:
            kmax = kmax_dict.get(ell, np.inf)
            segments.append(k <= kmax)
    return np.concatenate(segments)


def _build_params(ds_model, model_mode, quantiles, vary_cosmo=False):
    """Return (param_names, bounds) for the given model configuration."""
    if vary_cosmo:
        param_names = ["sigma8", "Omega_m", "b1"] + [f"bq1_{q}" for q in quantiles]
        bounds = np.array([
            [0.6, 1.2],
            [0.2, 0.5],
            [0.5, 4.0],
            *[[-4.0, 4.0] for _ in quantiles],
        ])
    else:
        param_names = ["b1"] + [f"bq1_{q}" for q in quantiles]
        bounds = np.array([
            [0.5, 4.0],
            *[[-4.0, 4.0] for _ in quantiles],
        ])
    if model_mode in ("eft_lite", "eft_full"):
        param_names += ["c0"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
    if model_mode == "eft_full":
        param_names += ["s0"]
        bounds = np.vstack([bounds, [[-5000.0, 5000.0]]])
    if ds_model == "phenomenological":
        param_names += [f"beta_q_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-2.0, 2.0] for _ in quantiles]])
    if model_mode in ("eft_lite", "eft_full"):
        param_names += [f"bq_nabla2_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-4.0, 4.0] for _ in quantiles]])
    return param_names, bounds


VARY_COSMO = False   # set True to jointly infer sigma8, Omega_m

PARAM_NAMES, BOUNDS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES, vary_cosmo=VARY_COSMO)


# ---------------------------------------------------------------------------
# Theory model
# ---------------------------------------------------------------------------
def _unpack_theta(theta, n_bq, mode, ds_model, vary_cosmo=False):
    """Unpack flat parameter vector into a dict."""
    idx = 0
    sigma8, omega_m = None, None
    if vary_cosmo:
        sigma8  = float(theta[idx]); idx += 1
        omega_m = float(theta[idx]); idx += 1
    b1 = theta[idx]; idx += 1
    bq1_vals = theta[idx:idx + n_bq]
    idx += n_bq
    c0, s0 = 0.0, 0.0
    if mode in ("eft_lite", "eft_full"):
        c0 = theta[idx]; idx += 1
    if mode == "eft_full":
        s0 = theta[idx]; idx += 1
    if ds_model == "phenomenological":
        beta_q_vals = theta[idx:idx + n_bq]; idx += n_bq
    else:
        beta_q_vals = [0.0] * n_bq
    bq_nabla2_vals = theta[idx:idx + n_bq] if mode in ("eft_lite", "eft_full") else [0.0] * n_bq
    return dict(b1=float(b1), bq1_vals=bq1_vals, c0=float(c0), s0=float(s0),
                beta_q_vals=beta_q_vals, bq_nabla2_vals=bq_nabla2_vals,
                sigma8=sigma8, omega_m=omega_m)


def make_eft_theory_model(cosmo, k, ells, quantiles, ds_model, mode,
                          cosmo_grid=None):
    """Return a callable theta -> flat_data_vector using TemplateEmulator.

    Parameters
    ----------
    cosmo_grid : LinearPowerGrid, optional
        If provided, sigma8 and Omega_m are treated as free parameters and
        the emulator is updated via ``update_cosmology`` each step.
    """
    vary_cosmo = cosmo_grid is not None
    n_bq = len(quantiles)
    emulator = TemplateEmulator(
        cosmo, k, ells=ells, z=Z, R=R,
        kernel=KERNEL, space=SPACE,
        ds_model=ds_model, mode=mode,
    )

    def theory(theta):
        p = _unpack_theta(theta, n_bq, mode, ds_model, vary_cosmo=vary_cosmo)
        if vary_cosmo:
            plin, f = cosmo_grid.predict(p["sigma8"], p["omega_m"])
            emulator.update_cosmology(plin, f)
        params = {
            "b1": p["b1"],
            "bq1": [float(v) for v in p["bq1_vals"]],
            "c0": p["c0"],
            "s0": p["s0"],
            "beta_q": [float(v) for v in p["beta_q_vals"]],
            "bq_nabla2": [float(v) for v in p["bq_nabla2_vals"]],
        }
        return emulator.predict(params)

    return theory


def make_direct_theory_model(cosmo, k, ells, quantiles, ds_model, mode,
                             vary_cosmo=False):
    """Return a callable theta -> flat_data_vector using direct GL quadrature.

    Slower reference implementation — useful as a sanity-check against the
    template emulator, and as the evaluation path for best-fit plotting.
    When vary_cosmo=True, sigma8 and Omega_m are taken from theta and a
    fresh cosmology object is built per call.
    """
    from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
    from drift.eft_models import pqg_eft_mu
    from drift.multipoles import compute_multipoles

    n_bq = len(quantiles)

    def theory(theta):
        p = _unpack_theta(theta, n_bq, mode, ds_model, vary_cosmo=vary_cosmo)
        eval_cosmo = (
            get_cosmology({"sigma8": p["sigma8"], "Omega_m": p["omega_m"]})
            if vary_cosmo else cosmo
        )
        gal = GalaxyEFTParams(b1=p["b1"], c0=p["c0"], s0=p["s0"])
        vec = []
        for q, bq1, betaq, bq_nabla2 in zip(
            quantiles, p["bq1_vals"], p["beta_q_vals"], p["bq_nabla2_vals"]
        ):
            ds_bin = DSSplitBinEFT(
                label=f"DS{q}", bq1=float(bq1),
                beta_q=float(betaq), bq_nabla2=float(bq_nabla2),
            )

            def model(kk, mu, _ds=ds_bin, _gal=gal, _cosmo=eval_cosmo):
                return pqg_eft_mu(
                    kk, mu, z=Z, cosmo=_cosmo,
                    ds_params=_ds, gal_params=_gal,
                    R=R, kernel=KERNEL, space=SPACE,
                    ds_model=ds_model, mode=mode,
                )

            poles = compute_multipoles(k, model, ells=ells)
            for ell in ells:
                vec.append(poles[ell])
        return np.concatenate(vec)

    return theory


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def make_diagonal_cov(data_y, noise_frac=0.05, floor=50.0):
    """Diagonal covariance: fractional relative noise with absolute floor."""
    var = (noise_frac * np.abs(data_y)) ** 2 + floor ** 2
    return np.diag(var)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------
def make_log_likelihood(data_y, precision_matrix, theory_fn):
    def log_likelihood(theta):
        pred = theory_fn(theta)
        diff = data_y - pred
        return -0.5 * diff @ precision_matrix @ diff
    return log_likelihood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-emulator",
        action="store_true",
        help=(
            "Use direct Gauss-Legendre quadrature instead of the analytic "
            "template emulator. Slower but useful as a sanity check."
        ),
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Jointly infer sigma8 and Omega_m alongside EFT/bias parameters.",
    )
    parser.add_argument(
        "--kmax",
        nargs="+",
        default=None,
        metavar="[ELL:]VALUE",
        help=(
            "Maximum k to include in the fit. Either a single float (applied "
            "to all multipoles) or 'ell:value' pairs, e.g. '0:0.3 2:0.2'."
        ),
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
        # Rebuild param names / bounds for the chosen mode
        global PARAM_NAMES, BOUNDS
        PARAM_NAMES, BOUNDS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES,
                                            vary_cosmo=vary_cosmo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements
    print(f"Loading measurements from {MEAS_PATH} ...")
    k, multipoles_per_bin = load_measurements(MEAS_PATH, nquantiles=max(QUANTILES), ells=ELLS, rebin=args.rebin)
    print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")

    # Build flat data vector
    data_y = np.concatenate([
        multipoles_per_bin[f"DS{q}"][ell]
        for q in QUANTILES
        for ell in ELLS
    ])
    print(f"  Data vector length: {len(data_y)}")

    # Apply per-ell kmax masking
    kmax_dict = _parse_kmax(args.kmax, ELLS)
    mask = _build_data_mask(k, ELLS, QUANTILES, kmax_dict)
    print("  kmax cuts:")
    for ell in ELLS:
        kmax_val = kmax_dict.get(ell, np.inf)
        n_kept = (k <= kmax_val).sum()
        print(f"    ell={ell}: kmax={kmax_val:.4g} h/Mpc  ({n_kept}/{len(k)} k-bins kept)")
    data_y_masked = data_y[mask]
    print(f"  Masked data vector length: {len(data_y_masked)}")

    # 2. Cosmology
    cosmo = get_cosmology()

    # 3. Theory model
    print(f"  DS model: {DS_MODEL},  EFT mode: {MODEL_MODE}")
    cosmo_grid = None
    if vary_cosmo and not args.no_emulator:
        print("Precomputing LinearPowerGrid (sigma8 × Omega_m) ...")
        cosmo_grid = LinearPowerGrid(k, z=Z)
        print("  done.")
    if args.no_emulator:
        print("Using direct GL quadrature (--no-emulator) ...")
        theory_fn = make_direct_theory_model(
            cosmo, k, ells=ELLS,
            quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
        )
    else:
        print("Building template emulator ...")
        theory_fn = make_eft_theory_model(
            cosmo, k, ells=ELLS,
            quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
            cosmo_grid=cosmo_grid,
        )
    print("  done.")
    theory_fn_masked = lambda theta: theory_fn(theta)[mask]
    cov = make_diagonal_cov(data_y_masked) / args.cov_rescale
    precision_matrix = np.linalg.inv(cov)

    # 5. Log-likelihood
    log_likelihood = make_log_likelihood(data_y_masked, precision_matrix, theory_fn_masked)

    # 6. Prior (built from BOUNDS to stay in sync with _build_params)
    dists = [sp_uniform(loc=lo, scale=hi - lo) for lo, hi in BOUNDS]
    prior = Prior(dists)

    # 7. Sampler
    print("Initialising PocoMC sampler ...")
    sampler = Sampler(
        prior=prior,
        likelihood=log_likelihood,
        n_dim=len(PARAM_NAMES),
        n_effective=512,
        n_active=256,
        output_dir=str(OUTPUT_DIR),
        random_state=42,
    )
    sampler.run(n_total=2000, n_evidence=0, progress=True)

    # 8. Save results
    samples, weights, logl, logp = sampler.posterior()
    chains_path = OUTPUT_DIR / "chains.npz"
    np.savez(
        chains_path,
        samples=samples,
        weights=weights,
        logl=logl,
        param_names=np.array(PARAM_NAMES),
    )
    print(f"\nChains saved to {chains_path}")

    # 9. Print summary
    print("\nParameter estimates (posterior mean ± std):")
    print(f"  {'name':>6}  {'mean':>8}  {'std':>8}  {'truth':>8}")
    truths = {"b1": 2.0}  # known mock truth for galaxy bias; bq values not known a priori
    for i, name in enumerate(PARAM_NAMES):
        mean = np.average(samples[:, i], weights=weights)
        var  = np.average((samples[:, i] - mean) ** 2, weights=weights)
        std  = np.sqrt(var)
        truth_str = f"{truths[name]:.3f}" if name in truths else "  ---  "
        print(f"  {name:>6}  {mean:>8.3f}  {std:>8.3f}  {truth_str:>8}")


if __name__ == "__main__":
    main()
