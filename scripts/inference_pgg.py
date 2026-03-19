"""Galaxy auto-power P_gg multipole fitting with PocoMC.

Recovers galaxy bias b1 (and optionally EFT/stochastic nuisances and cosmology)
from galaxy auto-power spectrum multipole measurements.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

from drift.cosmology import get_cosmology, LinearPowerGrid, OneLoopPowerGrid
from drift.galaxy_emulator import GalaxyTemplateEmulator
from drift.io import load_pgg_measurements, mock_covariance, diagonal_covariance
from drift.synthetic import make_synthetic_pgg

# Import shared utilities from inference_dsg
from inference_dsg import make_log_likelihood

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACE      = "redshift"  # "redshift" | "real"
MODEL_MODE = "one_loop"  # "tree_only" | "eft_lite" | "eft_full" | "one_loop"

MEAS_PATH  = Path(__file__).parents[1] / "outputs" / "hods" / "mesh2_spectrum_poles_c000_hod006.h5"
COV_DIR    = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
OUTPUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "inference_pgg" / SPACE / MODEL_MODE
)

Z    = 0.5
ELLS = (0, 2)

VARY_COSMO = False


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------
def _build_params(model_mode, vary_cosmo=False):
    """Return (param_names, bounds) for the given model configuration."""
    if vary_cosmo:
        param_names = ["sigma8", "Omega_m", "b1"]
        bounds = np.array([
            [0.6, 1.2],
            [0.2, 0.5],
            [0.5, 4.0],
        ])
    else:
        param_names = ["b1"]
        bounds = np.array([[0.5, 4.0]])

    if model_mode in ("eft_lite", "eft_full", "one_loop", "one_loop_matter_only"):
        param_names += ["c0"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
    if model_mode in ("eft_full", "one_loop", "one_loop_matter_only"):
        param_names += ["s0"]
        bounds = np.vstack([bounds, [[-5000.0, 5000.0]]])
    if model_mode in ("one_loop", "one_loop_matter_only"):
        param_names += ["c2"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
    if model_mode == "one_loop":
        param_names += ["b2", "bs2", "b3nl"]
        bounds = np.vstack([bounds, [[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]]])

    return param_names, bounds


PARAM_NAMES, BOUNDS = _build_params(MODEL_MODE, vary_cosmo=VARY_COSMO)


# ---------------------------------------------------------------------------
# Theory model helpers
# ---------------------------------------------------------------------------
def _unpack_theta(theta, mode, vary_cosmo=False):
    """Unpack flat parameter vector into a dict."""
    idx = 0
    sigma8, omega_m = None, None
    if vary_cosmo:
        sigma8  = float(theta[idx]); idx += 1
        omega_m = float(theta[idx]); idx += 1
    b1 = float(theta[idx]); idx += 1
    c0, c2, s0, b2, bs2, b3nl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if mode in ("eft_lite", "eft_full", "one_loop", "one_loop_matter_only"):
        c0 = float(theta[idx]); idx += 1
    if mode in ("eft_full", "one_loop", "one_loop_matter_only"):
        s0 = float(theta[idx]); idx += 1
    if mode in ("one_loop", "one_loop_matter_only"):
        c2  = float(theta[idx]); idx += 1
    if mode == "one_loop":
        b2   = float(theta[idx]); idx += 1
        bs2  = float(theta[idx]); idx += 1
        b3nl = float(theta[idx]); idx += 1
    return dict(b1=b1, c0=c0, c2=c2, s0=s0, b2=b2, bs2=bs2, b3nl=b3nl,
                sigma8=sigma8, omega_m=omega_m)


def make_eft_theory_model(cosmo, k, ells, mode, cosmo_grid=None):
    """Return a callable theta -> flat_data_vector using GalaxyTemplateEmulator."""
    vary_cosmo = cosmo_grid is not None
    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=Z, space=SPACE, mode=mode,
    )

    def theory(theta):
        p = _unpack_theta(theta, mode, vary_cosmo=vary_cosmo)
        if vary_cosmo:
            if mode in ("one_loop", "one_loop_matter_only"):
                plin, f, loop_arrays = cosmo_grid.predict(p["sigma8"], p["omega_m"])
                emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
            else:
                plin, f = cosmo_grid.predict(p["sigma8"], p["omega_m"])
                emulator.update_cosmology(plin, f)
        params = {"b1": p["b1"], "c0": p["c0"], "c2": p["c2"], "s0": p["s0"],
                  "b2": p.get("b2", 0.0), "bs2": p.get("bs2", 0.0),
                  "b3nl": p.get("b3nl", 0.0)}
        return emulator.predict(params)

    return theory


def make_direct_theory_model(cosmo, k, ells, mode, vary_cosmo=False):
    """Return a callable theta -> flat_data_vector using direct GL quadrature."""
    from drift.eft_bias import GalaxyEFTParams
    from drift.galaxy_models import pgg_eft_mu
    from drift.multipoles import compute_multipoles

    def theory(theta):
        p = _unpack_theta(theta, mode, vary_cosmo=vary_cosmo)
        eval_cosmo = (
            get_cosmology({"sigma8": p["sigma8"], "Omega_m": p["omega_m"]})
            if vary_cosmo else cosmo
        )
        gal = GalaxyEFTParams(
            b1=p["b1"], c0=p["c0"], c2=p["c2"], s0=p["s0"],
            b2=p.get("b2", 0.0), bs2=p.get("bs2", 0.0),
            b3nl=p.get("b3nl", 0.0),
        )

        def model(kk, mu):
            return pgg_eft_mu(
                kk, mu, z=Z, cosmo=eval_cosmo,
                gal_params=gal, space=SPACE, mode=mode,
            )

        poles = compute_multipoles(k, model, ells=ells)
        return np.concatenate([poles[ell] for ell in ells])

    return theory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-emulator",
        action="store_true",
        help="Use direct GL quadrature instead of the analytic template emulator.",
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Jointly infer sigma8 and Omega_m alongside bias parameters.",
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
        default=13,
        metavar="N",
        help="Keep every Nth k-bin when loading measurements (default: 13).",
    )
    parser.add_argument(
        "--kmin",
        type=float,
        default=0.01,
        metavar="VALUE",
        help="Minimum k to include in the fit (default: 0.01).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic (noiseless) data vector instead of measurements from disk.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["tree_only", "eft_lite", "eft_full", "one_loop", "one_loop_matter_only"],
        default=None,
        metavar="MODE",
        help=(
            "Model mode used to generate synthetic data (default: same as MODEL_MODE). "
            "Allows cross-mode tests, e.g. generate with tree_only, fit with one_loop."
        ),
    )
    parser.add_argument(
        "--diag-cov",
        action="store_true",
        help="Use diagonal covariance instead of mock covariance.",
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
    global PARAM_NAMES, BOUNDS
    if vary_cosmo != VARY_COSMO:
        PARAM_NAMES, BOUNDS = _build_params(MODEL_MODE, vary_cosmo=vary_cosmo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements (or generate synthetic data)
    if args.synthetic:
        synthetic_mode = args.synthetic_mode or MODEL_MODE
        # True parameters depend on the synthetic mode:
        # simpler modes only need b1; extra params are zero by definition.
        if synthetic_mode == "tree_only":
            TRUE_PARAMS = {"b1": 2.0}
        elif synthetic_mode == "eft_lite":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0}
        elif synthetic_mode == "eft_full":
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "s0": 100.0}
        elif synthetic_mode in ("one_loop", "one_loop_matter_only"):
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "s0": 100.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
        else:
            TRUE_PARAMS = {"b1": 2.0, "c0": 5.0, "c2": 2.0, "s0": 100.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
        k = np.linspace(0.01, 0.3, 30)
        cosmo = get_cosmology()
        print(f"Generating synthetic P_gg data vector (mode={synthetic_mode}) ...")
        print(f"  True parameters: {TRUE_PARAMS}")
        data_y, _ = make_synthetic_pgg(
            k, ells=ELLS, z=Z, space=SPACE, mode=synthetic_mode,
            true_params=TRUE_PARAMS, cosmo=cosmo,
        )
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")
        print(f"  Data vector length: {len(data_y)}")
    else:
        print(f"Loading measurements from {MEAS_PATH} ...")
        k, poles = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=args.rebin, kmin=args.kmin)
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")

        data_y = np.concatenate([poles[ell] for ell in ELLS])
        print(f"  Data vector length: {len(data_y)}")

    # Apply per-ell kmax masking
    kmax_dict = {ell: 0.5 for ell in ELLS}
    if args.kmax is not None:
        if len(args.kmax) == 1 and ":" not in args.kmax[0]:
            val = float(args.kmax[0])
            kmax_dict = {ell: val for ell in ELLS}
        else:
            for item in args.kmax:
                ell_str, val_str = item.split(":")
                kmax_dict[int(ell_str)] = float(val_str)

    mask = np.concatenate([k <= kmax_dict[ell] for ell in ELLS])
    data_y_masked = data_y[mask]
    print(f"  Masked data vector length: {len(data_y_masked)}")

    # 2. Cosmology (may already be set in synthetic branch)
    if not args.synthetic:
        cosmo = get_cosmology()

    # 3. Theory model
    print(f"  EFT mode: {MODEL_MODE},  space: {SPACE}")
    cosmo_grid = None
    if vary_cosmo and not args.no_emulator:
        if MODEL_MODE in ("one_loop", "one_loop_matter_only"):
            print("Precomputing OneLoopPowerGrid (sigma8 × Omega_m) ...")
            cosmo_grid = OneLoopPowerGrid(k, z=Z)
        else:
            print("Precomputing LinearPowerGrid (sigma8 × Omega_m) ...")
            cosmo_grid = LinearPowerGrid(k, z=Z)
        print("  done.")

    if args.no_emulator:
        print("Using direct GL quadrature (--no-emulator) ...")
        theory_fn = make_direct_theory_model(
            cosmo, k, ells=ELLS, mode=MODEL_MODE, vary_cosmo=vary_cosmo,
        )
    else:
        print("Building template emulator ...")
        theory_fn = make_eft_theory_model(
            cosmo, k, ells=ELLS, mode=MODEL_MODE, cosmo_grid=cosmo_grid,
        )
    print("  done.")

    theory_fn_masked = lambda theta: theory_fn(theta)[mask]

    if args.diag_cov or args.synthetic:
        cov, precision_matrix = diagonal_covariance(data_y_masked, rescale=args.cov_rescale)
    else:
        print(f"Estimating covariance from mocks in {COV_DIR} ...")
        cov, precision_matrix = mock_covariance(
            COV_DIR, "pgg", ELLS, k_data=k, mask=mask,
            rescale=args.cov_rescale, rebin=args.rebin,
        )
        print(f"  Covariance matrix shape: {cov.shape}")

    # 4. Log-likelihood
    log_likelihood = make_log_likelihood(data_y_masked, precision_matrix, theory_fn_masked)

    # 5. Prior
    dists = [sp_uniform(loc=lo, scale=hi - lo) for lo, hi in BOUNDS]
    prior = Prior(dists)

    # 6. Sampler
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

    # 7. Save results
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

    # 8. Print summary
    if args.synthetic:
        print(f"\nTrue parameters: {TRUE_PARAMS}")
    print("\nParameter estimates (posterior mean ± std):")
    header = f"  {'name':>8}  {'mean':>8}  {'std':>8}"
    if args.synthetic:
        header += f"  {'true':>8}"
    print(header)
    for i, name in enumerate(PARAM_NAMES):
        mean = np.average(samples[:, i], weights=weights)
        var  = np.average((samples[:, i] - mean) ** 2, weights=weights)
        std  = np.sqrt(var)
        if args.synthetic:
            true_val = TRUE_PARAMS.get(name, 0.0)
            print(f"  {name:>8}  {mean:>8.3f}  {std:>8.3f}  {true_val:>8.3f}")
        else:
            print(f"  {name:>8}  {mean:>8.3f}  {std:>8.3f}")


if __name__ == "__main__":
    main()
