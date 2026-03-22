"""HOD parameter inference with PocoMC.

Recovers per-bin density biases bq[1..5] and galaxy tracer bias b1
from density-split × galaxy multipole measurements in
outputs/hods/dsc_pkqg_poles_c000_hod006.h5.
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
from drift.io import load_measurements, mock_covariance, taylor_cache_key
from inference_dsg import (
    _build_params, _build_data_mask, _parse_kmax,
    make_eft_theory_model, make_direct_theory_model,
    make_log_likelihood,
    SPACE, KERNEL, Z, R, ELLS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft_full"         # "tree_only" | "eft_lite" | "eft_full"

MEAS_PATH  = Path(__file__).parents[1] / "outputs" / "hods" / "dsc_pkqg_poles_c000_hod006.h5"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs" / "inference_hod" / DS_MODEL / MODEL_MODE
COV_DIR    = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
QUANTILES  = (1, 5)   # all 5 quantiles

VARY_COSMO = False   # set True to jointly infer sigma8, Omega_m

PARAM_NAMES, BOUNDS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES, vary_cosmo=VARY_COSMO)


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
        "--kmin",
        type=float,
        default=0.025,
        metavar="VALUE",
        help=(
            "Minimum k to include in the fit (applied to all multipoles). "
            "Default: 0.025, chosen to skip the k=0 bins and the ell=2 FFT-grid "
            "zero at k~0.022 h/Mpc where only line-of-sight modes exist."
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
        "--taylor",
        action="store_true",
        help="Wrap the theory model in a Taylor expansion emulator for fast evaluation.",
    )
    parser.add_argument(
        "--taylor-order",
        type=int,
        default=4,
        metavar="N",
        help="Maximum Taylor expansion order (default: 4). Only used with --taylor.",
    )
    parser.add_argument(
        "--taylor-step",
        type=float,
        default=0.01,
        metavar="FRAC",
        help="Relative step size for finite differences (default: 0.01). Only used with --taylor.",
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
        global PARAM_NAMES, BOUNDS
        PARAM_NAMES, BOUNDS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES,
                                            vary_cosmo=vary_cosmo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements
    print(f"Loading measurements from {MEAS_PATH} ...")
    k, multipoles_per_bin = load_measurements(MEAS_PATH, nquantiles=5, ells=ELLS, rebin=args.rebin)
    print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")

    # Build flat data vector
    data_y = np.concatenate([
        multipoles_per_bin[f"DS{q}"][ell]
        for q in QUANTILES
        for ell in ELLS
    ])
    print(f"  Data vector length: {len(data_y)}")

    # Apply per-ell kmax and global kmin masking
    kmax_dict = _parse_kmax(args.kmax, ELLS)
    mask = _build_data_mask(k, ELLS, QUANTILES, kmax_dict)
    kmin_mask = np.tile(k >= args.kmin, len(QUANTILES) * len(ELLS))
    mask = mask & kmin_mask
    print(f"  kmin: {args.kmin:.4g} h/Mpc  ({(k >= args.kmin).sum()}/{len(k)} k-bins kept)")
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

    if args.taylor:
        from drift.taylor import TaylorEmulator
        fiducial = {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(PARAM_NAMES, BOUNDS)}
        cache_hash = taylor_cache_key(
            ds_model=DS_MODEL, model_mode=MODEL_MODE, space=SPACE, z=Z,
            ells=ELLS, quantiles=QUANTILES, kmax=str(kmax_dict),
            kmin=args.kmin, rebin=args.rebin,
            vary_cosmo=vary_cosmo, taylor_order=args.taylor_order,
            taylor_step=args.taylor_step, fiducial=str(sorted(fiducial.items())),
            param_names=str(PARAM_NAMES),
        )
        cache_path = OUTPUT_DIR / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            # Build the base theory model (only needed on cache miss)
            print("Using direct GL quadrature ...")
            base_theory_fn = make_direct_theory_model(
                cosmo, k, ells=ELLS,
                quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
                vary_cosmo=vary_cosmo,
            )
            base_theory_masked = lambda theta: base_theory_fn(theta)[mask]

            def _dict_theory(params):
                theta = np.array([params[name] for name in PARAM_NAMES])
                return base_theory_masked(theta)

            print(f"Building Taylor emulator (order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(
                _dict_theory, fiducial, order=args.taylor_order,
                step_sizes=args.taylor_step,
            )
            taylor_emu.save_coefficients(cache_path)
            print(f"  Cached Taylor coefficients to {cache_path}")

        theory_fn_masked = lambda theta: taylor_emu.predict(
            {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
        )
        print("  done.")
    else:
        cosmo_grid = None
        if vary_cosmo and not args.no_emulator:
            print("Precomputing LinearPowerGrid (sigma8 × Omega_m) ...")
            cosmo_grid = LinearPowerGrid(
                k, z=Z,
                cosmo_ranges={"sigma8": (0.6, 1.2, 20), "Omega_m": (0.2, 0.5, 20)},
            )
            print("  done.")
        if args.no_emulator:
            print("Using direct GL quadrature ...")
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

    print(f"Estimating covariance from mocks in {COV_DIR} ...")
    cov, precision_matrix = mock_covariance(
        COV_DIR, "ds", ELLS, mask=mask, rescale=args.cov_rescale,
        rebin=args.rebin, nquantiles=5, quantiles=QUANTILES,
    )
    print(f"  Covariance matrix shape: {cov.shape}")

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
    print(f"  {'name':>12}  {'mean':>8}  {'std':>8}")
    for i, name in enumerate(PARAM_NAMES):
        mean = np.average(samples[:, i], weights=weights)
        var  = np.average((samples[:, i] - mean) ** 2, weights=weights)
        std  = np.sqrt(var)
        print(f"  {name:>12}  {mean:>8.3f}  {std:>8.3f}")


if __name__ == "__main__":
    main()
