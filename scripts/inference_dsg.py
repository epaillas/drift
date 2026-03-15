"""DSG parameter inference with PocoMC.

Recovers per-bin density biases bq[1..5] and galaxy tracer bias b1
from density-split × galaxy multipole measurements in outputs/dsg_measured.hdf5.
Cosmology is fixed to Planck 2018.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

from drift.cosmology import get_cosmology, get_linear_power
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.eft_models import pqg_eft_mu
from drift.one_loop import compute_one_loop_matter
from drift.multipoles import compute_multipoles
from drift.io import load_measurements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACE      = "redshift"         # "redshift" | "real"
DS_MODEL   = "phenomenological" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "eft_lite"         # "tree_only" | "eft_lite" | "eft_full"

_suffix   = "_real" if SPACE == "real" else ""
MEAS_PATH = Path(__file__).parents[1] / "outputs" / f"dsg_measured{_suffix}.hdf5"
OUTPUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "inference_dsg" / SPACE / DS_MODEL / MODEL_MODE
)

Z = 0.5
R = 10.0
KERNEL    = "gaussian"
ELLS      = (0, 2)
QUANTILES = (1, 5)


def _build_params(ds_model, model_mode, quantiles):
    """Return (param_names, bounds) for the given model configuration."""
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
    return param_names, bounds


PARAM_NAMES, BOUNDS = _build_params(DS_MODEL, MODEL_MODE, QUANTILES)


# ---------------------------------------------------------------------------
# Theory model
# ---------------------------------------------------------------------------
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements
    print(f"Loading measurements from {MEAS_PATH} ...")
    k, multipoles_per_bin = load_measurements(MEAS_PATH, nquantiles=max(QUANTILES), ells=ELLS)
    print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")

    # Build flat data vector
    data_y = np.concatenate([
        multipoles_per_bin[f"DS{q}"][ell]
        for q in QUANTILES
        for ell in ELLS
    ])
    print(f"  Data vector length: {len(data_y)}")

    # 2. Cosmology
    cosmo = get_cosmology()

    # 3. Precompute one-loop matter power spectrum (once, before sampler)
    p1loop = None
    if MODEL_MODE in ("eft_lite", "eft_full"):
        print("Precomputing one-loop matter power spectrum ...")
        plin_func = lambda kk: get_linear_power(cosmo, kk, Z)
        loop_result = compute_one_loop_matter(k, plin_func)
        p1loop = loop_result["p1loop"]
        print("  done.")

    # 4. Theory model and covariance
    print(f"  DS model: {DS_MODEL},  EFT mode: {MODEL_MODE}")
    theory_fn = make_eft_theory_model(
        cosmo, k, p1loop, ells=ELLS,
        quantiles=QUANTILES, ds_model=DS_MODEL, mode=MODEL_MODE,
    )
    cov = make_diagonal_cov(data_y)
    precision_matrix = np.linalg.inv(cov)

    # 5. Log-likelihood
    log_likelihood = make_log_likelihood(data_y, precision_matrix, theory_fn)

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
