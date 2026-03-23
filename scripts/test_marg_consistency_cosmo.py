"""Cosmo-varying consistency test: standard vs analytic-marginalized inference.

Extends the fixed-cosmo test by adding sigma8 as a free cosmological parameter.
Standard run: samples over (sigma8, b1, <linear_params>) with explicit chi-squared.
Marginalized run: samples over (sigma8, b1) with linear params analytically integrated out.

Before running the sampler, a grid diagnostic compares the profiled and marginalized
log-likelihoods on a coarse (sigma8, b1) grid to detect shape differences.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import uniform as sp_uniform, norm as sp_norm
import pocomc
from pocomc import Sampler, Prior

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.utils.cosmology import get_cosmology, LinearPowerGrid, OneLoopPowerGrid, _DEFAULT_PARAMS
from drift.emulators.galaxy import GalaxyTemplateEmulator
from drift.analytic_marginalization import MarginalizedLikelihood
from drift.synthetic import make_synthetic_pgg
from drift.io import diagonal_covariance

# --- Config ---
MODE = "one_loop"
ELLS = (0, 2)
Z = 0.5
SPACE = "redshift"
TRUE_PARAMS = {"b1": 2.0, "c0": 5.0}
TRUE_SIGMA8 = _DEFAULT_PARAMS["sigma8"]  # 0.8102
COV_RESCALE = 64.0

# sigma8 grid range for LinearPowerGrid
SIGMA8_RANGE = (0.7, 0.9, 11)

# Diagnostic grid resolution
DIAG_N_SIGMA8 = 9
DIAG_N_B1 = 9

# Sampler settings
N_TOTAL = 2000
N_EFFECTIVE = 256
N_ACTIVE = 128

# Prior sigmas for linear parameters (broad Gaussian priors).
# Will be set per-parameter after building the emulator.
DEFAULT_PRIOR_SIGMA = 100.0



def profile_linear(emulator, data_y, precision, b1, prior_sigmas):
    """Profile (optimize) over all linear parameters analytically.

    For model d = m + T @ alpha with Gaussian prior alpha ~ N(0, S),
    the MAP estimate is:
        alpha_hat = (T^T C^{-1} T + S^{-1})^{-1} T^T C^{-1} r
    where r = data - m.

    Uses the same prior regularization as the marginalized likelihood
    so that the profiled and marginalized log-L are directly comparable.

    Returns the profiled log-likelihood and the best-fit alpha vector.
    """
    m, T = emulator.predict_decomposed({"b1": b1})
    r = data_y - m
    Cinv_r = precision @ r
    Cinv_T = precision @ T
    Sinv = np.diag(1.0 / prior_sigmas ** 2)
    A = T.T @ Cinv_T + Sinv     # (n_linear, n_linear)
    v = T.T @ Cinv_r            # (n_linear,)
    alpha_hat = cho_solve(cho_factor(A), v)
    # Profiled log-likelihood (chi2 at alpha_hat, including prior penalty)
    pred = m + T @ alpha_hat
    diff = data_y - pred
    logl = -0.5 * (diff @ precision @ diff + alpha_hat @ Sinv @ alpha_hat)
    return logl, alpha_hat


def main():
    # 1. Generate synthetic data at fiducial cosmology
    k = np.linspace(0.01, 0.3, 30)
    cosmo = get_cosmology()
    print("Generating synthetic data ...")
    print(f"  True params: sigma8={TRUE_SIGMA8}, {TRUE_PARAMS}")
    data_y, _ = make_synthetic_pgg(k, ELLS, Z, SPACE, MODE, TRUE_PARAMS, cosmo)
    cov, precision = diagonal_covariance(data_y, rescale=COV_RESCALE)
    print(f"  Data vector length: {len(data_y)}")

    # 2. Build cosmology grid (1D over sigma8)
    cosmo_ranges = {"sigma8": SIGMA8_RANGE}
    needs_loops = MODE in ("one_loop",)
    if needs_loops:
        print("\nBuilding OneLoopPowerGrid over sigma8 ...")
        cosmo_grid = OneLoopPowerGrid(k, Z, cosmo_ranges=cosmo_ranges)
    else:
        print("\nBuilding LinearPowerGrid over sigma8 ...")
        cosmo_grid = LinearPowerGrid(k, Z, cosmo_ranges=cosmo_ranges)

    # 3. Build emulator (will be updated each likelihood call)
    emulator = GalaxyTemplateEmulator(cosmo, k, ells=ELLS, z=Z, space=SPACE, mode=MODE)

    def update_cosmo(s8):
        """Update emulator cosmology from the grid at given sigma8."""
        if needs_loops:
            plin, f, loop_arrays = cosmo_grid.predict(sigma8=s8)
            emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
        else:
            plin, f = cosmo_grid.predict(sigma8=s8)
            emulator.update_cosmology(plin, f)
    lin_names = emulator.linear_param_names
    n_linear = len(lin_names)
    print(f"  Mode: {MODE}")
    print(f"  Linear parameters ({n_linear}): {lin_names}")

    # True values for linear params (default to 0 if not in TRUE_PARAMS)
    true_linear = np.array([TRUE_PARAMS.get(name, 0.0) for name in lin_names])

    # 4. Marginalized likelihood object
    prior_sigmas = np.array([DEFAULT_PRIOR_SIGMA] * n_linear)
    marg_like = MarginalizedLikelihood(data_y, precision, prior_sigmas)

    # ================================================================
    # DIAGNOSTIC: Grid comparison of profiled vs marginalized log-L
    # ================================================================
    print("\n=== DIAGNOSTIC: Grid comparison ===")
    sigma8_grid = np.linspace(SIGMA8_RANGE[0] + 0.02, SIGMA8_RANGE[1] - 0.02, DIAG_N_SIGMA8)
    b1_grid = np.linspace(1.0, 3.0, DIAG_N_B1)

    logl_profiled = np.zeros((DIAG_N_SIGMA8, DIAG_N_B1))
    logl_marginalized = np.zeros((DIAG_N_SIGMA8, DIAG_N_B1))

    for i, s8 in enumerate(sigma8_grid):
        update_cosmo(s8)
        for j, b1 in enumerate(b1_grid):
            # Profiled log-L (optimize over all linear params)
            logl_prof, _ = profile_linear(emulator, data_y, precision, b1, prior_sigmas)
            logl_profiled[i, j] = logl_prof

            # Marginalized log-L
            m, T = emulator.predict_decomposed({"b1": b1})
            logl_marg = marg_like(m, T)
            logl_marginalized[i, j] = logl_marg

    # Compare: the difference should be roughly constant (up to the Occam factor)
    diff_grid = logl_marginalized - logl_profiled
    diff_range = diff_grid.max() - diff_grid.min()

    print(f"\n  log-L difference (marginalized - profiled):")
    print(f"    min:   {diff_grid.min():.4f}")
    print(f"    max:   {diff_grid.max():.4f}")
    print(f"    range: {diff_range:.4f}")

    # Check variation along sigma8 axis (at best b1)
    best_b1_idx = np.argmin(np.abs(b1_grid - TRUE_PARAMS["b1"]))
    diff_vs_sigma8 = diff_grid[:, best_b1_idx]
    sigma8_variation = diff_vs_sigma8.max() - diff_vs_sigma8.min()
    print(f"\n  Offset variation along sigma8 (at b1={b1_grid[best_b1_idx]:.2f}):")
    print(f"    range: {sigma8_variation:.4f}")
    for i, s8 in enumerate(sigma8_grid):
        print(f"    sigma8={s8:.4f}: profiled={logl_profiled[i, best_b1_idx]:.2f}  "
              f"marg={logl_marginalized[i, best_b1_idx]:.2f}  "
              f"diff={diff_grid[i, best_b1_idx]:.4f}")

    if sigma8_variation > 1.0:
        print("\n  WARNING: Occam factor (log|A| - log|S^{-1}|) varies significantly")
        print("  with sigma8 — this will bias the marginalized sigma8 posterior.")
    else:
        print("\n  OK: Occam factor is roughly constant across sigma8.")

    # ================================================================
    # RUN A: Standard sampling (sigma8, b1, <linear_params>)
    # ================================================================
    # IMPORTANT: Use the same Gaussian prior on linear params as the
    # marginalized run. A prior mismatch (e.g. flat vs Gaussian) causes
    # spurious width differences even when the likelihood is correct.
    n_dim_std = 2 + n_linear  # sigma8, b1, then linear params
    Sinv = np.diag(1.0 / prior_sigmas ** 2)
    std_param_names = ["sigma8", "b1"] + lin_names
    print(f"\n=== RUN A: Standard sampling ({', '.join(std_param_names)}) ===")
    print(f"  n_dim = {n_dim_std}")
    print(f"  Gaussian prior on linear params: sigma={DEFAULT_PRIOR_SIGMA}")

    def logl_standard(theta):
        s8, b1 = theta[0], theta[1]
        lin_vals = theta[2:]
        update_cosmo(s8)
        params = {"b1": b1}
        for name, val in zip(lin_names, lin_vals):
            params[name] = val
        pred = emulator.predict(params)
        diff = data_y - pred
        alpha = np.array(lin_vals)
        return -0.5 * float(diff @ precision @ diff + alpha @ Sinv @ alpha)

    prior_std_list = [
        sp_uniform(loc=SIGMA8_RANGE[0], scale=SIGMA8_RANGE[1] - SIGMA8_RANGE[0]),
        sp_uniform(loc=0.5, scale=3.5),       # b1 in [0.5, 4.0]
    ]
    for _ in lin_names:
        prior_std_list.append(sp_norm(loc=0, scale=DEFAULT_PRIOR_SIGMA))
    prior_std = Prior(prior_std_list)

    sampler_std = Sampler(
        prior=prior_std, likelihood=logl_standard, n_dim=n_dim_std,
        n_effective=N_EFFECTIVE, n_active=N_ACTIVE, random_state=42,
    )
    sampler_std.run(n_total=N_TOTAL, progress=True)
    samples_std, weights_std, logl_std, _ = sampler_std.posterior()

    # ================================================================
    # RUN B: Marginalized 2D sampling (sigma8, b1)
    # ================================================================
    print("\n=== RUN B: Marginalized sampling (sigma8, b1) ===")
    print(f"  Prior sigmas: {dict(zip(lin_names, prior_sigmas))}")

    def logl_marg_fn(theta):
        s8, b1 = theta
        update_cosmo(s8)
        m, T = emulator.predict_decomposed({"b1": b1})
        return marg_like(m, T)

    prior_marg = Prior([
        sp_uniform(loc=SIGMA8_RANGE[0], scale=SIGMA8_RANGE[1] - SIGMA8_RANGE[0]),
        sp_uniform(loc=0.5, scale=3.5),       # b1 in [0.5, 4.0]
    ])

    sampler_marg = Sampler(
        prior=prior_marg, likelihood=logl_marg_fn, n_dim=2,
        n_effective=N_EFFECTIVE, n_active=N_ACTIVE, random_state=42,
    )
    sampler_marg.run(n_total=N_TOTAL, progress=True)
    samples_marg, weights_marg, logl_marg_vals, _ = sampler_marg.posterior()

    # ================================================================
    # Compare posteriors
    # ================================================================
    def weighted_stats(x, w):
        mean = np.average(x, weights=w)
        std = np.sqrt(np.average((x - mean) ** 2, weights=w))
        return mean, std

    s8_mean_std, s8_std_std = weighted_stats(samples_std[:, 0], weights_std)
    b1_mean_std, b1_std_std = weighted_stats(samples_std[:, 1], weights_std)

    s8_mean_marg, s8_std_marg = weighted_stats(samples_marg[:, 0], weights_marg)
    b1_mean_marg, b1_std_marg = weighted_stats(samples_marg[:, 1], weights_marg)

    # Standard run: linear param stats (columns 2..2+n_linear)
    lin_stats_std = {}
    for j, name in enumerate(lin_names):
        lin_stats_std[name] = weighted_stats(samples_std[:, 2 + j], weights_std)

    # Recover linear params for all marginalized posterior samples
    n_marg = samples_marg.shape[0]
    lin_marg_samples = np.zeros((n_marg, n_linear))
    for i in range(n_marg):
        update_cosmo(samples_marg[i, 0])
        m_i, T_i = emulator.predict_decomposed({"b1": samples_marg[i, 1]})
        lin_marg_samples[i] = marg_like.bestfit_linear_params(m_i, T_i)
    lin_stats_marg = {}
    for j, name in enumerate(lin_names):
        lin_stats_marg[name] = weighted_stats(lin_marg_samples[:, j], weights_marg)

    # Recover linear params at best-fit point
    best_idx = np.argmax(logl_marg_vals)
    update_cosmo(samples_marg[best_idx, 0])
    m_best, T_best = emulator.predict_decomposed({"b1": samples_marg[best_idx, 1]})
    lin_bestfit = marg_like.bestfit_linear_params(m_best, T_best)

    print(f"\n{'=' * 65}")
    print(f"{'COMPARISON':^65}")
    print(f"{'=' * 65}")
    print(f"{'':>25} {'mean':>10} {'std':>10} {'true':>10}")
    print(f"{'-' * 65}")
    print(f"{'Standard sigma8':>25} {s8_mean_std:>10.4f} {s8_std_std:>10.4f} {TRUE_SIGMA8:>10.4f}")
    print(f"{'Marginalized sigma8':>25} {s8_mean_marg:>10.4f} {s8_std_marg:>10.4f} {TRUE_SIGMA8:>10.4f}")
    print(f"{'-' * 65}")
    print(f"{'Standard b1':>25} {b1_mean_std:>10.4f} {b1_std_std:>10.4f} {TRUE_PARAMS['b1']:>10.4f}")
    print(f"{'Marginalized b1':>25} {b1_mean_marg:>10.4f} {b1_std_marg:>10.4f} {TRUE_PARAMS['b1']:>10.4f}")
    print(f"{'-' * 65}")
    for j, name in enumerate(lin_names):
        true_val = TRUE_PARAMS.get(name, 0.0)
        m_s, s_s = lin_stats_std[name]
        m_m, s_m = lin_stats_marg[name]
        print(f"{'Standard ' + name:>25} {m_s:>10.4f} {s_s:>10.4f} {true_val:>10.4f}")
        print(f"{'Marginalized ' + name:>25} {m_m:>10.4f} {s_m:>10.4f} {true_val:>10.4f}")
        print(f"{'bestfit ' + name:>25} {lin_bestfit[j]:>10.4f} {'':>10} {true_val:>10.4f}")
    print(f"{'=' * 65}")

    # Diagnostics
    print("\nDiagnostics:")

    for name, mean_std, std_std, mean_marg, std_marg, true_val in [
        ("sigma8", s8_mean_std, s8_std_std, s8_mean_marg, s8_std_marg, TRUE_SIGMA8),
        ("b1", b1_mean_std, b1_std_std, b1_mean_marg, b1_std_marg, TRUE_PARAMS["b1"]),
    ]:
        diff = abs(mean_std - mean_marg)
        combined_err = np.sqrt(std_std ** 2 + std_marg ** 2)
        tension = diff / combined_err if combined_err > 0 else 0.0
        ratio = std_marg / std_std if std_std > 0 else float('inf')

        print(f"\n  {name}:")
        print(f"    Mean difference:       {diff:.4f}")
        print(f"    Tension (sigma):       {tension:.2f}")
        print(f"    Width ratio (marg/std): {ratio:.3f}")
        if ratio > 1.5:
            print(f"    WARNING: Marginalized {name} is significantly wider than standard.")
        elif ratio < 0.67:
            print(f"    WARNING: Marginalized {name} is significantly narrower than standard.")
        else:
            print(f"    OK: {name} widths are consistent.")
        if tension > 2.0:
            print(f"    WARNING: {name} means disagree at {tension:.1f} sigma.")
        else:
            print(f"    OK: {name} means are consistent.")

    # Save results
    output_path = Path(__file__).parent / "marg_consistency_cosmo_results.npz"
    np.savez(
        output_path,
        samples_std=samples_std,
        weights_std=weights_std,
        logl_std=logl_std,
        samples_marg=samples_marg,
        weights_marg=weights_marg,
        logl_marg=logl_marg_vals,
        lin_marg_samples=lin_marg_samples,
        true_params=np.array([TRUE_SIGMA8, TRUE_PARAMS["b1"]] +
                             [TRUE_PARAMS.get(n, 0.0) for n in lin_names]),
        sigma8_grid=sigma8_grid,
        b1_grid=b1_grid,
        logl_profiled=logl_profiled,
        logl_marginalized=logl_marginalized,
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
