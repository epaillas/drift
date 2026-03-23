"""Compare standard vs analytic-marginalized inference for eft_lite.

Runs both approaches on the same synthetic data and compares the
posterior of b1 to verify consistency. Uses eft_lite mode (only c0 is
linear) for speed: 2D (b1, c0) standard vs 1D (b1) marginalized.
Fixed cosmology isolates the marginalization effect.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.utils.cosmology import get_cosmology
from drift.emulators.galaxy import GalaxyTemplateEmulator
from drift.analytic_marginalization import MarginalizedLikelihood
from drift.synthetic import make_synthetic_pgg
from drift.io import diagonal_covariance

# --- Config ---
MODE = "eft_ct"
ELLS = (0, 2)
Z = 0.5
SPACE = "redshift"
TRUE_PARAMS = {"b1": 2.0, "c0": 5.0}
COV_RESCALE = 64.0
PRIOR_SIGMA_C0 = 100.0  # Gaussian prior sigma for c0 in marginalized run

# Sampler settings (small for speed)
N_TOTAL = 1000
N_EFFECTIVE = 256
N_ACTIVE = 128


def main():
    # 1. Generate synthetic data
    k = np.linspace(0.01, 0.3, 30)
    cosmo = get_cosmology()
    print("Generating synthetic data ...")
    print(f"  True params: {TRUE_PARAMS}")
    data_y, _ = make_synthetic_pgg(k, ELLS, Z, SPACE, MODE, TRUE_PARAMS, cosmo)
    cov, precision = diagonal_covariance(data_y, rescale=COV_RESCALE)
    print(f"  Data vector length: {len(data_y)}")

    # 2. Build emulator
    emulator = GalaxyTemplateEmulator(cosmo, k, ells=ELLS, z=Z, space=SPACE, mode=MODE)

    # ---- RUN A: Standard (sample b1 and c0 explicitly) ----
    print("\n=== RUN A: Standard sampling (b1, c0) ===")

    def theory_standard(theta):
        b1, c0 = theta
        return emulator.predict({"b1": b1, "c0": c0})

    def logl_standard(theta):
        pred = theory_standard(theta)
        diff = data_y - pred
        return -0.5 * diff @ precision @ diff

    prior_std = Prior([
        sp_uniform(loc=0.5, scale=3.5),      # b1 in [0.5, 4.0]
        sp_uniform(loc=-50.0, scale=100.0),   # c0 in [-50, 50]
    ])

    sampler_std = Sampler(
        prior=prior_std, likelihood=logl_standard, n_dim=2,
        n_effective=N_EFFECTIVE, n_active=N_ACTIVE, random_state=42,
    )
    sampler_std.run(n_total=N_TOTAL, progress=True)
    samples_std, weights_std, logl_std, _ = sampler_std.posterior()

    # ---- RUN B: Marginalized (sample b1 only, c0 analytically integrated) ----
    print("\n=== RUN B: Marginalized sampling (b1 only, c0 analytic) ===")
    print(f"  Prior sigma for c0: {PRIOR_SIGMA_C0}")

    prior_sigmas = np.array([PRIOR_SIGMA_C0])
    marg_like = MarginalizedLikelihood(data_y, precision, prior_sigmas)

    def logl_marg(theta):
        b1 = theta[0]
        m, T = emulator.predict_decomposed({"b1": b1})
        return marg_like(m, T)

    prior_marg = Prior([sp_uniform(loc=0.5, scale=3.5)])  # b1 only

    sampler_marg = Sampler(
        prior=prior_marg, likelihood=logl_marg, n_dim=1,
        n_effective=N_EFFECTIVE, n_active=N_ACTIVE, random_state=42,
    )
    sampler_marg.run(n_total=N_TOTAL, progress=True)
    samples_marg, weights_marg, logl_marg_vals, _ = sampler_marg.posterior()

    # ---- Compare ----
    b1_mean_std = np.average(samples_std[:, 0], weights=weights_std)
    b1_std_std = np.sqrt(np.average((samples_std[:, 0] - b1_mean_std) ** 2, weights=weights_std))

    c0_mean_std = np.average(samples_std[:, 1], weights=weights_std)
    c0_std_std = np.sqrt(np.average((samples_std[:, 1] - c0_mean_std) ** 2, weights=weights_std))

    b1_mean_marg = np.average(samples_marg[:, 0], weights=weights_marg)
    b1_std_marg = np.sqrt(np.average((samples_marg[:, 0] - b1_mean_marg) ** 2, weights=weights_marg))

    # Recover c0 at best-fit b1 from marginalized run
    best_idx = np.argmax(logl_marg_vals)
    b1_best_marg = samples_marg[best_idx, 0]
    m_best, T_best = emulator.predict_decomposed({"b1": b1_best_marg})
    c0_recovered = marg_like.bestfit_linear_params(m_best, T_best)[0]

    # Recover c0 for all marginalized posterior samples
    n_marg = samples_marg.shape[0]
    c0_marg_samples = np.zeros(n_marg)
    for i in range(n_marg):
        m_i, T_i = emulator.predict_decomposed({"b1": samples_marg[i, 0]})
        c0_marg_samples[i] = marg_like.bestfit_linear_params(m_i, T_i)[0]
    c0_mean_marg = np.average(c0_marg_samples, weights=weights_marg)
    c0_std_marg = np.sqrt(np.average((c0_marg_samples - c0_mean_marg) ** 2, weights=weights_marg))

    print(f"\n{'=' * 55}")
    print(f"{'COMPARISON':^55}")
    print(f"{'=' * 55}")
    print(f"{'':>20} {'mean':>10} {'std':>10} {'true':>10}")
    print(f"{'-' * 55}")
    print(f"{'Standard b1':>20} {b1_mean_std:>10.4f} {b1_std_std:>10.4f} {TRUE_PARAMS['b1']:>10.4f}")
    print(f"{'Marginalized b1':>20} {b1_mean_marg:>10.4f} {b1_std_marg:>10.4f} {TRUE_PARAMS['b1']:>10.4f}")
    print(f"{'-' * 55}")
    print(f"{'Standard c0':>20} {c0_mean_std:>10.4f} {c0_std_std:>10.4f} {TRUE_PARAMS['c0']:>10.4f}")
    print(f"{'Marginalized c0':>20} {c0_mean_marg:>10.4f} {c0_std_marg:>10.4f} {TRUE_PARAMS['c0']:>10.4f}")
    print(f"{'c0 at best b1':>20} {c0_recovered:>10.4f} {'':>10} {TRUE_PARAMS['c0']:>10.4f}")
    print(f"{'=' * 55}")

    # Diagnostic: check if b1 posteriors are consistent
    b1_diff = abs(b1_mean_std - b1_mean_marg)
    b1_combined_err = np.sqrt(b1_std_std ** 2 + b1_std_marg ** 2)
    tension = b1_diff / b1_combined_err if b1_combined_err > 0 else 0.0
    std_ratio = b1_std_marg / b1_std_std if b1_std_std > 0 else float('inf')

    print(f"\nDiagnostics:")
    print(f"  b1 mean difference:  {b1_diff:.4f}")
    print(f"  Tension (sigma):     {tension:.2f}")
    print(f"  Width ratio (marg/std): {std_ratio:.3f}")
    if std_ratio > 1.5:
        print("  WARNING: Marginalized b1 is significantly wider than standard.")
        print("  Possible causes: prior mismatch, log|A| Occam term, template bug.")
    elif std_ratio < 0.67:
        print("  WARNING: Marginalized b1 is significantly narrower than standard.")
    else:
        print("  OK: b1 widths are consistent.")

    if tension > 2.0:
        print(f"  WARNING: b1 means disagree at {tension:.1f} sigma.")
    else:
        print(f"  OK: b1 means are consistent.")

    # Save results for further analysis
    output_path = Path(__file__).parent / "marg_consistency_results.npz"
    np.savez(
        output_path,
        samples_std=samples_std,
        weights_std=weights_std,
        logl_std=logl_std,
        samples_marg=samples_marg,
        weights_marg=weights_marg,
        logl_marg=logl_marg_vals,
        c0_marg_samples=c0_marg_samples,
        true_params=np.array([TRUE_PARAMS["b1"], TRUE_PARAMS["c0"]]),
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
