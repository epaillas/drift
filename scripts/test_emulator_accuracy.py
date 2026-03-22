"""Test accuracy of grid-interpolated P_gg vs exact CLASS computations.

Quantifies how well LinearPowerGrid / OneLoopPowerGrid cubic-spline
interpolation reproduces exact cosmoprimo P_lin(k) and loop integrals
across random cosmologies. Produces diagnostic plots.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from drift.cosmology import (
    get_cosmology, LinearPowerGrid, OneLoopPowerGrid,
    _DEFAULT_PARAMS, DEFAULT_COSMO_RANGES, ALL_COSMO_NAMES,
)
from drift.galaxy_emulator import GalaxyTemplateEmulator
from inference_pgg import parse_fix_cosmo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z = 0.5
ELLS = (0, 2, 4)
SPACE = "redshift"

BIAS_PARAMS = dict(b1=2.0, c0=5.0, c2=2.0, c4=0.0, s0=100.0, s2=0.0, b2=0.5, bs2=-0.5, b3nl=0.1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Test grid emulator accuracy for P_gg")
    p.add_argument("--mode", default="one_loop",
                   choices=["tree_only", "eft_lite", "eft_full", "one_loop", "one_loop_matter_only"])
    p.add_argument("--n-test", type=int, default=50, help="Number of random test cosmologies")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kmin", type=float, default=0.01)
    p.add_argument("--kmax", type=float, default=0.3)
    p.add_argument("--nk", type=int, default=60)
    p.add_argument("--fix-cosmo", nargs="+", default=None, metavar="PARAM[=VALUE]",
                   help="Fix cosmological parameters (same syntax as inference_pgg.py).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    k = np.linspace(args.kmin, args.kmax, args.nk)
    outdir = Path(__file__).resolve().parents[1] / "outputs" / "emulator_accuracy" / args.mode
    outdir.mkdir(parents=True, exist_ok=True)

    use_one_loop = args.mode in ("one_loop", "one_loop_matter_only")

    # Resolve free vs fixed cosmo params
    fixed_cosmo = parse_fix_cosmo(args.fix_cosmo)
    free_cosmo_names = [name for name in ALL_COSMO_NAMES if name not in fixed_cosmo]
    # Fill defaults for fixed params not explicitly given
    for name in ALL_COSMO_NAMES:
        if name not in fixed_cosmo and name not in free_cosmo_names:
            fixed_cosmo[name] = _DEFAULT_PARAMS[name]

    cosmo_ranges = {name: DEFAULT_COSMO_RANGES[name] for name in free_cosmo_names}

    print(f"Free cosmo params: {free_cosmo_names}")
    if fixed_cosmo:
        print(f"Fixed cosmo params: {fixed_cosmo}")

    # --- Build cosmology grid ---
    print(f"Building {'OneLoopPowerGrid' if use_one_loop else 'LinearPowerGrid'} "
          f"for mode={args.mode} ({len(free_cosmo_names)}D) ...")
    if use_one_loop:
        grid = OneLoopPowerGrid(k, Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)
    else:
        grid = LinearPowerGrid(k, Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)

    # Grid bounds (slightly inset to avoid boundary artifacts)
    axis_bounds = {}
    for i, name in enumerate(grid._axis_names):
        lo, hi = grid._axis_values[i][0], grid._axis_values[i][-1]
        margin = 0.05 * (hi - lo)
        axis_bounds[name] = (lo + margin, hi - margin)

    # --- Sample random test cosmologies ---
    rng = np.random.default_rng(args.seed)
    n_free = len(free_cosmo_names)
    test_cosmo = {}
    for name in free_cosmo_names:
        lo, hi = axis_bounds[name]
        test_cosmo[name] = rng.uniform(lo, hi, args.n_test)

    # --- Reference emulator (will be rebuilt per cosmology for exact path) ---
    first_params = dict(fixed_cosmo)
    for name in free_cosmo_names:
        first_params[name] = float(test_cosmo[name][0])
    ref_cosmo = get_cosmology(first_params)
    emu_grid = GalaxyTemplateEmulator(ref_cosmo, k, ells=ELLS, z=Z, space=SPACE, mode=args.mode)

    # --- Compare grid vs exact ---
    n_ells = len(ELLS)
    nk = len(k)
    all_exact = np.empty((args.n_test, n_ells * nk))
    all_grid = np.empty((args.n_test, n_ells * nk))
    max_errors = np.empty(args.n_test)
    mean_errors = np.empty(args.n_test)

    print(f"Testing {args.n_test} random cosmologies ...")
    for i in range(args.n_test):
        cosmo_params = dict(fixed_cosmo)
        grid_kw = {}
        for name in free_cosmo_names:
            val = float(test_cosmo[name][i])
            cosmo_params[name] = val
            grid_kw[name] = val

        # Exact path: fresh emulator from CLASS
        cosmo_exact = get_cosmology(cosmo_params)
        emu_exact = GalaxyTemplateEmulator(cosmo_exact, k, ells=ELLS, z=Z, space=SPACE, mode=args.mode)
        p_exact = emu_exact.predict(BIAS_PARAMS)

        # Grid path: interpolate then update
        if use_one_loop:
            plin, f, loop_arrays = grid.predict(**grid_kw)
            emu_grid.update_cosmology(plin, f, loop_arrays=loop_arrays)
        else:
            plin, f = grid.predict(**grid_kw)
            emu_grid.update_cosmology(plin, f)
        p_grid = emu_grid.predict(BIAS_PARAMS)

        all_exact[i] = p_exact
        all_grid[i] = p_grid

        # Fractional error with floor to handle zero-crossings
        floor = 1e-4 * np.max(np.abs(p_exact))
        frac_err = np.abs(p_grid - p_exact) / np.maximum(np.abs(p_exact), floor)
        max_errors[i] = np.max(frac_err)
        mean_errors[i] = np.mean(frac_err)

        if (i + 1) % 10 == 0 or i == 0:
            param_str = "  ".join(f"{name}={cosmo_params[name]:.4f}" for name in free_cosmo_names)
            print(f"  [{i+1:3d}/{args.n_test}] {param_str} "
                  f"max_err={max_errors[i]:.2e} mean_err={mean_errors[i]:.2e}")

    # --- Summary statistics ---
    print("\n" + "=" * 60)
    print(f"SUMMARY  (mode={args.mode}, n_test={args.n_test}, "
          f"free={free_cosmo_names})")
    print("=" * 60)
    print(f"  Max fractional error:    {np.max(max_errors):.4e}")
    print(f"  Median max error:        {np.median(max_errors):.4e}")
    print(f"  Mean fractional error:   {np.mean(mean_errors):.4e}")
    print(f"  95th pctl max error:     {np.percentile(max_errors, 95):.4e}")
    print("=" * 60)

    # --- Save results ---
    save_dict = dict(
        k=k, ells=np.array(ELLS),
        all_exact=all_exact, all_grid=all_grid,
        max_errors=max_errors, mean_errors=mean_errors,
        free_cosmo_names=np.array(free_cosmo_names),
    )
    for name in free_cosmo_names:
        save_dict[f"test_{name}"] = test_cosmo[name]
    np.savez(outdir / "accuracy_results.npz", **save_dict)
    print(f"\nResults saved to {outdir / 'accuracy_results.npz'}")

    # --- Plots ---
    _plot_frac_error_vs_k(k, all_exact, all_grid, ELLS, outdir)
    _plot_max_error_histogram(max_errors, outdir)
    _plot_max_error_strip(test_cosmo, free_cosmo_names, max_errors, outdir)
    _plot_example_comparisons(k, all_exact, all_grid, max_errors,
                              test_cosmo, free_cosmo_names, ELLS, outdir)
    print(f"Plots saved to {outdir}/")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_frac_error_vs_k(k, all_exact, all_grid, ells, outdir):
    """Fractional error vs k, one panel per ell."""
    n_ells = len(ells)
    nk = len(k)

    fig, axes = plt.subplots(1, n_ells, figsize=(5 * n_ells, 4), sharey=True)
    if n_ells == 1:
        axes = [axes]

    for idx, (ax, ell) in enumerate(zip(axes, ells)):
        sl = slice(idx * nk, (idx + 1) * nk)
        exact_block = all_exact[:, sl]
        grid_block = all_grid[:, sl]
        floor = 1e-4 * np.max(np.abs(exact_block))
        frac = np.abs(grid_block - exact_block) / np.maximum(np.abs(exact_block), floor)

        # Thin lines per cosmology
        for i in range(frac.shape[0]):
            ax.plot(k, frac[i], color="C0", alpha=0.15, lw=0.5)

        # Median and 5-95% band
        med = np.median(frac, axis=0)
        lo = np.percentile(frac, 5, axis=0)
        hi = np.percentile(frac, 95, axis=0)
        ax.plot(k, med, color="C1", lw=2, label="Median")
        ax.fill_between(k, lo, hi, color="C1", alpha=0.25, label="5-95%")

        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(rf"$\ell = {ell}$")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Fractional error")
    fig.tight_layout()
    fig.savefig(outdir / "frac_error_vs_k.png", dpi=150)
    plt.close(fig)


def _plot_max_error_histogram(max_errors, outdir):
    """Histogram of max fractional error across cosmologies."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(max_errors, bins=20, edgecolor="k", alpha=0.7)
    ax.axvline(np.median(max_errors), color="C1", ls="--", label=f"Median: {np.median(max_errors):.2e}")
    ax.set_xlabel("Max fractional error")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "max_error_histogram.png", dpi=150)
    plt.close(fig)


def _plot_max_error_strip(test_cosmo, free_cosmo_names, max_errors, outdir):
    """Strip chart: max error vs each free cosmo param."""
    n_free = len(free_cosmo_names)
    if n_free == 0:
        return

    fig, axes = plt.subplots(1, n_free, figsize=(4 * n_free, 4), squeeze=False)
    for i, name in enumerate(free_cosmo_names):
        ax = axes[0, i]
        ax.scatter(test_cosmo[name], max_errors, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(name)
        ax.set_ylabel("Max fractional error")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outdir / "error_strip.png", dpi=150)
    plt.close(fig)


def _plot_example_comparisons(k, all_exact, all_grid, max_errors,
                              test_cosmo, free_cosmo_names, ells, outdir):
    """k*P_ell for worst/median/best cosmology, grid vs exact + residual."""
    n_ells = len(ells)
    nk = len(k)

    idx_worst = np.argmax(max_errors)
    idx_best = np.argmin(max_errors)
    idx_median = np.argsort(max_errors)[len(max_errors) // 2]

    cases = [
        ("Best", idx_best),
        ("Median", idx_median),
        ("Worst", idx_worst),
    ]

    fig = plt.figure(figsize=(5 * len(cases), 7))
    gs = GridSpec(2, len(cases), height_ratios=[3, 1], hspace=0.05)

    for col, (label, idx) in enumerate(cases):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)

        for ell_i, ell in enumerate(ells):
            sl = slice(ell_i * nk, (ell_i + 1) * nk)
            exact = all_exact[idx, sl]
            grid_val = all_grid[idx, sl]

            color = f"C{ell_i}"
            ax_top.plot(k, k * exact, color=color, ls="-", lw=1.5,
                        label=rf"$\ell={ell}$ exact")
            ax_top.plot(k, k * grid_val, color=color, ls="--", lw=1.5,
                        label=rf"$\ell={ell}$ grid")

            floor = 1e-4 * np.max(np.abs(exact))
            residual = (grid_val - exact) / np.maximum(np.abs(exact), floor)
            ax_bot.plot(k, residual, color=color, lw=1)

        param_str = ", ".join(
            f"{name}={test_cosmo[name][idx]:.3f}" for name in free_cosmo_names
        )
        ax_top.set_title(f"{label} (err={max_errors[idx]:.2e})\n{param_str}",
                         fontsize=9)
        ax_top.set_ylabel(r"$k \, P_\ell(k)$")
        ax_top.tick_params(labelbottom=False)
        if col == 0:
            ax_top.legend(fontsize=6, ncol=2)

        ax_bot.axhline(0, color="gray", lw=0.5)
        ax_bot.set_xlabel(r"$k$ [$h$/Mpc]")
        ax_bot.set_ylabel("Frac. residual")

    fig.savefig(outdir / "example_comparisons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
