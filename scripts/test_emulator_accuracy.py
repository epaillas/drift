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

from drift.cosmology import get_cosmology, LinearPowerGrid, OneLoopPowerGrid
from drift.galaxy_emulator import GalaxyTemplateEmulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z = 0.5
ELLS = (0, 2, 4)
SPACE = "redshift"

BIAS_PARAMS = dict(b1=2.0, c0=5.0, c2=2.0, c4=0.0, s0=100.0, s2=0.0, b2=0.5, bs2=-0.5)


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

    # --- Build cosmology grid ---
    print(f"Building {'OneLoopPowerGrid' if use_one_loop else 'LinearPowerGrid'} "
          f"for mode={args.mode} ...")
    if use_one_loop:
        grid = OneLoopPowerGrid(k, Z)
    else:
        grid = LinearPowerGrid(k, Z)

    # Grid bounds (slightly inset to avoid boundary artifacts)
    s8_lo, s8_hi = grid._s8_range
    om_lo, om_hi = grid._om_range
    margin_s8 = 0.05 * (s8_hi - s8_lo)
    margin_om = 0.05 * (om_hi - om_lo)

    # --- Sample random test cosmologies ---
    rng = np.random.default_rng(args.seed)
    test_s8 = rng.uniform(s8_lo + margin_s8, s8_hi - margin_s8, args.n_test)
    test_om = rng.uniform(om_lo + margin_om, om_hi - margin_om, args.n_test)

    # --- Reference emulator (will be rebuilt per cosmology for exact path) ---
    # Build one grid-path emulator that we update
    ref_cosmo = get_cosmology({"sigma8": float(test_s8[0]), "Omega_m": float(test_om[0])})
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
        s8, om = float(test_s8[i]), float(test_om[i])
        cosmo_params = {"sigma8": s8, "Omega_m": om}

        # Exact path: fresh emulator from CLASS
        cosmo_exact = get_cosmology(cosmo_params)
        emu_exact = GalaxyTemplateEmulator(cosmo_exact, k, ells=ELLS, z=Z, space=SPACE, mode=args.mode)
        p_exact = emu_exact.predict(BIAS_PARAMS)

        # Grid path: interpolate then update
        if use_one_loop:
            plin, f, loop_arrays = grid.predict(s8, om)
            emu_grid.update_cosmology(plin, f, loop_arrays=loop_arrays)
        else:
            plin, f = grid.predict(s8, om)
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
            print(f"  [{i+1:3d}/{args.n_test}] sigma8={s8:.4f} Omega_m={om:.4f} "
                  f"max_err={max_errors[i]:.2e} mean_err={mean_errors[i]:.2e}")

    # --- Summary statistics ---
    print("\n" + "=" * 60)
    print(f"SUMMARY  (mode={args.mode}, n_test={args.n_test})")
    print("=" * 60)
    print(f"  Max fractional error:    {np.max(max_errors):.4e}")
    print(f"  Median max error:        {np.median(max_errors):.4e}")
    print(f"  Mean fractional error:   {np.mean(mean_errors):.4e}")
    print(f"  95th pctl max error:     {np.percentile(max_errors, 95):.4e}")
    print("=" * 60)

    # --- Save results ---
    np.savez(
        outdir / "accuracy_results.npz",
        k=k, ells=np.array(ELLS),
        test_sigma8=test_s8, test_omega_m=test_om,
        all_exact=all_exact, all_grid=all_grid,
        max_errors=max_errors, mean_errors=mean_errors,
    )
    print(f"\nResults saved to {outdir / 'accuracy_results.npz'}")

    # --- Plots ---
    _plot_frac_error_vs_k(k, all_exact, all_grid, ELLS, outdir)
    _plot_max_error_histogram(max_errors, outdir)
    _plot_error_heatmap(test_s8, test_om, max_errors, grid, outdir)
    _plot_example_comparisons(k, all_exact, all_grid, max_errors,
                              test_s8, test_om, ELLS, outdir)
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


def _plot_error_heatmap(test_s8, test_om, max_errors, grid, outdir):
    """Scatter in (sigma8, Omega_m) colored by max error, grid nodes overlaid."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(test_om, test_s8, c=max_errors, cmap="viridis",
                    edgecolors="k", linewidths=0.3, s=40, zorder=5)
    plt.colorbar(sc, ax=ax, label="Max fractional error")

    # Overlay grid nodes
    s8_lo, s8_hi = grid._s8_range
    om_lo, om_hi = grid._om_range
    # Recover grid node positions from the interpolator
    s8_nodes = grid._plin_interp.grid[0]
    om_nodes = grid._plin_interp.grid[1]
    om_grid, s8_grid = np.meshgrid(om_nodes, s8_nodes)
    ax.scatter(om_grid.ravel(), s8_grid.ravel(), marker="+", color="red",
               s=20, alpha=0.5, zorder=4, label="Grid nodes")

    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "error_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_example_comparisons(k, all_exact, all_grid, max_errors,
                              test_s8, test_om, ells, outdir):
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

        ax_top.set_title(f"{label} (err={max_errors[idx]:.2e})\n"
                         rf"$\sigma_8$={test_s8[idx]:.3f}, $\Omega_m$={test_om[idx]:.3f}",
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
