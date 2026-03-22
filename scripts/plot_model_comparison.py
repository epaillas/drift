"""Overlay tree-level and EFT model predictions for DS×galaxy cross spectra.

Four curves per panel:
  tree          — pqg_mu (tree-level baseline)
  eft/tree_only — pqg_eft_mu(mode="tree")  [must match tree exactly]
  eft/eft_lite  — pqg_eft_mu(mode="eft_ct")   [+ galaxy ct + DS ct; no loop promotion]
  eft/eft_full  — pqg_eft_mu(mode="eft")   [+ stochastic term]

The galaxy EFT counterterm is weighted by the DS-side amplitude bq1*W_R(k),
so DS1 (bq1 < 0) and DS5 (bq1 > 0) receive opposite-sign contributions.
Non-zero C0_GAL is set below to make this effect visible in the plots.

No measurements are loaded; this is a pure forward-model comparison.

Saves to outputs/model_comparison/model_comparison.png.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.cosmology import get_cosmology
from drift.bias import DSSplitBin
from drift.models import pqg_mu
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.eft_models import pqg_eft_mu
from drift.multipoles import compute_multipoles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPACE    = "redshift"   # "redshift" | "real"
DS_MODEL = "baseline"   # "baseline" | "rsd_selection" | "phenomenological"
Z        = 0.5
R        = 10.0
KERNEL   = "gaussian"
ELLS     = (0, 2)

B1               = 2.0
BQ1_PER_QUANTILE = {1: -1.5, 5: 1.5}
QUANTILES        = (1, 5)

# Non-zero galaxy EFT counterterm coefficient (Mpc/h)^2 — makes the corrected
# DS-amplitude weighting (bq1 * W_R) visible in the eft_lite / eft_full curves.
C0_GAL = 5.0

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"

# Wavenumber grid (no measurements file needed)
K = np.logspace(-2, 0, 80)   # 0.01 to 1 h/Mpc

# Plot colours for the four model variants
COLORS = {
    "tree":          "#1f77b4",
    "eft/tree_only": "#ff7f0e",
    "eft/eft_lite":  "#2ca02c",
    "eft/eft_full":  "#d62728",
}
LINESTYLES = {
    "tree":          "-",
    "eft/tree_only": "--",
    "eft/eft_lite":  "-.",
    "eft/eft_full":  ":",
}


def _make_eft_callable(cosmo, ds_eft, gal_eft, mode):
    """Return a (k, mu) -> P(k,mu) callable for the given EFT mode."""
    def model(k, mu):
        return pqg_eft_mu(
            k, mu, z=Z, cosmo=cosmo,
            ds_params=ds_eft, gal_params=gal_eft,
            R=R, kernel=KERNEL, space=SPACE, ds_model=DS_MODEL,
            mode=mode,
        )
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cosmo = get_cosmology()

    nrows = len(QUANTILES)
    ncols = len(ELLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, q_idx in enumerate(QUANTILES):
        bq1 = BQ1_PER_QUANTILE[q_idx]
        label_q = f"DS{q_idx}"

        # --- Tree-level model (pqg_mu) ---
        ds_tree = DSSplitBin(label=label_q, bq=bq1)

        def tree_model(k, mu, _ds=ds_tree):
            return pqg_mu(
                k, mu, z=Z, cosmo=cosmo,
                ds_params=_ds, tracer_bias=B1,
                R=R, kernel=KERNEL, space=SPACE, ds_model=DS_MODEL,
            )

        # --- EFT containers ---
        # C0_GAL is set to a non-zero value so the galaxy counterterm
        # (now weighted by bq1*W_R) is visible; DS1/DS5 show opposite signs.
        ds_eft = DSSplitBinEFT(label=label_q, bq1=bq1)
        gal_eft = GalaxyEFTParams(b1=B1, c0=C0_GAL)

        all_callables = {
            "tree":          tree_model,
            "eft/tree_only": _make_eft_callable(cosmo, ds_eft, gal_eft, "tree"),
            "eft/eft_lite":  _make_eft_callable(cosmo, ds_eft, gal_eft, "eft_ct"),
            "eft/eft_full":  _make_eft_callable(cosmo, ds_eft, gal_eft, "eft"),
        }

        # Compute multipoles for each model
        all_poles = {}
        for model_label, callable_ in all_callables.items():
            all_poles[model_label] = compute_multipoles(K, callable_, ells=ELLS)

        for col, ell in enumerate(ELLS):
            ax = axes[row, col]

            for model_label, poles in all_poles.items():
                ax.plot(
                    K, K * poles[ell],
                    color=COLORS[model_label],
                    ls=LINESTYLES[model_label],
                    lw=1.5,
                    label=model_label,
                )

            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xscale("log")
            ax.set_ylabel(rf"$k\,P_{ell}(k)$ $[(\mathrm{{Mpc}}/h)^2]$")
            ax.set_title(rf"{label_q} — $P_{ell}$", fontsize=9)

            # Legend only in the top-left panel
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    for ax in axes[-1]:
        ax.set_xlabel(r"$k$ [$h$/Mpc]")

    fig.suptitle(
        rf"DRIFT model comparison  ($z={Z}$, $R={R}$ Mpc/$h$, {SPACE} space)",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
