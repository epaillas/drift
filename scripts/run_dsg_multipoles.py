"""Compute density-split × galaxy power spectrum multipoles.

Loads the Planck-like cosmology and 5 density-split bin parameters from
configs/example.yaml, then evaluates the tree-level cross power spectrum
for each bin q_i (DS1 ... DS5) using two DS selection models:

  - "baseline"       : P = [bq + cq*(kR)^2] * [b1 + f*mu^2] * P_lin * W_R
  - "rsd_selection"  : P = [bq + cq*(kR)^2 + f*mu^2] * [b1 + f*mu^2] * P_lin * W_R

Results are saved to:
  outputs/dsg_multipoles_baseline.npz
  outputs/dsg_multipoles_rsd_selection.npz

Run plot_dsg_multipoles.py to visualise them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift import load_config, compute_multipoles
from drift.utils.cosmology import get_cosmology
from drift.theory.density_split.power_spectrum import pqg_mu
from drift.io import save_predictions

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def run_model(cfg, cosmo, k, ds_model: str) -> dict:
    """Compute multipoles for all bins with the given ds_model."""
    multipoles_per_bin = {}
    for ds_bin in cfg.split_bins:
        print(f"    {ds_bin.label} (bq={ds_bin.bq:.2f}) ...")

        def model(kk, mu, _bin=ds_bin):
            return pqg_mu(
                kk, mu, cfg.z, cosmo, _bin, cfg.tracer_bias, cfg.R,
                kernel=cfg.kernel, ds_model=ds_model,
            )

        poles = compute_multipoles(k, model, ells=(0, 2, 4))
        multipoles_per_bin[ds_bin.label] = poles
    return multipoles_per_bin


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfg = load_config(CONFIG_PATH)

    if cfg.tracer_bias is None:
        raise ValueError(
            "tracer_bias must be set in the config file for ds-galaxy predictions. "
            "Uncomment the 'tracer_bias' line in configs/example.yaml."
        )

    cosmo_params = {
        "h": cfg.cosmo.h,
        "Omega_m": cfg.cosmo.Omega_m,
        "Omega_b": cfg.cosmo.Omega_b,
        "sigma8": cfg.cosmo.sigma8,
        "n_s": cfg.cosmo.n_s,
        "engine": cfg.cosmo.engine,
    }
    cosmo = get_cosmology(cosmo_params)

    print(f"  Tracer bias b1 = {cfg.tracer_bias}")

    k = np.logspace(np.log10(0.005), np.log10(0.3), 100)

    for ds_model in ("baseline", "rsd_selection"):
        print(f"\n  Model: {ds_model}")
        multipoles_per_bin = run_model(cfg, cosmo, k, ds_model)
        out_path = OUTPUT_DIR / f"dsg_multipoles_{ds_model}.npz"
        save_predictions(out_path, k, multipoles_per_bin)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
