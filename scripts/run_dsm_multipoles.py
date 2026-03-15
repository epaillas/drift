"""Run example: compute density-split × matter power spectrum multipoles.

Loads the Planck-like cosmology and 5 density-split bin parameters from
configs/example.yaml, then evaluates the tree-level cross power spectrum

    P_{q_i, m}(k, mu) = [bq + bq_nabla*(kR)^2] * [1 + f*mu^2] * P_lin(k) * W_R(k)

for each bin q_i (DS1 ... DS5) and projects onto Legendre multipoles
P_0, P_2, P_4 via Gauss-Legendre quadrature.

Results are saved to outputs/dsm_multipoles.npz.
Run plot_dsm_multipoles.py to visualise them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift import load_config, compute_multipoles
from drift.cosmology import get_cosmology
from drift.models import pqm_mu
from drift.io import save_predictions

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfg = load_config(CONFIG_PATH)

    cosmo_params = {
        "h": cfg.cosmo.h,
        "Omega_m": cfg.cosmo.Omega_m,
        "Omega_b": cfg.cosmo.Omega_b,
        "sigma8": cfg.cosmo.sigma8,
        "n_s": cfg.cosmo.n_s,
        "engine": cfg.cosmo.engine,
    }
    cosmo = get_cosmology(cosmo_params)

    k = np.logspace(np.log10(0.005), np.log10(0.3), 100)

    multipoles_per_bin = {}
    for ds_bin in cfg.split_bins:
        print(f"  Computing {ds_bin.label} (bq={ds_bin.bq:.2f}) ...")

        def model(kk, mu, _bin=ds_bin):
            return pqm_mu(kk, mu, cfg.z, cosmo, _bin, cfg.R, kernel=cfg.kernel)

        poles = compute_multipoles(k, model, ells=(0, 2, 4))
        multipoles_per_bin[ds_bin.label] = poles

    out_path = OUTPUT_DIR / "dsm_multipoles.npz"
    save_predictions(out_path, k, multipoles_per_bin)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
