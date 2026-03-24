"""Compute density-split×galaxy power-spectrum and correlation-function multipoles."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift import compute_ds_galaxy_correlation_multipoles, compute_multipoles
from drift.theory.density_split.config import load_config
from drift.theory.density_split.eft_config import load_density_split_eft_config
from drift.theory.density_split.eft_power_spectrum import pqg_eft_mu
from drift.theory.density_split.power_spectrum import pqg_mu
from drift.utils.cosmology import get_cosmology

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def _build_cosmology(cfg):
    return get_cosmology(
        {
            "h": cfg.cosmo.h,
            "Omega_m": cfg.cosmo.Omega_m,
            "Omega_b": cfg.cosmo.Omega_b,
            "sigma8": cfg.cosmo.sigma8,
            "n_s": cfg.cosmo.n_s,
            "engine": cfg.cosmo.engine,
        }
    )


def _build_metadata(args, z, ds_model):
    return {
        "observable": "pqg",
        "mode": args.mode,
        "space": args.space,
        "ds_model": ds_model,
        "z": z,
        "q": args.q,
    }


def save_ds_galaxy_multipoles(path, k, s, poles_by_label, xi_by_label, metadata):
    """Save DS×galaxy P_ell(k) and xi_ell(s) arrays to an .npz bundle."""
    arrays = {
        "k": np.asarray(k, dtype=float),
        "s": np.asarray(s, dtype=float),
        "ells": np.asarray(sorted(next(iter(poles_by_label.values())).keys()), dtype=int),
        "labels": np.asarray(sorted(poles_by_label), dtype=str),
        "observable": np.asarray(str(metadata["observable"])),
        "mode": np.asarray(str(metadata["mode"])),
        "space": np.asarray(str(metadata["space"])),
        "ds_model": np.asarray(str(metadata["ds_model"])),
        "z": np.asarray(float(metadata["z"])),
        "q": np.asarray(float(metadata["q"])),
    }
    for label, poles in poles_by_label.items():
        for ell, values in poles.items():
            arrays[f"{label}_P{ell}"] = np.asarray(values, dtype=float)
    for label, poles in xi_by_label.items():
        for ell, values in poles.items():
            arrays[f"{label}_XI{ell}"] = np.asarray(values, dtype=float)
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH")
    parser.add_argument("--eft-config", type=Path, default=None, metavar="PATH")
    parser.add_argument("--mode", choices=["tree", "eft_ct", "eft", "one_loop"], default="tree")
    parser.add_argument("--space", choices=["redshift", "real"], default="redshift")
    parser.add_argument("--ds-model", type=str, default=None, metavar="NAME")
    parser.add_argument("--kmin", type=float, default=1.0e-4, metavar="VALUE")
    parser.add_argument("--kmax", type=float, default=1.0, metavar="VALUE")
    parser.add_argument("--nk", type=int, default=512, metavar="N")
    parser.add_argument("--q", type=float, default=1.0, metavar="VALUE")
    parser.add_argument("--minfolds", type=int, default=4, metavar="N")
    parser.add_argument("--output", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    if args.kmin <= 0.0:
        raise ValueError("--kmin must be positive.")
    if args.kmax <= args.kmin:
        raise ValueError("--kmax must be larger than --kmin.")
    if args.nk < 2:
        raise ValueError("--nk must be at least 2.")

    use_eft = args.eft_config is not None
    if args.mode != "tree" and not use_eft:
        raise ValueError("Non-tree DS×galaxy correlation predictions require --eft-config.")

    if use_eft:
        cfg = load_density_split_eft_config(args.eft_config)
        if cfg.gal_params is None:
            raise ValueError("The EFT config must define gal_params for DS×galaxy predictions.")
        z = cfg.z
        ds_model = cfg.ds_model if args.ds_model is None else args.ds_model
        bins = cfg.split_bins
        cosmo = _build_cosmology(cfg)
    else:
        cfg = load_config(args.config)
        if cfg.tracer_bias is None:
            raise ValueError("tracer_bias must be set in the config file for DS×galaxy predictions.")
        z = cfg.z
        ds_model = "baseline" if args.ds_model is None else args.ds_model
        bins = cfg.split_bins
        cosmo = _build_cosmology(cfg)

    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)
    ells = (0, 2, 4)
    poles_by_label = {}
    xi_by_label = {}

    for ds_bin in bins:
        if use_eft:
            def model(kk, mu, _bin=ds_bin):
                return pqg_eft_mu(
                    kk,
                    mu,
                    z=z,
                    cosmo=cosmo,
                    ds_params=_bin,
                    gal_params=cfg.gal_params,
                    R=cfg.R,
                    kernel=cfg.kernel,
                    space=args.space,
                    ds_model=ds_model,
                    mode=args.mode,
                )

            s, xi = compute_ds_galaxy_correlation_multipoles(
                k,
                z=z,
                cosmo=cosmo,
                ds_params=ds_bin,
                gal_params=cfg.gal_params,
                R=cfg.R,
                kernel=cfg.kernel,
                space=args.space,
                ds_model=ds_model,
                mode=args.mode,
                ells=ells,
                q=args.q,
                extrap="edge",
                fftlog_kwargs={"minfolds": args.minfolds},
            )
        else:
            def model(kk, mu, _bin=ds_bin):
                return pqg_mu(
                    kk,
                    mu,
                    z=z,
                    cosmo=cosmo,
                    ds_params=_bin,
                    tracer_bias=cfg.tracer_bias,
                    R=cfg.R,
                    kernel=cfg.kernel,
                    space=args.space,
                    ds_model=ds_model,
                )

            s, xi = compute_ds_galaxy_correlation_multipoles(
                k,
                z=z,
                cosmo=cosmo,
                ds_params=ds_bin,
                tracer_bias=cfg.tracer_bias,
                R=cfg.R,
                kernel=cfg.kernel,
                space=args.space,
                ds_model=ds_model,
                ells=ells,
                q=args.q,
                extrap="edge",
                fftlog_kwargs={"minfolds": args.minfolds},
            )

        poles_by_label[ds_bin.label] = compute_multipoles(k, model, ells=ells)
        xi_by_label[ds_bin.label] = xi

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_DIR / f"dsg_xi_multipoles_{args.mode}_{args.space}_{ds_model}.npz"

    save_ds_galaxy_multipoles(
        out_path,
        k=k,
        s=s,
        poles_by_label=poles_by_label,
        xi_by_label=xi_by_label,
        metadata=_build_metadata(args, z=z, ds_model=ds_model),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
