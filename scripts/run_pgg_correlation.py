"""Compute galaxy power-spectrum and correlation-function multipoles."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift import compute_correlation_multipoles, compute_multipoles
from drift.theory.density_split.config import load_config
from drift.theory.galaxy.bias import GalaxyEFTParameters
from drift.theory.galaxy.power_spectrum import galaxy_eft_pkmu, galaxy_pkmu
from drift.utils.cosmology import get_cosmology

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def _build_galaxy_params(args, default_b1):
    return GalaxyEFTParameters(
        b1=float(default_b1 if args.b1 is None else args.b1),
        b2=float(args.b2),
        bs2=float(args.bs2),
        b3nl=float(args.b3nl),
        sigma_fog=float(args.sigma_fog),
        c0=float(args.c0),
        c2=float(args.c2),
        c4=float(args.c4),
        s0=float(args.s0),
        s2=float(args.s2),
    )


def _power_model(args, cosmo, gal_params):
    if args.mode == "tree":
        return lambda kk, mu: galaxy_pkmu(
            kk,
            mu,
            z=args.z,
            cosmo=cosmo,
            b1=gal_params.b1,
            space=args.space,
        )

    return lambda kk, mu: galaxy_eft_pkmu(
        kk,
        mu,
        z=args.z,
        cosmo=cosmo,
        gal_params=gal_params,
        space=args.space,
        mode=args.mode,
    )


def save_galaxy_multipoles(path, k, s, poles, xi_poles, metadata):
    """Save galaxy P_ell(k) and xi_ell(s) arrays to an .npz bundle."""
    arrays = {
        "k": np.asarray(k, dtype=float),
        "s": np.asarray(s, dtype=float),
        "ells": np.asarray(sorted(poles), dtype=int),
        "mode": np.asarray(str(metadata["mode"])),
        "space": np.asarray(str(metadata["space"])),
        "z": np.asarray(float(metadata["z"])),
        "q": np.asarray(float(metadata["q"])),
    }
    for ell, values in poles.items():
        arrays[f"P{ell}"] = np.asarray(values, dtype=float)
    for ell, values in xi_poles.items():
        arrays[f"XI{ell}"] = np.asarray(values, dtype=float)
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH")
    parser.add_argument("--mode", choices=["tree", "eft_ct", "eft", "one_loop"], default="eft")
    parser.add_argument("--space", choices=["redshift", "real"], default="redshift")
    parser.add_argument("--z", type=float, default=None, metavar="VALUE")
    parser.add_argument("--b1", type=float, default=None, metavar="VALUE")
    parser.add_argument("--b2", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--bs2", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--b3nl", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--sigma-fog", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--c0", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--c2", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--c4", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--s0", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--s2", type=float, default=0.0, metavar="VALUE")
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

    cfg = load_config(args.config)
    z = float(cfg.z if args.z is None else args.z)
    b1 = cfg.tracer_bias if cfg.tracer_bias is not None else 1.8
    gal_params = _build_galaxy_params(args, b1)

    cosmo = get_cosmology(
        {
            "h": cfg.cosmo.h,
            "Omega_m": cfg.cosmo.Omega_m,
            "Omega_b": cfg.cosmo.Omega_b,
            "sigma8": cfg.cosmo.sigma8,
            "n_s": cfg.cosmo.n_s,
            "engine": cfg.cosmo.engine,
        }
    )

    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)
    model = _power_model(argparse.Namespace(**{**vars(args), "z": z}), cosmo, gal_params)

    ells = (0, 2, 4)
    poles = compute_multipoles(k, model, ells=ells)
    s, xi_poles = compute_correlation_multipoles(
        k,
        model,
        ells=ells,
        q=args.q,
        fftlog_kwargs={"minfolds": args.minfolds},
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_DIR / f"pgg_xi_multipoles_{args.mode}_{args.space}.npz"

    save_galaxy_multipoles(
        out_path,
        k=k,
        s=s,
        poles=poles,
        xi_poles=xi_poles,
        metadata={"mode": args.mode, "space": args.space, "z": z, "q": args.q},
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
