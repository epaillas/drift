"""Compute density-split pair power-spectrum and correlation-function multipoles."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift import compute_dspair_correlation_multipoles, compute_multipoles
from drift.theory.density_split.config import load_config
from drift.theory.density_split.eft_config import load_density_split_eft_config
from drift.theory.density_split.eft_power_spectrum import pqq_eft_mu
from drift.theory.density_split.power_spectrum import pqq_mu
from drift.utils.cosmology import get_cosmology

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "example.yaml"
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def pair_to_key(pair):
    """Return a canonical on-disk token for a DS pair."""
    return f"{pair[0]}__{pair[1]}"


def key_to_pair(key):
    """Parse a canonical DS pair token."""
    left, right = key.split("__", 1)
    return left, right


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


def _pair_order(labels):
    return tuple((label, label) for label in labels)


def _parse_pairs(values, labels):
    if values is None:
        return _pair_order(labels)

    index = {label: idx for idx, label in enumerate(labels)}
    pairs = []
    for value in values:
        left, right = value.split("-", 1)
        if left not in index or right not in index:
            raise ValueError(f"Unknown pair selection '{value}'. Available labels: {labels}.")
        if index[left] <= index[right]:
            pairs.append((left, right))
        else:
            pairs.append((right, left))
    return tuple(pairs)


def save_dspair_multipoles(path, k, s, poles_by_pair, xi_by_pair, metadata):
    """Save DS-pair P_ell(k) and xi_ell(s) arrays to an .npz bundle."""
    arrays = {
        "k": np.asarray(k, dtype=float),
        "s": np.asarray(s, dtype=float),
        "ells": np.asarray(sorted(next(iter(poles_by_pair.values())).keys()), dtype=int),
        "pairs": np.asarray([pair_to_key(pair) for pair in poles_by_pair], dtype=str),
        "observable": np.asarray(str(metadata["observable"])),
        "mode": np.asarray(str(metadata["mode"])),
        "space": np.asarray(str(metadata["space"])),
        "ds_model": np.asarray(str(metadata["ds_model"])),
        "z": np.asarray(float(metadata["z"])),
        "q": np.asarray(float(metadata["q"])),
    }
    for pair, poles in poles_by_pair.items():
        token = pair_to_key(pair)
        for ell, values in poles.items():
            arrays[f"{token}_P{ell}"] = np.asarray(values, dtype=float)
    for pair, poles in xi_by_pair.items():
        token = pair_to_key(pair)
        for ell, values in poles.items():
            arrays[f"{token}_XI{ell}"] = np.asarray(values, dtype=float)
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="PATH")
    parser.add_argument("--eft-config", type=Path, default=None, metavar="PATH")
    parser.add_argument("--mode", choices=["tree", "eft_ct", "eft", "one_loop"], default="tree")
    parser.add_argument("--space", choices=["redshift", "real"], default="redshift")
    parser.add_argument("--ds-model", type=str, default=None, metavar="NAME")
    parser.add_argument("--pairs", nargs="+", default=None, metavar="DSI-DSJ")
    parser.add_argument("--sqq0", type=float, default=0.0, metavar="VALUE")
    parser.add_argument("--sqq2", type=float, default=0.0, metavar="VALUE")
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
        raise ValueError("Non-tree DS-pair correlation predictions require --eft-config.")

    if use_eft:
        cfg = load_density_split_eft_config(args.eft_config)
        z = cfg.z
        ds_model = cfg.ds_model if args.ds_model is None else args.ds_model
        bins = {ds_bin.label: ds_bin for ds_bin in cfg.split_bins}
        cosmo = _build_cosmology(cfg)
    else:
        cfg = load_config(args.config)
        z = cfg.z
        ds_model = "baseline" if args.ds_model is None else args.ds_model
        bins = {ds_bin.label: ds_bin for ds_bin in cfg.split_bins}
        cosmo = _build_cosmology(cfg)

    selected_pairs = _parse_pairs(args.pairs, tuple(bins))
    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)
    ells = (0, 2, 4)
    poles_by_pair = {}
    xi_by_pair = {}

    for pair in selected_pairs:
        ds_a = bins[pair[0]]
        ds_b = bins[pair[1]]

        if use_eft:
            def model(kk, mu, _a=ds_a, _b=ds_b):
                return pqq_eft_mu(
                    kk,
                    mu,
                    z=z,
                    cosmo=cosmo,
                    ds_params_a=_a,
                    ds_params_b=_b,
                    R=cfg.R,
                    kernel=cfg.kernel,
                    space=args.space,
                    ds_model=ds_model,
                    mode=args.mode,
                    sqq0=args.sqq0,
                    sqq2=args.sqq2,
                )

            s, xi = compute_dspair_correlation_multipoles(
                k,
                z=z,
                cosmo=cosmo,
                ds_params_a=ds_a,
                ds_params_b=ds_b,
                R=cfg.R,
                kernel=cfg.kernel,
                space=args.space,
                ds_model=ds_model,
                mode=args.mode,
                sqq0=args.sqq0,
                sqq2=args.sqq2,
                ells=ells,
                q=args.q,
                extrap="edge",
                fftlog_kwargs={"minfolds": args.minfolds},
            )
        else:
            def model(kk, mu, _a=ds_a, _b=ds_b):
                return pqq_mu(
                    kk,
                    mu,
                    z=z,
                    cosmo=cosmo,
                    ds_params_a=_a,
                    ds_params_b=_b,
                    R=cfg.R,
                    kernel=cfg.kernel,
                    space=args.space,
                    ds_model=ds_model,
                )

            s, xi = compute_dspair_correlation_multipoles(
                k,
                z=z,
                cosmo=cosmo,
                ds_params_a=ds_a,
                ds_params_b=ds_b,
                R=cfg.R,
                kernel=cfg.kernel,
                space=args.space,
                ds_model=ds_model,
                ells=ells,
                q=args.q,
                extrap="edge",
                fftlog_kwargs={"minfolds": args.minfolds},
            )

        poles_by_pair[pair] = compute_multipoles(k, model, ells=ells)
        xi_by_pair[pair] = xi

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output
    if out_path is None:
        out_path = OUTPUT_DIR / f"pqq_xi_multipoles_{args.mode}_{args.space}_{ds_model}.npz"

    save_dspair_multipoles(
        out_path,
        k=k,
        s=s,
        poles_by_pair=poles_by_pair,
        xi_by_pair=xi_by_pair,
        metadata={
            "observable": "pqq",
            "mode": args.mode,
            "space": args.space,
            "ds_model": ds_model,
            "z": z,
            "q": args.q,
        },
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
