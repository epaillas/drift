"""EFT configuration dataclass and YAML loader for density-split theory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from .bias import DensitySplitEFTParameters
from .config import CosmoConfig
from ..galaxy.bias import GalaxyEFTParameters


@dataclass
class DensitySplitEFTConfig:
    """Top-level EFT configuration for density-split theory."""

    cosmo: CosmoConfig = field(default_factory=CosmoConfig)
    z: float = 0.5
    R: float = 10.0
    kernel: str = "gaussian"
    ds_model: str = "baseline"
    mode: str = "eft_ct"
    split_bins: List[DensitySplitEFTParameters] = field(default_factory=list)
    gal_params: Optional[GalaxyEFTParameters] = None
    loop_kwargs: dict = field(default_factory=lambda: {
        "q_min": 1e-4,
        "q_max": 10.0,
        "n_q_22": 128,
        "n_mu_22": 128,
        "n_q_13": 256,
    })


def load_density_split_eft_config(path) -> DensitySplitEFTConfig:
    """Load a density-split EFT configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    DensitySplitEFTConfig
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    # Cosmology
    cosmo_raw = raw.get("cosmology", {})
    cosmo = CosmoConfig(
        h=cosmo_raw.get("h", 0.6766),
        Omega_m=cosmo_raw.get("Omega_m", 0.3111),
        Omega_b=cosmo_raw.get("Omega_b", 0.049),
        sigma8=cosmo_raw.get("sigma8", 0.8102),
        n_s=cosmo_raw.get("n_s", 0.9665),
        engine=cosmo_raw.get("engine", "class"),
    )

    # Density-split EFT bins
    bins_raw = raw.get("split_bins", [])
    split_bins = [
        DensitySplitEFTParameters(
            label=b["label"],
            bq1=float(b["bq1"]),
            bq2=float(b.get("bq2", 0.0)),
            bqK2=float(b.get("bqK2", 0.0)),
            bq_nabla2=float(b.get("bq_nabla2", 0.0)),
            beta_q=float(b.get("beta_q", 0.0)),
        )
        for b in bins_raw
    ]

    # Galaxy EFT params
    gal_raw = raw.get("gal_params", None)
    gal_params = None
    if gal_raw is not None:
        gal_params = GalaxyEFTParameters(
            b1=float(gal_raw["b1"]),
            b2=float(gal_raw.get("b2", 0.0)),
            bs2=float(gal_raw.get("bs2", 0.0)),
            b3nl=float(gal_raw.get("b3nl", 0.0)),
            c0=float(gal_raw.get("c0", 0.0)),
            c2=float(gal_raw.get("c2", 0.0)),
            c4=float(gal_raw.get("c4", 0.0)),
            s0=float(gal_raw.get("s0", 0.0)),
            s2=float(gal_raw.get("s2", 0.0)),
        )

    # Loop kwargs
    lk_raw = raw.get("loop_kwargs", {})
    loop_kwargs = {
        "q_min": float(lk_raw.get("q_min", 1e-4)),
        "q_max": float(lk_raw.get("q_max", 10.0)),
        "n_q_22": int(lk_raw.get("n_q_22", 128)),
        "n_mu_22": int(lk_raw.get("n_mu_22", 128)),
        "n_q_13": int(lk_raw.get("n_q_13", 256)),
    }

    return DensitySplitEFTConfig(
        cosmo=cosmo,
        z=float(raw.get("z", 0.5)),
        R=float(raw.get("R", 10.0)),
        kernel=str(raw.get("kernel", "gaussian")),
        ds_model=str(raw.get("ds_model", "baseline")),
        mode=str(raw.get("mode", "eft_ct")),
        split_bins=split_bins,
        gal_params=gal_params,
        loop_kwargs=loop_kwargs,
    )


EFTConfig = DensitySplitEFTConfig
load_eft_config = load_density_split_eft_config
