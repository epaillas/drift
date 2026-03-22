"""EFT configuration dataclass and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from .config import CosmoConfig
from .eft_bias import DSSplitBinEFT, GalaxyEFTParams


@dataclass
class EFTConfig:
    """Top-level EFT configuration."""

    cosmo: CosmoConfig = field(default_factory=CosmoConfig)
    z: float = 0.5
    R: float = 10.0
    kernel: str = "gaussian"
    ds_model: str = "baseline"
    mode: str = "eft_ct"
    split_bins: List[DSSplitBinEFT] = field(default_factory=list)
    gal_params: Optional[GalaxyEFTParams] = None
    loop_kwargs: dict = field(default_factory=lambda: {
        "q_min": 1e-4,
        "q_max": 10.0,
        "n_q_22": 128,
        "n_mu_22": 128,
        "n_q_13": 256,
    })


def load_eft_config(path) -> EFTConfig:
    """Load an EFTConfig from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    EFTConfig
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
        DSSplitBinEFT(
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
        gal_params = GalaxyEFTParams(
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

    return EFTConfig(
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
