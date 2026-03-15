"""Configuration dataclasses and YAML loader for DRIFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from .bias import DSSplitBin


@dataclass
class CosmoConfig:
    """Cosmological parameters."""

    h: float = 0.6766
    Omega_m: float = 0.3111
    Omega_b: float = 0.049
    sigma8: float = 0.8102
    n_s: float = 0.9665
    engine: str = "class"


@dataclass
class DriftConfig:
    """Top-level DRIFT configuration."""

    cosmo: CosmoConfig = field(default_factory=CosmoConfig)
    z: float = 0.5
    R: float = 10.0          # smoothing scale in Mpc/h
    kernel: str = "gaussian"
    split_bins: List[DSSplitBin] = field(default_factory=list)
    tracer_bias: Optional[float] = None


def load_config(path: str | Path) -> DriftConfig:
    """Load a DRIFT configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    DriftConfig
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

    # Density-split bins
    bins_raw = raw.get("split_bins", [])
    split_bins = [
        DSSplitBin(
            label=b["label"],
            bq=float(b["bq"]),
            cq=float(b.get("cq", b.get("bq_nabla", 0.0))),
            beta_q=float(b.get("beta_q", 0.0)),
        )
        for b in bins_raw
    ]

    return DriftConfig(
        cosmo=cosmo,
        z=float(raw.get("z", 0.5)),
        R=float(raw.get("R", 10.0)),
        kernel=str(raw.get("kernel", "gaussian")),
        split_bins=split_bins,
        tracer_bias=raw.get("tracer_bias", None),
    )
