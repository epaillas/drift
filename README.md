<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/drift_logo_dark_mode.png" width="200">
    <img src="docs/drift_logo_light_mode.png" width="200" alt="DRIFT logo">
  </picture>
</div>

# DRIFT — Density-split Renormalized Inference and Field Theory

DRIFT computes theory predictions for density-split (DS) power spectrum multipoles in redshift space, including tree-level, EFT counterterms, and one-loop corrections. Measurements are made separately (e.g. with [ACM](https://github.com/epaillas/acm)); this package is theory-prediction only.

---

## Physics

Density-split statistics divide the survey volume into quantiles ranked by local density. Each quantile $q_i$ has a characteristic bias that encodes its environment. DRIFT models the cross power spectra of these quantiles with matter or a galaxy tracer.

### Density-split × matter

$$P_{q_i, m}(k, \mu, z) = \left[b_{q_i} + b_{q_i}^\nabla (kR)^2\right] \left[1 + f(z)\,\mu^2\right] P_\mathrm{lin}(k, z)\, W_R(k)$$

### Density-split × galaxy

$$P_{q_i, g}(k, \mu, z) = \left[b_{q_i} + b_{q_i}^\nabla (kR)^2\right] \left[b_1 + f(z)\,\mu^2\right] P_\mathrm{lin}(k, z)\, W_R(k)$$

where $W_R(k)$ is one smoothing kernel factor (cross-spectrum convention), $f(z)$ is the linear growth rate, $b_{q_i}$ is the linear density-split bias, and $b_{q_i}^\nabla$ is the Laplacian derivative bias.

### Multipoles

$$P_\ell(k) = \frac{2\ell+1}{2} \int_{-1}^{1} d\mu\; P(k,\mu)\, \mathcal{L}_\ell(\mu)$$

for $\ell = 0, 2, 4$, evaluated via Gauss-Legendre quadrature.

**Units:** $k$ in $h/\mathrm{Mpc}$, $P(k)$ in $(\mathrm{Mpc}/h)^3$.

### EFT / One-loop model (pqg_eft_mu)

Three modes are available via the `mode` argument:
- `tree_only` — reproduces `pqg_mu` exactly
- `eft_lite`  — tree + partial one-loop matter power spectrum + galaxy EFT counterterm
- `eft_full`  — eft_lite + stochastic contributions

The EFT bias containers are `DSSplitBinEFT` (adds bq2, bqK2, bq_nabla2) and
`GalaxyEFTParams` (adds b2, bs2, b3nl, c0, c2, c4, s0, s2).

---

## Installation

```bash
pip install -e /path/to/drift/
```

Requires: `numpy`, `scipy`, `matplotlib`, `cosmoprimo`, `pyyaml`.

---

## Quick start

```python
import numpy as np
import drift
from drift.cosmology import get_cosmology
from drift.models import pqm_mu

# Build cosmology
cosmo = get_cosmology()  # Planck 2018 defaults

# Define a density-split bin
ds_bin = drift.DSSplitBin(label="DS1", bq=-1.5)

# Evaluate P(k, mu) on a grid
k  = np.logspace(np.log10(0.005), np.log10(0.3), 100)
mu = np.linspace(-1, 1, 200)
pkmu = pqm_mu(k, mu, z=0.5, cosmo=cosmo, ds_params=ds_bin, R=10.0)  # shape (nk, nmu)

# Project onto multipoles
poles = drift.compute_multipoles(k, lambda kk, mu: pqm_mu(kk, mu, 0.5, cosmo, ds_bin, 10.0))
P0, P2, P4 = poles[0], poles[2], poles[4]
```

### EFT / one-loop model

```python
from drift.eft_models import pqg_eft_mu
from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
from drift.one_loop import compute_one_loop_matter

ds_bin = DSSplitBinEFT(label="DS1", bq1=-1.5)
gal    = GalaxyEFTParams(b1=2.0)

plin_func = lambda kk: get_linear_power(cosmo, kk, z=0.5)
p1loop = compute_one_loop_matter(k, plin_func)["p1loop"]

pkmu = pqg_eft_mu(k, mu, z=0.5, cosmo=cosmo,
                  ds_params=ds_bin, gal_params=gal, R=10.0,
                  mode="eft_ct",
                  loop_kwargs={"p1loop_precomputed": p1loop})
```

### Using a config file

```python
cfg = drift.load_config("configs/example.yaml")
```

See `configs/example.yaml` for the full YAML schema.

---

## Repository layout

```
drift/
├── drift/
│   ├── cosmology.py    # cosmoprimo wrapper: P_lin(k,z), f(z)
│   ├── kernels.py      # Gaussian and top-hat smoothing kernels W_R(k)
│   ├── bias.py         # DSSplitBin dataclass
│   ├── rsd.py          # Kaiser RSD factors
│   ├── models.py       # pqm_mu, pqg_mu — P(k,mu) at tree level
│   ├── multipoles.py   # Legendre projection via Gauss-Legendre quadrature
│   ├── config.py       # DriftConfig dataclass + YAML loader
│   ├── io.py           # save/load .npz and text files
│   ├── eft_bias.py     # DSSplitBinEFT, GalaxyEFTParams dataclasses
│   ├── eft_terms.py    # EFT counterterm and stochastic term implementations
│   ├── eft_models.py   # pqg_eft_mu — EFT P(k,mu) assembler
│   ├── eft_config.py   # EFTConfig dataclass + YAML loader
│   └── one_loop.py     # One-loop matter power spectrum (P22, P13, SPT)
├── configs/
│   └── example.yaml
├── scripts/
│   ├── run_dsm_multipoles.py
│   ├── run_dsg_multipoles.py
│   ├── plot_dsm_multipoles.py
│   ├── plot_dsg_multipoles.py
│   ├── plot_rsd_comparison.py
│   └── plot_model_comparison.py
└── tests/
    ├── test_kernels.py
    ├── test_multipoles.py
    ├── test_models.py
    ├── test_eft_models.py
    └── test_one_loop.py
```

---

## Running the examples

```bash
# Density-split × matter
python scripts/run_dsm_multipoles.py   # produces outputs/dsm_multipoles.npz
python scripts/plot_dsm_multipoles.py  # produces outputs/dsm_multipoles.png

# Density-split × galaxy  (requires tracer_bias in the config)
python scripts/run_dsg_multipoles.py   # produces outputs/dsg_multipoles.npz
python scripts/plot_dsg_multipoles.py  # produces outputs/dsg_multipoles.png

# Model comparison (tree vs EFT modes)
python scripts/plot_model_comparison.py  # → outputs/model_comparison/model_comparison.png
```

---

## Running tests

```bash
pytest tests/
```
