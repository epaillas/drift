<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/drift_logo_dark_mode.png" width="200">
    <img src="docs/drift_logo_light_mode.png" width="200" alt="DRIFT logo">
  </picture>
</div>

# DRIFT — Density-split Renormalized Inference and Field Theory

DRIFT computes perturbation-theory predictions for density-split (DS) and galaxy power-spectrum multipoles in redshift space. Measurements are performed separately (e.g. with [ACM](https://github.com/epaillas/acm)); this package is theory-prediction only.

---

## Philosophy

Density-split statistics divide a survey volume into quantiles $q_i$ ranked by local smoothed density. Each quantile carries a characteristic large-scale bias that encodes its environment. DRIFT models the Fourier-space cross-spectra of these quantiles with matter or a galaxy tracer, projects them onto Legendre multipoles, and optionally transforms them to configuration space.

The tree-level models use Kaiser redshift-space perturbation theory. The higher-order models are EFT descriptions of large-scale structure: their one-loop matter and bias kernels are computed from SPT, then assembled together with EFT counterterms and nuisance terms. Inference-time speed is achieved through analytic template decomposition: bias-independent loop integrals are pre-computed once per cosmology and re-assembled for any parameter combination at negligible cost.

---

## Available theory models

### Tree-level (Kaiser)

| Function | Observable |
|---|---|
| `ds_matter_pkmu` / `pqm_mu` | $P_{q_i, m}(k, \mu)$ — DS × matter |
| `ds_galaxy_pkmu` / `pqg_mu` | $P_{q_i, g}(k, \mu)$ — DS × galaxy |
| `dspair_pkmu` / `pqq_mu` | $P_{q_i q_j}(k, \mu)$ — DS-pair auto |
| `galaxy_pkmu` / `pgg_mu` | $P_{gg}(k, \mu)$ — galaxy auto (Kaiser) |

### EFT models

| Function | Observable | Modes |
|---|---|---|
| `ds_galaxy_eft_pkmu` / `pqg_eft_mu` | $P_{q_i, g}(k, \mu)$ | `tree`, `eft_ct`, `eft`, `one_loop` |
| `dspair_eft_pkmu` / `pqq_eft_mu` | $P_{q_i q_j}(k, \mu)$ | `tree`, `eft_ct`, `eft`, `one_loop` |
| `galaxy_eft_pkmu` / `pgg_eft_mu` | $P_{gg}(k, \mu)$ | `tree`, `eft_ct`, `eft`, `one_loop` |

Mode meanings: `tree` — tree-level EFT containers with no counterterms or stochastic terms; `eft_ct` — tree + EFT counterterms; `eft` — eft_ct + stochastic terms; `one_loop` — full one-loop EFT model: SPT one-loop matter/bias terms plus EFT counterterms, FoG, and stochastic terms.

All functions accept `space='redshift'` (default) or `space='real'`, and `ds_model` variants `'baseline'`, `'rsd_selection'`, or `'phenomenological'`.

### Configuration space

`compute_ds_galaxy_correlation_multipoles` and `compute_dspair_correlation_multipoles` wrap the Fourier-space models with an FFTLog Hankel transform to produce $\xi_\ell(s)$.

### Multipole projection

`compute_multipoles` and `project_multipole` project any $P(k, \mu)$ callable onto Legendre multipoles via Gauss-Legendre quadrature.

---

## Additional capabilities

| Component | Purpose |
|---|---|
| `analytic_pgg_covariance` / `analytic_pqq_covariance` / `analytic_pqg_covariance` | Analytic Gaussian covariance for power-spectrum multipoles, with optional effective CNG and SSC corrections |
| `analytic_xigg_covariance` / `analytic_xiqq_covariance` / `analytic_xiqg_covariance` | Same, propagated to configuration space |
| `GalaxyPowerSpectrumEmulator` | Template emulator for $P_{gg}$ — separates linear and non-linear bias contributions for fast MCMC |
| `DensitySplitGalaxyPowerSpectrumEmulator` | Template emulator for $P_{q_i, g}$ |
| `TaylorEmulator` | Taylor-expansion emulator around a fiducial parameter point, for arbitrary callables |
| `MarginalizedLikelihood` | Analytic marginalisation over linear nuisance parameters |

---

## Key parameter containers

| Class | Parameters |
|---|---|
| `DensitySplitBin` | `bq`, `cq`, `beta_q` |
| `DensitySplitEFTParameters` | `bq1`, `bq2`, `bqK2`, `bq_nabla2`, `beta_q` |
| `GalaxyEFTParameters` | `b1`, `b2`, `bs2`, `b3nl`, `sigma_fog`, `c0`, `c2`, `c4`, `s0`, `s2` |

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

# Cosmology (Planck 2018 defaults)
cosmo = drift.get_cosmology()

k  = np.logspace(np.log10(0.005), np.log10(0.3), 100)
mu = np.linspace(-1, 1, 200)

# Tree-level DS × galaxy
ds_bin = drift.DensitySplitBin(label="DS1", bq=-1.5)
pkmu = drift.ds_galaxy_pkmu(k, mu, z=0.5, cosmo=cosmo,
                             ds_params=ds_bin, tracer_bias=2.0, R=10.0)  # (nk, nmu)

# Project to multipoles
poles = drift.compute_multipoles(k, lambda kk, mm: drift.ds_galaxy_pkmu(
    kk, mm, z=0.5, cosmo=cosmo, ds_params=ds_bin, tracer_bias=2.0, R=10.0))
P0, P2, P4 = poles[0], poles[2], poles[4]

# EFT DS × galaxy
ds_eft = drift.DensitySplitEFTParameters(label="DS1", bq1=-1.5)
gal    = drift.GalaxyEFTParameters(b1=2.0, c0=0.5)
pkmu_eft = drift.ds_galaxy_eft_pkmu(k, mu, z=0.5, cosmo=cosmo,
                                     ds_params=ds_eft, gal_params=gal,
                                     R=10.0, mode="eft_ct")
```

---

## Repository layout

```
drift/
├── drift/
│   ├── theory/
│   │   ├── density_split/   # DS×m, DS×g, DS-pair models (tree + EFT)
│   │   └── galaxy/          # Galaxy auto-power models (tree + EFT)
│   ├── emulators/           # Template emulators for DS×g and Pgg
│   ├── utils/
│   │   ├── cosmology.py     # cosmoprimo wrappers, LinearPowerGrid
│   │   ├── kernels.py       # Gaussian and top-hat smoothing kernels
│   │   ├── multipoles.py    # Legendre projection, FFTLog transforms
│   │   ├── one_loop.py      # SPT P22, P13, bias loops
│   │   ├── ir_resummation.py # BAO damping, wiggle/no-wiggle split
│   │   └── rsd.py           # Kaiser RSD factors
│   ├── covariance.py        # Analytic covariance matrices
│   ├── taylor.py            # Taylor-expansion emulator
│   ├── analytic_marginalization.py  # Marginalised Gaussian likelihood
│   ├── synthetic.py         # Noiseless synthetic data vectors
│   └── io.py                # Save/load, mock covariance helpers
├── configs/
│   └── example.yaml
├── scripts/                 # Analysis and plotting scripts
└── tests/
```

---

## Running tests

```bash
pytest tests/
```
