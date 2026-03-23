# Analytic Covariance for Power-Spectrum Multipoles

This document describes the analytic covariance currently implemented in `drift` for three observable families:

- galaxy auto-power spectrum multipoles, $P_{gg,\ell}(k)$
- density-split pair multipoles, $P_{q_i q_j,\ell}(k)$
- density-split-galaxy cross-spectrum multipoles, $P_{q_i g,\ell}(k)$

The implementation lives in [`drift/covariance.py`](/Users/epaillas/code/drift/drift/covariance.py).

The current code supports:

- a disconnected Gaussian covariance for a cubic box for `P_{gg}`, `P_{q_i q_j}`, and `P_{q_i g}`
- an optional phenomenological connected correction labeled `effective_cng` for `P_{gg}` only

For density-split observables, any beyond-Gaussian term request currently raises `NotImplementedError`. Those code paths are deliberate placeholders for future extensions.

---

## Common Gaussian Setup

All covariance paths assume:

- a periodic cubic volume $V$
- shell-averaged Fourier modes in $k$ bins
- no survey window convolution or mask-induced mode mixing
- fiducial multipoles reconstructed into $P(k,\mu)$ via

$$
P(k,\mu) \approx \sum_{\ell \in \mathrm{ells}} P_\ell(k)\,\mathcal{L}_\ell(\mu)
$$

- mode counts

$$
N_i = \frac{V\,k_i^2\,\Delta k_i}{2\pi^2}
$$

with $\Delta k_i$ inferred from adjacent bin centers
- numerical $\mu$ integration via Gauss-Legendre quadrature

As in the usual Gaussian multipole treatment, the disconnected covariance is diagonal in the radial bin index $k_i$ but can correlate different multipoles at fixed $k_i$. This is the same box-level approximation commonly used as the starting point for anisotropic power-spectrum covariance models; see for example [Wadekar & Scoccimarro (2020)](https://arxiv.org/abs/1910.02914).

---

## Galaxy Multipoles $P_{gg,\ell}(k)$

The galaxy covariance is exposed through `analytic_pgg_covariance(...)`.

For a Gaussian field, the disconnected covariance of the anisotropic estimator is

$$
\mathrm{Cov}\!\left[P(k_i,\mu),P(k_j,\mu')\right]
=
\frac{\delta_{ij}\,\delta_{\mathrm D}(\mu-\mu')}{N_i}
\left[P_\mathrm{tot}(k_i,\mu)\right]^2,
$$

with

$$
P_\mathrm{tot}(k,\mu)=P_{gg}(k,\mu)+P_\mathrm{shot}.
$$

Projecting both legs onto Legendre multipoles gives

$$
\mathrm{Cov}\!\left[P_\ell(k_i),P_{\ell'}(k_j)\right]
=
\delta_{ij}\,
\frac{(2\ell+1)(2\ell'+1)}{N_i}
\int_{-1}^{1} d\mu\,
\mathcal{L}_\ell(\mu)\mathcal{L}_{\ell'}(\mu)
\left[P_\mathrm{tot}(k_i,\mu)\right]^2.
$$

This is the structure implemented by `_gaussian_covariance(...)`.

The `P_{gg}` path also supports `terms="gaussian+effective_cng"`, which adds the existing phenomenological connected correction on top of the Gaussian covariance.

---

## Density-Split Pair Multipoles $P_{q_i q_j,\ell}(k)$

The density-split pair covariance is exposed through `analytic_pqq_covariance(...)`.

Because `P_{q_i q_j}` is a cross-spectrum estimator in the general case, the disconnected Gaussian covariance uses the standard Wick-contracted cross-spectrum form. For DS pairs $(a,b)$ and $(c,d)$,

$$
\mathrm{Cov}\!\left[P_{ab}(k_i,\mu),P_{cd}(k_j,\mu')\right]
=
\frac{\delta_{ij}\,\delta_{\mathrm D}(\mu-\mu')}{N_i}
\left[
P^{\mathrm{tot}}_{ac}(k_i,\mu)\,P^{\mathrm{tot}}_{bd}(k_i,\mu)
+
P^{\mathrm{tot}}_{ad}(k_i,\mu)\,P^{\mathrm{tot}}_{bc}(k_i,\mu)
\right].
$$

Each total pair spectrum is

$$
P^{\mathrm{tot}}_{ab}(k,\mu)=P_{ab}(k,\mu)+N_{ab},
$$

where $N_{ab}$ is a constant pairwise noise term supplied by the caller.

Projecting both covariance legs onto multipoles yields

$$
\mathrm{Cov}\!\left[P_{ab,\ell}(k_i),P_{cd,\ell'}(k_j)\right]
=
\delta_{ij}\,
\frac{(2\ell+1)(2\ell'+1)}{N_i}
\int_{-1}^{1} d\mu\,
\mathcal{L}_\ell(\mu)\mathcal{L}_{\ell'}(\mu)
\left[
P^{\mathrm{tot}}_{ac}P^{\mathrm{tot}}_{bd}
+
P^{\mathrm{tot}}_{ad}P^{\mathrm{tot}}_{bc}
\right].
$$

This is the structure implemented in `_gaussian_dspair_covariance(...)`.

`pair_order` defines the observed DS-pair blocks in the returned covariance. The code canonicalizes pairs symmetrically, so `("DS2", "DS1")` is treated as `("DS1", "DS2")`. Because the Wick contractions require intermediate pairs such as $(a,c)$ and $(b,d)$, the fiducial `poles` and `shot_noise` dictionaries must provide all canonical DS pairs over the bins appearing in `pair_order`.

The returned DS covariance uses pair-major ordering, then multipole-major ordering within each pair:

$$
[P_{q_1 q_1,\ell_0}(k_1),\ldots,P_{q_1 q_1,\ell_0}(k_{n_k}),
P_{q_1 q_1,\ell_1}(k_1),\ldots,
P_{q_1 q_2,\ell_0}(k_1),\ldots].
$$

---

## Density-Split-Galaxy Multipoles $P_{q_i g,\ell}(k)$

The density-split-galaxy covariance is exposed through `analytic_pqg_covariance(...)`.

For DS labels $a$ and $b$, the Gaussian disconnected covariance of the anisotropic cross-spectrum estimator follows the usual Wick contraction for two cross spectra with one common galaxy leg:

$$
\mathrm{Cov}\!\left[P_{ag}(k_i,\mu),P_{bg}(k_j,\mu')\right]
=
\frac{\delta_{ij}\,\delta_{\mathrm D}(\mu-\mu')}{N_i}
\left[
P^{\mathrm{tot}}_{ab}(k_i,\mu)\,P^{\mathrm{tot}}_{gg}(k_i,\mu)
+
P^{\mathrm{tot}}_{ag}(k_i,\mu)\,P^{\mathrm{tot}}_{bg}(k_i,\mu)
\right].
$$

The total spectra entering the contractions are modeled as

$$
P^{\mathrm{tot}}_{gg}(k,\mu)=P_{gg}(k,\mu)+N_{gg},
$$

$$
P^{\mathrm{tot}}_{ab}(k,\mu)=P_{q_a q_b}(k,\mu)+N_{ab},
$$

$$
P^{\mathrm{tot}}_{ag}(k,\mu)=P_{q_a g}(k,\mu)+N_{ag}.
$$

Here:

- $N_{gg}$ is a constant galaxy shot-noise term
- $N_{ab}$ is a constant DS-pair noise term, provided for every canonical pair needed by the selected quantiles
- $N_{ag}$ is an optional constant DS×g cross-noise term, provided per DS label and defaulting naturally to zero if the caller chooses that model

Projecting both covariance legs onto multipoles gives

$$
\mathrm{Cov}\!\left[P_{ag,\ell}(k_i),P_{bg,\ell'}(k_j)\right]
=
\delta_{ij}\,
\frac{(2\ell+1)(2\ell'+1)}{N_i}
\int_{-1}^{1} d\mu\,
\mathcal{L}_\ell(\mu)\mathcal{L}_{\ell'}(\mu)
\left[
P^{\mathrm{tot}}_{ab}P^{\mathrm{tot}}_{gg}
+
P^{\mathrm{tot}}_{ag}P^{\mathrm{tot}}_{bg}
\right].
$$

This is the structure implemented in `_gaussian_dsg_covariance(...)`.

The returned DSG covariance uses label-major ordering, then multipole-major ordering within each label:

$$
[P_{q_1 g,\ell_0}(k_1),\ldots,P_{q_1 g,\ell_0}(k_{n_k}),
P_{q_1 g,\ell_1}(k_1),\ldots,
P_{q_2 g,\ell_0}(k_1),\ldots].
$$

This `P_{qg}` block structure is the natural Gaussian companion to the density-split clustering measurements used in recent analyses such as [Paillas et al. (2023)](https://doi.org/10.1093/mnras/stad1017).

---

## Beyond-Gaussian Placeholders

For density-split observables, the only implemented term is:

- `"gaussian"`

If `terms` includes `effective_cng` or any other non-Gaussian alias:

- `analytic_pqq_covariance(...)` raises `NotImplementedError`
- `analytic_pqg_covariance(...)` raises `NotImplementedError`

This is an intentional placeholder for future work on:

- connected DS trispectrum terms
- connected DS×g trispectrum terms
- super-sample covariance
- survey-window mixing
- joint covariance blocks involving `P_{gg}`, `P_{q_i q_j}`, and `P_{q_i g}`

For context on where a beyond-Gaussian multipole treatment would need to go next, see for example [Kobayashi et al. (2023)](https://arxiv.org/abs/2308.08593).

---

## Current APIs

The implemented covariance interfaces are:

```python
analytic_pgg_covariance(
    k,
    poles,
    ells=(0, 2, 4),
    volume=...,
    number_density=None,
    shot_noise=None,
    mask=None,
    rescale=1.0,
    terms="gaussian",
    mu_points=256,
    cng_amplitude=0.0,
    cng_coherence=0.35,
)
```

```python
analytic_pqq_covariance(
    k,
    poles,
    ells=(0, 2, 4),
    volume=...,
    pair_order=...,
    shot_noise=...,
    mask=None,
    rescale=1.0,
    terms="gaussian",
    mu_points=256,
)
```

```python
analytic_pqg_covariance(
    k,
    pqg_poles,
    pqq_poles,
    pgg_poles,
    ells=(0, 2, 4),
    volume=...,
    ds_labels=...,
    galaxy_shot_noise=...,
    ds_pair_shot_noise=...,
    ds_cross_shot_noise=None,
    mask=None,
    rescale=1.0,
    terms="gaussian",
    mu_points=256,
)
```

---

## What Is Still Missing

The following pieces are not implemented in the current analytic covariance path:

1. First-principles connected non-Gaussian covariance for either `P_{gg}`, `P_{q_i q_j}`, or `P_{q_i g}`
2. Beyond-Gaussian density-split covariance of any kind
3. Survey-window convolution and mask-induced mode mixing
4. Super-sample covariance and response terms
5. Joint cross-covariance blocks involving multiple observable families
6. Covariance dependence on geometry beyond the cubic-box approximation

So the current implementation should be read as a fast Gaussian box covariance model, with a phenomenological connected extension for `P_{gg}` only.
