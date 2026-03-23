# Analytic Covariance for Power-Spectrum Multipoles

This document describes the analytic covariance currently implemented in `drift` for three observable families:

- galaxy auto-power spectrum multipoles, $P_{gg,\ell}(k)$
- density-split pair multipoles, $P_{q_i q_j,\ell}(k)$
- density-split-galaxy cross-spectrum multipoles, $P_{q_i g,\ell}(k)$

The implementation lives in [`drift/covariance.py`](/Users/epaillas/code/drift/drift/covariance.py).

The current code supports:

- a disconnected Gaussian covariance for a cubic box for `P_{gg}`, `P_{q_i q_j}`, and `P_{q_i g}`
- an optional phenomenological connected correction labeled `effective_cng` for `P_{gg}`, `P_{q_i q_j}`, and `P_{q_i g}`
- a density-only super-sample covariance term labeled `ssc` for `P_{gg}`, `P_{q_i q_j}`, and `P_{q_i g}`

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

As in the usual Gaussian multipole treatment, the disconnected covariance is diagonal in the radial bin index $k_i$ but can correlate different multipoles at fixed $k_i$. This is the same box-level approximation commonly used as the starting point for anisotropic power-spectrum covariance models; see for example [Wadekar & Scoccimarro (2020)](https://doi.org/10.1103/PhysRevD.102.123517).

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

The `P_{gg}` path also supports `terms` including `effective_cng` and `ssc`, which add the existing phenomenological connected correction and the density-only super-sample covariance term on top of the Gaussian covariance.

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

## Super-Sample Covariance

The implemented SSC term follows the standard response-based density-only approximation

$$
\mathrm{Cov}_{\rm SSC}[X_i, X_j] = \sigma_b^2 \, R_i \, R_j,
$$

where $\sigma_b^2$ is the variance of the survey-scale background density mode and $R_i \equiv \partial X_i / \partial \delta_b$ is the response of a data-vector element to that long-wavelength isotropic density perturbation. This is the same rank-1 SSC structure emphasized in [Takada & Hu (2013)](https://doi.org/10.1103/PhysRevD.87.123504), [Li, Hu & Takada (2014)](https://doi.org/10.1103/PhysRevD.89.083519), and the broader response overview in [Bayer et al. (2023)](https://doi.org/10.1103/PhysRevD.108.043521).

To keep the covariance API coherent with the current implementation, the code infers the response directly from the supplied fiducial spectra rather than from a separate-universe tracer model. It reconstructs the fiducial anisotropic spectrum and applies

$$
R_P(k,\mu)
\approx
\left(
\frac{47}{21}
- \frac{1}{3}\frac{\partial}{\partial \ln k}
\right)
P(k,\mu),
$$

which combines a universal growth term with the standard dilation term. The response is then projected back to multipoles using

$$
R_\ell(k)
=
\frac{2\ell+1}{2}
\int_{-1}^{1} d\mu \,
\mathcal{L}_\ell(\mu)\,
R_P(k,\mu).
$$

The projected response is flattened in the same ordering as the covariance data vector and used in the outer product above.

This is an intentionally simplified SSC model. In particular:

- only the isotropic density SSC is included
- tidal SSC is not implemented
- shot-noise terms are treated as having zero SSC response
- tracer-specific separate-universe response corrections are not modeled explicitly

The low-level covariance APIs take `ssc_sigma_b2` explicitly. The package also provides `estimate_ssc_sigma_b2(volume, z, cosmo=None)`, which evaluates

$$
\sigma_b^2(R)
=
\int \frac{k^2 dk}{2\pi^2}
P_{\rm lin}(k, z)\,
W_{\rm TH}^2(kR),
$$

with an equivalent-volume spherical top-hat window and

$$
R = \left(\frac{3V}{4\pi}\right)^{1/3}.
$$

This estimator is meant as a coherent default for the current box-style covariance interface, not as a full survey-window treatment.

For all currently supported observable families, the SSC term is added on top of the Gaussian covariance and any enabled `effective_cng` term:

- `analytic_pgg_covariance(...)`
- `analytic_pqq_covariance(...)`
- `analytic_pqg_covariance(...)`

---

## Beyond-Gaussian Terms

For all currently supported observable families, the implemented terms are:

- `"gaussian"`
- `"gaussian+effective_cng"`
- `"gaussian+ssc"`
- `"gaussian+effective_cng+ssc"`

For density-split observables, `effective_cng` should still be read as a phenomenological connected extension rather than a first-principles DS trispectrum model.

Future work is still needed on:

- connected DS trispectrum terms
- connected DS×g trispectrum terms
- tidal super-sample covariance
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
    ssc_sigma_b2=None,
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
    cng_amplitude=0.0,
    cng_coherence=0.35,
    ssc_sigma_b2=None,
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
    cng_amplitude=0.0,
    cng_coherence=0.35,
    ssc_sigma_b2=None,
)
```

---

## What Is Still Missing

The following pieces are not implemented in the current analytic covariance path:

1. First-principles connected non-Gaussian covariance for either `P_{gg}`, `P_{q_i q_j}`, or `P_{q_i g}`
2. First-principles beyond-Gaussian density-split covariance
3. Survey-window convolution and mask-induced mode mixing
4. Tidal super-sample covariance and tracer-calibrated response terms
5. Joint cross-covariance blocks involving multiple observable families
6. Covariance dependence on geometry beyond the cubic-box approximation

So the current implementation should be read as a fast Gaussian box covariance model, augmented by a phenomenological connected correction and a density-only SSC approximation for the currently supported observable families.
