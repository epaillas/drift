# Analytic Covariance for Galaxy Power-Spectrum Multipoles

This document describes the analytic covariance currently implemented in `drift` for the galaxy auto-power spectrum multipoles,

$$P_{gg,\ell}(k), \qquad \ell \in \{0, 2, 4\}.$$

The implementation lives in [`drift/covariance.py`](/Users/epaillas/code/drift/drift/covariance.py) and is exposed through `analytic_pgg_covariance(...)`.

The current code supports:

- a disconnected Gaussian covariance for a cubic box
- an optional phenomenological connected correction labeled `effective_cng`

It does **not** yet implement the full survey covariance beyond that approximation.

---

## Scope and Assumptions

The implemented analytic covariance assumes:

- a periodic cubic volume $V$
- shell-averaged Fourier modes in $k$ bins
- no survey window convolution or mask-induced mode mixing
- a constant shot-noise contribution $P_\mathrm{shot} = 1/\bar{n}$, or an equivalent user-supplied constant `shot_noise`
- fiducial galaxy multipoles supplied as either a dictionary $\{P_\ell(k)\}$ or a flat data vector

Under these assumptions, the Gaussian term is diagonal in the radial bin index $k_i$, but different multipoles at the same $k_i$ remain correlated.

---

## Gaussian Covariance Before Multipole Projection

For a Gaussian density field in a cubic box, the disconnected covariance of the anisotropic power spectrum estimator is

$$
\mathrm{Cov}\left[P(k_i, \mu), P(k_j, \mu')\right]
= \frac{1}{N_i}\,
\delta_{ij}\,
\delta_\mathrm{D}(\mu - \mu')\,
\left[P(k_i, \mu) + P_\mathrm{shot}\right]^2,
$$

where $N_i$ is the number of Fourier modes in the shell centered on $k_i$.

In the code, the shell mode count is approximated as

$$
N_i \equiv N_\mathrm{modes}(k_i)
= \frac{V\, k_i^2\, \Delta k_i}{2\pi^2},
$$

with $\Delta k_i$ inferred from adjacent bin centers. This is implemented by `_bin_widths_from_centers(...)` and `_nmodes_cubic_box(...)`.

The quantity entering the covariance is the total power,

$$
P_\mathrm{tot}(k, \mu) = P_{gg}(k, \mu) + P_\mathrm{shot}.
$$

---

## Multipole Projection

The galaxy power-spectrum multipoles are defined by

$$
P_\ell(k) = \frac{2\ell + 1}{2}
\int_{-1}^{1} d\mu\,
\mathcal{L}_\ell(\mu)\,
P(k, \mu).
$$

Starting from the Gaussian covariance above and projecting both legs onto Legendre multipoles gives

$$
\mathrm{Cov}\left[P_\ell(k_i), P_{\ell'}(k_j)\right]
= \delta_{ij}\,
\frac{(2\ell + 1)(2\ell' + 1)}{N_i}
\int_{-1}^{1} d\mu\,
\mathcal{L}_\ell(\mu)\,
\mathcal{L}_{\ell'}(\mu)\,
\left[P(k_i, \mu) + P_\mathrm{shot}\right]^2.
$$

This is the exact structure implemented in `_gaussian_covariance(...)`, except that the $\mu$ integral is evaluated numerically with Gauss-Legendre quadrature:

$$
\int_{-1}^{1} d\mu\, f(\mu)
\;\rightarrow\;
\sum_{a=1}^{N_\mu} w_a\, f(\mu_a),
$$

with `mu_points=256` by default.

Because of the Kronecker factor $\delta_{ij}$, the Gaussian covariance is block-diagonal in $k$. However, for fixed $k_i$, the integral over

$$
\mathcal{L}_\ell(\mu)\mathcal{L}_{\ell'}(\mu)\left[P_\mathrm{tot}(k_i,\mu)\right]^2
$$

is generally non-zero when $\ell \neq \ell'$, so the multipoles remain correlated within each $k$ bin.

---

## Reconstructing $P(k,\mu)$ From Fiducial Multipoles

The current implementation takes fiducial multipoles and reconstructs the anisotropic power spectrum as

$$
P(k, \mu) \approx \sum_{\ell \in \mathrm{ells}} P_\ell(k)\, \mathcal{L}_\ell(\mu).
$$

In code:

```python
pkmu = np.zeros((len(k), len(mu)), dtype=float)
for ell in ells:
    pkmu += pole_dict[ell][:, None] * L[ell][None, :]
total_power = pkmu + shot
```

This means the covariance model is only as complete as the supplied multipole set. If only `(0, 2)` are passed, then the reconstruction omits any fiducial $\ell=4$ or higher contribution to $P(k,\mu)$.

---

## Matrix Structure Used in `drift`

For `n_k` radial bins and `n_\ell` multipoles, the covariance is stored as a dense matrix of shape

$$
(n_\ell n_k) \times (n_\ell n_k),
$$

with blocks ordered by multipole:

$$
[P_{\ell_0}(k_1), \ldots, P_{\ell_0}(k_{n_k}),
P_{\ell_1}(k_1), \ldots, P_{\ell_1}(k_{n_k}), \ldots ].
$$

Each $(\ell,\ell')$ block is diagonal in $k$ for the Gaussian term:

$$
\mathbf{C}_{\ell\ell'}^\mathrm{G}
= \mathrm{diag}\left(C_{\ell\ell'}^\mathrm{G}(k_1), \ldots, C_{\ell\ell'}^\mathrm{G}(k_{n_k})\right).
$$

This structure is tested explicitly in [`tests/test_covariance.py`](/Users/epaillas/code/drift/tests/test_covariance.py).

---

## Shot Noise, Masking, and Rescaling

The analytic interface accepts either:

- `number_density`, interpreted as $P_\mathrm{shot} = 1/\bar{n}$
- `shot_noise`, interpreted directly as a constant additive power

Exactly one must be supplied.

Two additional implementation details matter:

- `mask`: after constructing the full covariance, the code applies a Boolean mask to the flattened data vector and returns the masked covariance and precision matrix
- `rescale`: the covariance is divided by this factor, so

$$
\mathbf{C} \rightarrow \mathbf{C} / r
$$

for `rescale = r`

These behaviors are also covered in the covariance tests.

---

## The `effective_cng` Correction

The optional `terms="gaussian+effective_cng"` path adds a non-Gaussian correction on top of the Gaussian covariance, but this is **not** a first-principles trispectrum calculation.

Instead, the code builds a smooth positive-semidefinite kernel in $\ln k$,

$$
K_{ij}^{(k)} =
\exp\left[
-\frac{(\ln k_i - \ln k_j)^2}{2\,\sigma_{\ln k}^2}
\right],
$$

where `cng_coherence` sets $\sigma_{\ln k}$, and then constructs

$$
\mathbf{C}^\mathrm{eff\_cng}
= A_\mathrm{cng}\,
\left(\mathbf{r}\mathbf{r}^T\right)
\odot \mathbf{K},
$$

with:

- $A_\mathrm{cng}$ set by `cng_amplitude`
- $\mathbf{r}$ derived from the square root of the Gaussian diagonal
- higher multipoles down-weighted by an ad hoc factor
  $$
  w_\ell = \frac{1}{\sqrt{1 + \ell/2}}
  $$
- $\mathbf{K}$ built from the same $k$-kernel in every multipole block

This term is best understood as a placeholder for broad connected mode coupling that preserves the overall Gaussian scaling while introducing off-diagonal correlations in $k$.

---

## What Is Implemented

Today the code implements:

1. The disconnected Gaussian covariance for galaxy power-spectrum multipoles in a cubic box.
2. Constant shot noise through either `number_density` or `shot_noise`.
3. Optional masking and global covariance rescaling.
4. A phenomenological `effective_cng` term that adds smooth off-diagonal mode coupling.

---

## What Is Still Missing

The following pieces are **not** implemented in the current analytic covariance path:

1. A first-principles connected non-Gaussian covariance derived from the matter or galaxy trispectrum.
2. Survey-window convolution and mask-induced mixing between different $k$ bins and multipoles.
3. Super-sample covariance and response-type large-scale background modulation terms.
4. Cross-covariance involving density-split statistics such as $P_{qg}$.
5. More realistic shot-noise or stochastic covariance models beyond a constant additive term.
6. Any covariance dependence on geometry beyond the cubic-box mode-counting approximation.

So, while `effective_cng` is useful as a flexible nuisance model, it should not be interpreted as the full connected covariance predicted by perturbation theory.

---

## Current API

The implemented analytic covariance is exposed as:

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

The supported `terms` values are:

- `"gaussian"`
- `"gaussian+effective_cng"`

---

## Interpretation

The analytic covariance currently in `drift` is a controlled Gaussian box covariance with a convenient phenomenological extension for connected mode coupling. It is appropriate as a fast fiducial covariance model for `P_{gg}` analyses, but it is not yet a complete survey covariance model.
