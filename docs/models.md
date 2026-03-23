# Theoretical Models in `drift`

This document describes the theoretical models implemented in the `drift` package for computing galaxy power spectrum multipoles. Three observables are supported:

- **$P_{gg}(\mathbf{k})$** — Galaxy auto-power spectrum
- **$P_{qg}(\mathbf{k})$** — Density-split × galaxy cross-power spectrum
- **$P_{q_i q_j}(\mathbf{k})$** — Density-split pair power spectrum

Both are functions of wavenumber $k$ and angle-to-line-of-sight cosine $\mu$, and are projected onto Legendre multipoles $\ell = 0, 2, 4$ for comparison with observations.

---

## Common Ingredients

### Linear power spectrum and growth rate

All models are built from the linear matter power spectrum $P_\mathrm{lin}(k)$ and the linear growth rate

$$f \equiv \frac{d \ln D}{d \ln a},$$

both evaluated at the target redshift $z$, using [cosmoprimo](https://github.com/cosmodesi/cosmoprimo) as the Boltzmann interface.

### Smoothing kernels

The density-split field is defined by smoothing the galaxy density field with a kernel $W_R(k)$ of radius $R$ (in $h^{-1}\mathrm{Mpc}$). Two kernels are supported:

**Gaussian:**

$$W_R(k) = \exp\!\left(-\frac{k^2 R^2}{2}\right)$$

**Top-hat** (spherical in real space):

$$W_R(k) = \frac{3\left[\sin(kR) - kR\cos(kR)\right]}{(kR)^3}$$

with the regularization $W_R(0) = 1$.

### Legendre multipole projection

The 2D spectrum $P(k, \mu)$ is projected onto multipoles via

$$P_\ell(k) = \frac{2\ell + 1}{2} \int_{-1}^{1} \mathcal{L}_\ell(\mu)\, P(k, \mu)\, d\mu,$$

where $\mathcal{L}_\ell$ is the Legendre polynomial of order $\ell$. The emulators precompute the analytic Legendre moments $M_{\ell,n} = \frac{2\ell+1}{2}\int_{-1}^1 \mathcal{L}_\ell(\mu)\,\mu^n\,d\mu$ to avoid repeated numerical quadrature.

---

## Galaxy Auto-Power Spectrum $P_{gg}(k, \mu)$

### Tree level (`mode="tree"`)

**Redshift space (Kaiser model):**

$$P_{gg}(k, \mu) = \left(b_1 + f\mu^2\right)^2 P_\mathrm{lin}(k)$$

**Real space:**

$$P_{gg}(k) = b_1^2\, P_\mathrm{lin}(k)$$

### EFT counterterms (`mode="eft_ct"`)

Extends the tree level with EFT higher-derivative corrections and a leading-order finger-of-god (FoG) term:

$$P_{gg}(k, \mu) = \left(b_1 + f\mu^2\right)^2 P_\mathrm{lin}(k)$$
$$- 2k^2\left(c_0 + c_2\mu^2 + c_4\mu^4\right)\left(b_1 + f\mu^2\right) P_\mathrm{lin}(k)$$
$$- \sigma_\mathrm{FoG}\, k^2 P_\mathrm{lin}(k)\left(b_1^2\mu^2 + 2b_1 f\mu^4 + f^2\mu^6\right)$$

The three counterterms $c_0$, $c_2$, $c_4$ renormalize the EFT expansion in $\mu^0$, $\mu^2$, and $\mu^4$, respectively. The FoG term captures the leading-order pairwise velocity dispersion effect, with $\sigma_\mathrm{FoG} \approx \sigma_v^2$ (units: $(h^{-1}\mathrm{Mpc})^2$).

### EFT with stochasticity (`mode="eft"`)

Extends `eft_ct` with a stochastic term:

$$P_{gg}(k, \mu) = P_{gg}^\mathrm{eft\_ct}(k, \mu) + s_0 + s_2\,(k\mu)^2$$

The $s_0$ term is isotropic shot noise; $s_2\,(k\mu)^2$ is a scale- and angle-dependent correction.

### Full one-loop EFT (`mode="one_loop"`)

Adds the complete one-loop SPT corrections in redshift space:

$$P_{gg}(k, \mu) = \underbrace{\left(b_1 + f\mu^2\right)^2 P_\mathrm{lin}(k)}_{\text{tree}}$$
$$+ \underbrace{b_1^2\left[P_{22}(k) + P_{13}(k)\right]}_{\text{density loop, }\mu^0}$$
$$+ \underbrace{2b_1 f\mu^2\left[P_{22}^{d\theta}(k) + P_{13}^{d\theta}(k)\right]}_{\text{density–velocity loop, }\mu^2}$$
$$+ \underbrace{f^2\mu^4\left[P_{22}^{\theta\theta}(k) + P_{13}^{\theta\theta}(k)\right]}_{\text{velocity loop, }\mu^4}$$
$$+ \underbrace{2b_1 b_2\, I_{12}(k) + 2b_1 b_{s^2} J_{12}(k) + b_2^2\, I_{22}(k) + 2b_2 b_{s^2}\, I_{2K}(k) + b_{s^2}^2\, J_{22}(k) + 4b_1 b_{3\mathrm{nl}}\, I_{b_{3\mathrm{nl}}}(k)}_{\text{bias loops, }\mu^0}$$
$$+ \underbrace{2f\mu^2\left[b_2\, I_{12}^v(k) + b_{s^2}\, J_{12}^v(k)\right]}_{\text{RSD bias–velocity loops, }\mu^2}$$
$$- 2k^2\left(c_0 + c_2\mu^2 + c_4\mu^4\right)\left(b_1 + f\mu^2\right) P_\mathrm{lin}(k)$$
$$- \sigma_\mathrm{FoG}\, k^2 P_\mathrm{lin}(k)\left(b_1^2\mu^2 + 2b_1 f\mu^4 + f^2\mu^6\right)$$
$$+ s_0 + s_2\,(k\mu)^2$$

### IR resummation (optional)

When `ir_resum=True`, the linear power spectrum in the tree-level Kaiser factor is split into a wiggle part $P_w(k)$ and a no-wiggle part $P_{nw}(k)$. An IR damping factor $D_\mathrm{IR}(k,\mu)$ is applied to the wiggle component to model BAO broadening by large-scale displacements:

$$P_\mathrm{lin}(k) \to P_{nw}(k) + D_\mathrm{IR}(k,\mu)\, P_w(k)$$

### Parameter table for $P_{gg}$

| Parameter | Description | Units | Active modes |
|---|---|---|---|
| $b_1$ | Linear galaxy bias | — | all |
| $b_2$ | Quadratic galaxy bias | — | `one_loop` |
| $b_{s^2}$ | Tidal galaxy bias | — | `one_loop` |
| $b_{3\mathrm{nl}}$ | Non-local cubic bias | — | `one_loop` |
| $c_0$ | Isotropic EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $c_2$ | $\mu^2$ EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $c_4$ | $\mu^4$ EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $\sigma_\mathrm{FoG}$ | FoG damping coefficient ($\approx \sigma_v^2$) | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $s_0$ | White-noise stochastic amplitude | $(h^{-1}\mathrm{Mpc})^3$ | `eft`, `one_loop` |
| $s_2$ | $k^2$ stochastic amplitude | $(h^{-1}\mathrm{Mpc})^5$ | `eft`, `one_loop` |

---

## Density-Split × Galaxy Cross-Power Spectrum $P_{qg}(k, \mu)$

The cross-spectrum between a density-split bin $q_i$ and the galaxy field depends on (i) the **EFT mode** controlling the loop order and (ii) the **DS model** controlling how the density-split field acquires redshift-space distortions.

### DS angular models

Three prescriptions are available for the DS angular structure in redshift space.

#### `ds_model="baseline"`

The density-split field is selected in real space; only the galaxy field acquires Kaiser RSD:

$$P_{qg}(k, \mu) = b_{q1}\, P_\mathrm{lin}(k)\, W_R(k) \left(b_1 + f\mu^2\right)$$

Non-zero multipoles: $\ell = 0, 2$ only ($P_4 = 0$ at tree level).

#### `ds_model="rsd_selection"`

Both fields are selected in redshift space. The DS field acquires the Kaiser factor $(1 + f\mu^2)$:

$$P_{qg}(k, \mu) = b_{q1}\left(1 + f\mu^2\right)\left(b_1 + f\mu^2\right) P_\mathrm{lin}(k)\, W_R(k)$$

Non-zero multipoles: $\ell = 0, 2, 4$ (the product generates $\mu^4$).

#### `ds_model="phenomenological"`

A free anisotropy parameter $\beta_q$ models the DS angular structure:

$$P_{qg}(k, \mu) = \left(b_{q1} + \beta_q f\mu^2\right)\left(b_1 + f\mu^2\right) P_\mathrm{lin}(k)\, W_R(k)$$

Non-zero multipoles: $\ell = 0, 2, 4$.

### EFT modes for $P_{qg}$

All modes include the appropriate DS angular structure from the choice of `ds_model`.

#### `mode="tree"`

Tree-level only (as described above for each DS model).

#### `mode="eft_ct"`

Tree level plus EFT higher-derivative corrections. Two counterterms are added:

**Galaxy counterterm** — renormalizes the galaxy tracer response:

$$P_{ct}^{gal}(k, \mu) = -k^2\left(c_0 + c_2\mu^2 + c_4\mu^4\right) P_{q \times \mathrm{lin}}(k, \mu)$$

where $P_{q \times \mathrm{lin}}(k, \mu)$ is the tree-level DS × linear matter cross-spectrum (no galaxy bias factor), encoding the full angular structure of the DS model.

**DS higher-derivative counterterm** — renormalizes the density-split field:

$$P_{ct}^{DS}(k, \mu) = b_{q,\nabla^2}\, (kR)^2\, \tilde{P}_{qg}^\mathrm{tree}(k, \mu)$$

where $\tilde{P}_{qg}^\mathrm{tree}$ is the tree-level model evaluated with $b_{q1} = 1$, while preserving the selected DS angular prescription. For `phenomenological`, this means the $\beta_q f\mu^2$ anisotropy is retained.

#### `mode="eft"`

Extends `eft_ct` with a stochastic term:

$$P_{qg}(k, \mu) = P_{qg}^{eft\_ct}(k, \mu) + s_0 + s_2\, k^2$$

Note: the stochastic term here is isotropic in $\mu$ (unlike the $P_{gg}$ stochastic, which carries $\mu^2$ dependence).

#### `mode="one_loop"`

The leading $b_{q1} \times b_1$ contribution is promoted to one-loop matter accuracy. The galaxy–matter cross-spectrum is computed to one-loop order and then multiplied by the DS angular factor. The galaxy bias and RSD loops are also included:

$$P_{gm}^{(0)}(k) = b_1 \left[P_\mathrm{lin} + P_{22} + P_{13}\right] W_R + \left[b_2\, I_{12} + b_{s^2}\, J_{12} + 2b_{3\mathrm{nl}}\, I_{b_{3\mathrm{nl}}}\right] W_R$$

$$P_{gm}^{(2)}(k) = f \left[P_{22}^{d\theta} + P_{13}^{d\theta}\right] W_R + f\left[b_2\, I_{12}^v + b_{s^2}\, J_{12}^v\right] W_R$$

The full cross-spectrum is then:

$$P_{qg}(k, \mu) = \mathrm{DS\text{-}factor}(k, \mu) \times \left[P_{gm}^{(0)}(k) + P_{gm}^{(2)}(k)\,\mu^2\right]$$

where $\mathrm{DS\text{-}factor}(k, \mu)$ is $b_{q1}$, $b_{q1}(1 + f\mu^2)$, or $(b_{q1} + \beta_q f\mu^2)$ for `baseline`, `rsd_selection`, and `phenomenological`, respectively.

EFT counterterms, a FoG term, and stochastic are added on top:

$$- \sigma_\mathrm{FoG}\, k^2 P_\mathrm{lin} W_R \cdot \mathrm{DS\text{-}factor}(k, \mu) \cdot \left(b_1\mu^2 + f\mu^4\right)$$

$$+ s_0 + s_2\, k^2\mu^2$$

> **Note:** The `bq2` and `bqK2` quadratic DS bias terms in `one_loop` mode are not implemented. The code raises `NotImplementedError` if either is set to a non-zero value.

### Parameter table for $P_{qg}$

| Parameter | Description | Units | Active modes |
|---|---|---|---|
| $b_{q1}$ | Linear DS bias | — | all |
| $b_{q2}$ | Quadratic DS bias | — | `one_loop` (not yet implemented) |
| $b_{qK^2}$ | Tidal DS bias | — | `one_loop` (not yet implemented) |
| $b_{q,\nabla^2}$ | DS higher-derivative counterterm | — | `eft_ct`, `eft`, `one_loop` |
| $\beta_q$ | Phenomenological DS anisotropy | — | `phenomenological` model |
| $b_1$ | Linear galaxy bias | — | all |
| $b_2$ | Quadratic galaxy bias | — | `one_loop` |
| $b_{s^2}$ | Tidal galaxy bias | — | `one_loop` |
| $b_{3\mathrm{nl}}$ | Non-local cubic bias | — | `one_loop` |
| $c_0$ | Isotropic EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $c_2$ | $\mu^2$ EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $c_4$ | $\mu^4$ EFT counterterm | $(h^{-1}\mathrm{Mpc})^2$ | `eft_ct`, `eft`, `one_loop` |
| $\sigma_\mathrm{FoG}$ | FoG damping coefficient | $(h^{-1}\mathrm{Mpc})^2$ | `one_loop` |
| $s_0$ | White-noise stochastic amplitude | $(h^{-1}\mathrm{Mpc})^3$ | `eft`, `one_loop` |
| $s_2$ | $k^2$ stochastic amplitude | $(h^{-1}\mathrm{Mpc})^5$ | `eft`, `one_loop` |

---

## Density-Split Pair Power Spectrum $P_{q_i q_j}(k, \mu)$

The density-split bins are treated as effective tracers of the smoothed density field. At leading order, each bin contributes a smoothed response factor built from the linear matter field and the smoothing kernel $W_R(k)$. This follows the same large-scale bias logic used for density-split clustering analyses, where each quantile is modeled as a biased tracer of the smoothed environment field.

For each leg we define

$$B_{q_i}(k) \equiv b_{q_i} + c_{q_i}(kR)^2,$$

so the pair spectrum always carries the matter kernel $P_\mathrm{lin}(k)\,W_R(k)^2$.

The model is symmetric by construction:

$$P_{q_i q_j}(k,\mu) = P_{q_j q_i}(k,\mu).$$

### DS angular models

#### `ds_model="baseline"`

The density split is selected in real space on both legs, so the pair spectrum is isotropic even when evaluated in redshift space:

$$P_{q_i q_j}(k,\mu) = B_{q_i}(k)\,B_{q_j}(k)\,P_\mathrm{lin}(k)\,W_R(k)^2.$$

Only the monopole is non-zero at tree level.

#### `ds_model="rsd_selection"`

Both density-split legs are selected in redshift space and acquire Kaiser factors:

$$P_{q_i q_j}(k,\mu) = B_{q_i}(k)\,B_{q_j}(k)\,\left(1 + f\mu^2\right)^2 P_\mathrm{lin}(k)\,W_R(k)^2.$$

Writing $A_{ij}(k) \equiv B_{q_i}(k) B_{q_j}(k) P_\mathrm{lin}(k) W_R(k)^2$, the tree spectrum is

$$P_{q_i q_j}(k,\mu) = A_{ij}(k)\left[1 + 2f\mu^2 + f^2\mu^4\right],$$

with multipoles

$$P_0 = A_{ij}\left(1 + \frac{2f}{3} + \frac{f^2}{5}\right), \qquad P_2 = A_{ij}\left(\frac{4f}{3} + \frac{4f^2}{7}\right), \qquad P_4 = A_{ij}\left(\frac{8f^2}{35}\right).$$

#### `ds_model="phenomenological"`

Each density-split leg carries an independent free anisotropy parameter:

$$F_{q_i}(k,\mu) = B_{q_i}(k) + \beta_{q_i} f\mu^2,$$

so

$$P_{q_i q_j}(k,\mu) = F_{q_i}(k,\mu)\,F_{q_j}(k,\mu)\,P_\mathrm{lin}(k)\,W_R(k)^2.$$

Expanding the angular dependence,

$$P_{q_i q_j}(k,\mu) = P_\mathrm{lin} W_R^2 \left[B_{q_i}B_{q_j} + f(B_{q_i}\beta_{q_j} + B_{q_j}\beta_{q_i})\mu^2 + \beta_{q_i}\beta_{q_j} f^2 \mu^4\right].$$

Setting $\beta_{q_i} = \beta_{q_j} = 0$ recovers the baseline model exactly.

### EFT modes for $P_{q_i q_j}$

#### `mode="tree"`

Tree-level only, with the EFT linear DS biases $b_{q1,i}$ and $b_{q1,j}$ replacing the tree-level $b_{q_i}$ parameters:

$$P_{q_i q_j}^\mathrm{tree}(k,\mu) = F_{q_i}^{(1)}(k,\mu)\,F_{q_j}^{(1)}(k,\mu)\,P_\mathrm{lin}(k)\,W_R(k)^2,$$

where each $F_{q}^{(1)}$ is built from $b_{q1}$ and the chosen DS angular model.

#### `mode="eft_ct"`

Two DS higher-derivative counterterms are added, one for each leg:

$$P_{ct,i}^{DS}(k,\mu) = b_{q,\nabla^2,i}\,(kR)^2\,\widetilde{P}_{q_i q_j}^{(i)}(k,\mu),$$

$$P_{ct,j}^{DS}(k,\mu) = b_{q,\nabla^2,j}\,(kR)^2\,\widetilde{P}_{q_i q_j}^{(j)}(k,\mu).$$

Here $\widetilde{P}_{q_i q_j}^{(i)}$ means the tree-level pair model evaluated with leg $i$ normalized to $b_{q1,i}=1$ while keeping the other leg physical; $\widetilde{P}_{q_i q_j}^{(j)}$ is defined analogously.

The full EFT-counterterm model is

$$P_{q_i q_j}^\mathrm{eft\_ct}(k,\mu) = P_{q_i q_j}^\mathrm{tree}(k,\mu) + P_{ct,i}^{DS}(k,\mu) + P_{ct,j}^{DS}(k,\mu).$$

#### `mode="eft"`

An isotropic DS-pair stochastic contribution is added:

$$P_{q_i q_j}^\mathrm{eft}(k,\mu) = P_{q_i q_j}^\mathrm{eft\_ct}(k,\mu) + s_{qq,0} + s_{qq,2} k^2.$$

Unlike the galaxy stochastic term used in $P_{gg}$, this correction is independent of $\mu$.

#### `mode="one_loop"`

The matter kernel is promoted from $P_\mathrm{lin}$ to the one-loop SPT matter spectrum while keeping the DS angular factors multiplicative:

$$P_{q_i q_j}^\mathrm{1\text{-}loop}(k,\mu) = F_{q_i}^{(1)}(k,\mu)\,F_{q_j}^{(1)}(k,\mu)\,\left[P_\mathrm{lin}(k) + P_{22}(k) + P_{13}(k)\right] W_R(k)^2.$$

The two DS higher-derivative counterterms and the isotropic stochastic term are then added on top.

> **Note:** the quadratic and tidal DS operators ($b_{q2}$ and $b_{qK^2}$) are not implemented for either leg in `mode="one_loop"`. The code raises `NotImplementedError` if any of these parameters are non-zero.

### Parameter table for $P_{q_i q_j}$

| Parameter | Description | Units | Active modes |
|---|---|---|---|
| $b_{q1,i}, b_{q1,j}$ | Linear DS biases on the two legs | — | all |
| $b_{q2,i}, b_{q2,j}$ | Quadratic DS biases | — | `one_loop` (not yet implemented) |
| $b_{qK^2,i}, b_{qK^2,j}$ | Tidal DS biases | — | `one_loop` (not yet implemented) |
| $b_{q,\nabla^2,i}, b_{q,\nabla^2,j}$ | DS higher-derivative counterterms on each leg | — | `eft_ct`, `eft`, `one_loop` |
| $\beta_{q_i}, \beta_{q_j}$ | Phenomenological DS anisotropy parameters | — | `phenomenological` model |
| $s_{qq,0}$ | Isotropic DS-pair stochastic amplitude | $(h^{-1}\mathrm{Mpc})^3$ | `eft`, `one_loop` |
| $s_{qq,2}$ | $k^2$ DS-pair stochastic amplitude | $(h^{-1}\mathrm{Mpc})^5$ | `eft`, `one_loop` |

---

## One-Loop SPT Integrals

### $P_{22}$ — two-field loop

$$P_{22}(k) = \frac{1}{4\pi^2} \int_0^\infty dq \int_{-1}^{1} d\mu_q\; 2q^3\, P_\mathrm{lin}(q)\, P_\mathrm{lin}(|\mathbf{k}-\mathbf{q}|)\, \left[F_2(\mathbf{q}, \mathbf{k}-\mathbf{q})\right]^2$$

using the SPT $F_2$ kernel:

$$F_2(\mathbf{k}_1, \mathbf{k}_2) = \frac{5}{7} + \frac{1}{2}\left(\frac{k_1}{k_2} + \frac{k_2}{k_1}\right)\cos\theta_{12} + \frac{2}{7}\cos^2\theta_{12}$$

Computed numerically via a 2D grid with $N_q = 128$ log-spaced points and $N_\mu = 128$ Gauss–Legendre points.

The velocity loop integrals use the $G_2^\mathrm{SPT}$ kernel:

$$G_2^\mathrm{SPT}(\mathbf{k}_1, \mathbf{k}_2) = \frac{3}{7} + \frac{1}{2}\left(\frac{k_1}{k_2} + \frac{k_2}{k_1}\right)\cos\theta_{12} + \frac{4}{7}\cos^2\theta_{12}$$

giving $P_{22}^{d\theta}$ (with $F_2 G_2^\mathrm{SPT}$) and $P_{22}^{\theta\theta}$ (with $G_2^{\mathrm{SPT}\,2}$).

### $P_{13}$ — propagator loop (EFT-renormalized)

$$2P_{13}(k) = \frac{6\, P_\mathrm{lin}(k)}{4\pi^2} \int_0^\infty dq\; q^3\, P_\mathrm{lin}(q)\, \mathcal{I}_{13}(k, q)$$

where $\mathcal{I}_{13}(k, q)$ is the analytic angular average of the symmetrized $F_3^{(s)}$ kernel, with the UV constant $-122/315$ subtracted so that only the scale-dependent finite correction remains. Three regimes handle the integral stably: a regular polynomial-plus-logarithm form, a Taylor expansion near $q = k$, and an asymptotic expansion for $q \gg k$.

The $P_{13}^{\theta\theta}$ term uses an analogous $G_3$ angular integral (UV constant $-1/5$ subtracted), and the identity $P_{13}^{d\theta} = (P_{13}^{dd} + P_{13}^{\theta\theta})/2$ is used.

### Bias loop integrals

All bias loops share the generic form

$$X(k) = \frac{1}{4\pi^2}\int_0^\infty dq \int_{-1}^{1} d\mu_q\; q^3\, P_\mathrm{lin}(q)\, P_\mathrm{lin}(|\mathbf{k}-\mathbf{q}|)\, \mathcal{K}(\mathbf{q}, \mathbf{k}-\mathbf{q})$$

with the following kernels, where $S_2(\cos\theta_{12}) \equiv \cos^2\theta_{12} - 1/3$:

| Integral | Kernel $\mathcal{K}$ | Bias combination |
|---|---|---|
| $I_{12}$ | $F_2$ | $2b_1 b_2$ |
| $J_{12}$ | $F_2 \cdot S_2$ | $2b_1 b_{s^2}$ |
| $I_{22}$ | $1/2$ | $b_2^2$ |
| $I_{2K}$ | $S_2/2$ | $2b_2 b_{s^2}$ |
| $J_{22}$ | $S_2^2/2$ | $b_{s^2}^2$ |
| $I_{12}^v$ | $G_2^\mathrm{SPT}$ | $2f b_2$ (RSD) |
| $J_{12}^v$ | $G_2^\mathrm{SPT} \cdot S_2$ | $2f b_{s^2}$ (RSD) |
| $I_{b_{3\mathrm{nl}}}$ | $F_2 \cdot (G_2^\mathrm{SPT} - F_2)$ | $4b_1 b_{3\mathrm{nl}}$ |

All bias loops are computed simultaneously in a single 2D pass over the $(q, \mu_q)$ grid.

---

## Analytic Marginalization

For linear parameters $\boldsymbol{\alpha}$ (e.g. stochastic terms, counterterms), the model takes the form

$$\mathbf{d} = \mathbf{m}(\boldsymbol{\theta}_\mathrm{nl}) + \mathbf{T}(\boldsymbol{\theta}_\mathrm{nl})\,\boldsymbol{\alpha}$$

where $\mathbf{m}$ is the nonlinear model piece and $\mathbf{T}$ is the template matrix. With a Gaussian prior $\boldsymbol{\alpha} \sim \mathcal{N}(\mathbf{0}, \mathbf{S})$, the marginalized log-likelihood is

$$-2\ln \mathcal{L}_\mathrm{marg} = \mathbf{r}^T \mathbf{C}^{-1} \mathbf{r} - \mathbf{v}^T \mathbf{A}^{-1} \mathbf{v} + \ln|\mathbf{A}| - \ln|\mathbf{S}^{-1}| + \mathrm{const}$$

where

$$\mathbf{r} = \mathbf{d} - \mathbf{m}, \qquad \mathbf{v} = \mathbf{T}^T \mathbf{C}^{-1} \mathbf{r}, \qquad \mathbf{A} = \mathbf{T}^T \mathbf{C}^{-1} \mathbf{T} + \mathbf{S}^{-1}.$$

The matrix $\mathbf{A}$ (of size $n_\mathrm{lin} \times n_\mathrm{lin}$) is solved via Cholesky decomposition. The MAP estimates of the linear parameters can be recovered as $\hat{\boldsymbol{\alpha}} = \mathbf{A}^{-1}\mathbf{v}$.

---

## Mode Summary

| Mode | Loop order | Counterterms | Stochastic | FoG | Bias params |
|---|---|---|---|---|---|
| `tree` | none | — | — | — | $b_1$ (and $b_{q1}$, $\beta_q$) |
| `eft_ct` | none | $c_0, c_2, c_4$; $b_{q,\nabla^2}$ | — | $P_{gg}$ only | $b_1$ |
| `eft` | none | same as above | $s_0$, $s_2$ or $s_{qq,0}$, $s_{qq,2}$ | $P_{gg}$ only | $b_1$ |
| `one_loop` | 1-loop SPT | same as above | $s_0$, $s_2$ or $s_{qq,0}$, $s_{qq,2}$ | $P_{gg}$ and $P_{qg}$ | $b_1, b_2, b_{s^2}, b_{3\mathrm{nl}}$ |

---

## References

- Kaiser, N. (1987), "Clustering in real space and in redshift space", *MNRAS* 227, 1. DOI: [10.1093/mnras/227.1.1](https://doi.org/10.1093/mnras/227.1.1)
- Ivanov, M., Simonovic, M., and Zaldarriaga, M. (2020), "Cosmological parameters from the BOSS galaxy power spectrum", for the EFT and one-loop redshift-space large-scale-structure modeling context used here. arXiv: [2003.00577](https://arxiv.org/abs/2003.00577)
- Paillas, E. et al. (2023), "Constraining $\nu\Lambda$CDM with density-split clustering", for the interpretation of density-split quantiles as effective large-scale tracers of the smoothed density field. DOI: [10.1093/mnras/stad1349](https://doi.org/10.1093/mnras/stad1349)
