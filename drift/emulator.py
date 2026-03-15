"""Analytic template emulator for density-split × galaxy EFT multipoles.

For fixed cosmology, P(k, mu) is polynomial in mu (degree ≤ 4), so the
Legendre multipole projection is exact and analytic.  Each MCMC step reduces
to O(n_templates × nk) floating-point operations rather than a 2-D
Gauss-Legendre quadrature over (nk × nmu) points.
"""

import numpy as np

from .cosmology import get_linear_power, get_growth_rate
from .kernels import gaussian_kernel, tophat_kernel

_VALID_MODES = ("tree_only", "eft_lite", "eft_full")
_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")

# Analytic Legendre moments
#   _M[ell][n] = (2*ell+1)/2 * int_{-1}^{1} L_ell(mu) * mu^n dmu
# Non-zero entries for even n = 0, 2, 4 and ell = 0, 2, 4:
_M = {
    0: {0: 1.0,       2: 1.0 / 3.0,  4: 1.0 / 5.0},
    2: {0: 0.0,       2: 2.0 / 3.0,  4: 4.0 / 7.0},
    4: {0: 0.0,       2: 0.0,        4: 8.0 / 35.0},
}


class TemplateEmulator:
    """Analytic template emulator for DS × galaxy EFT multipoles.

    Precomputes k-space templates at construction time (once per cosmology).
    ``predict`` then evaluates theory multipoles via pure linear algebra.

    Parameters
    ----------
    cosmo : cosmoprimo.Cosmology
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    ells : tuple of int
        Multipole orders to compute (default ``(0, 2)``).
    z : float
        Redshift (default ``0.5``).
    R : float
        Smoothing radius in Mpc/h (default ``10.0``).
    kernel : str
        Smoothing kernel: ``'gaussian'`` (default) or ``'tophat'``.
    space : str
        ``'redshift'`` (default) or ``'real'``.
    ds_model : str
        DS RSD model: ``'baseline'``, ``'rsd_selection'``, or
        ``'phenomenological'`` (default ``'baseline'``).
    mode : str
        EFT mode: ``'tree_only'``, ``'eft_lite'`` (default), or
        ``'eft_full'``.
    """

    def __init__(
        self,
        cosmo,
        k,
        ells=(0, 2),
        z=0.5,
        R=10.0,
        kernel="gaussian",
        space="redshift",
        ds_model="baseline",
        mode="eft_lite",
    ):
        if ds_model not in _VALID_DS_MODELS:
            raise ValueError(
                f"Unknown ds_model '{ds_model}'. Choose one of {_VALID_DS_MODELS}."
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose one of {_VALID_MODES}.")
        for ell in ells:
            if ell not in _M:
                raise ValueError(
                    f"Unsupported ell={ell}. Supported: {list(_M)}."
                )

        self.k = np.asarray(k, dtype=float)
        self.ells = tuple(ells)
        self.z = float(z)
        self.R = float(R)
        self.kernel = kernel
        self.space = space
        self.ds_model = ds_model
        self.mode = mode

        self._precompute(cosmo)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _precompute(self, cosmo):
        """Compute cosmology-dependent templates (called once at init)."""
        k = self.k
        plin = get_linear_power(cosmo, k, self.z)
        if self.kernel == "gaussian":
            wk = gaussian_kernel(k, self.R)
        elif self.kernel == "tophat":
            wk = tophat_kernel(k, self.R)
        else:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. Choose 'gaussian' or 'tophat'."
            )
        self.f = get_growth_rate(cosmo, self.z) if self.space == "redshift" else 0.0

        A = plin * wk
        self._T_A = A                         # Plin * W_R
        self._T_k2A = k ** 2 * A             # k^2 * Plin * W_R
        self._T_k2R2A = (k * self.R) ** 2 * A  # (kR)^2 * Plin * W_R
        self._T_k2 = k ** 2                  # k^2 (stochastic shape)

    # ------------------------------------------------------------------
    # Core analytic projection
    # ------------------------------------------------------------------

    def _pole(self, b1, bq1, beta_q, bq_nabla2, c0, c2, c4, s0, s2, ell):
        """Analytic Legendre multipole P_ell(k) for one quantile.

        Parameters
        ----------
        b1, bq1, beta_q, bq_nabla2 : float
            Bias parameters (bq_nabla2 and beta_q default to 0).
        c0, c2, c4, s0, s2 : float
            Galaxy EFT and stochastic parameters (all default to 0).
        ell : int

        Returns
        -------
        np.ndarray, shape (nk,)
        """
        M0 = _M[ell][0]
        M2 = _M[ell][2]
        M4 = _M[ell][4]
        f = self.f
        A = self._T_A
        k2A = self._T_k2A
        k2R2A = self._T_k2R2A
        k2 = self._T_k2

        # ---- Tree-level mu-polynomial coefficients ----
        if self.ds_model == "baseline":
            # P_tree = bq1 * A * (b1 + f*mu^2)
            cm0 = bq1 * b1 * A
            cm2 = bq1 * f * A
            cm4 = np.zeros_like(A)

        elif self.ds_model == "rsd_selection":
            # P_tree = bq1 * A * (1 + f*mu^2) * (b1 + f*mu^2)
            #        = bq1 * A * (b1 + f*(1+b1)*mu^2 + f^2*mu^4)
            cm0 = bq1 * b1 * A
            cm2 = bq1 * f * (1.0 + b1) * A
            cm4 = bq1 * f ** 2 * A

        else:  # phenomenological
            # P_tree = A * (bq1 + beta_q*f*mu^2) * (b1 + f*mu^2)
            #        = A * (bq1*b1 + f*(bq1 + beta_q*b1)*mu^2 + beta_q*f^2*mu^4)
            cm0 = bq1 * b1 * A
            cm2 = f * (bq1 + beta_q * b1) * A
            cm4 = beta_q * f ** 2 * A

        # ---- EFT counterterms ----
        if self.mode != "tree_only":
            # Galaxy counterterm: -bq1 * k^2*A * (c0 + c2*mu^2 + c4*mu^4)
            # (ds_amplitude = bq1 * W_R, so factor = bq1 * k^2 * A)
            cm0 = cm0 - c0 * bq1 * k2A
            cm2 = cm2 - c2 * bq1 * k2A
            cm4 = cm4 - c4 * bq1 * k2A

            # DS higher-derivative counterterm: bq_nabla2 * (kR)^2 * tree_normed
            # tree_normed uses bq1=1 and beta_q=0 (ds_normed defaults).
            # For baseline/phenomenological: tree_normed = A*(b1 + f*mu^2)
            # For rsd_selection:             tree_normed = A*(b1 + f*(1+b1)*mu^2 + f^2*mu^4)
            if self.ds_model == "rsd_selection":
                cm0 = cm0 + bq_nabla2 * b1 * k2R2A
                cm2 = cm2 + bq_nabla2 * f * (1.0 + b1) * k2R2A
                cm4 = cm4 + bq_nabla2 * f ** 2 * k2R2A
            else:  # baseline and phenomenological share tree_normed shape
                cm0 = cm0 + bq_nabla2 * b1 * k2R2A
                cm2 = cm2 + bq_nabla2 * f * k2R2A
                # cm4 unchanged (tree_normed has no mu^4 term for these models)

        # ---- Stochastic terms (eft_full only, mu-independent) ----
        if self.mode == "eft_full":
            cm0 = cm0 + s0 + s2 * k2

        return M0 * cm0 + M2 * cm2 + M4 * cm4

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, params: dict) -> np.ndarray:
        """Evaluate theory multipoles for all quantiles and ells.

        Parameters
        ----------
        params : dict
            ``'b1'`` : float
                Linear galaxy bias.
            ``'bq1'`` : array_like, shape (n_quantiles,)
                Per-quantile DS linear bias.
            ``'beta_q'`` : array_like, shape (n_quantiles,), optional
                Per-quantile DS anisotropy parameter (phenomenological model;
                default 0).
            ``'bq_nabla2'`` : array_like, shape (n_quantiles,), optional
                Per-quantile DS higher-derivative coefficient (default 0).
            ``'c0'``, ``'c2'``, ``'c4'`` : float, optional
                Isotropic / mu^2 / mu^4 galaxy EFT counterterm coefficients
                (default 0).
            ``'s0'``, ``'s2'`` : float, optional
                Stochastic amplitude parameters (eft_full; default 0).

        Returns
        -------
        np.ndarray, shape (n_quantiles * n_ells * nk,)
            Flat data vector ordered as
            ``[q0_ell0, q0_ell2, ..., q1_ell0, q1_ell2, ...]``.
        """
        b1 = float(params["b1"])
        bq1_arr = np.asarray(params["bq1"], dtype=float)
        n_q = len(bq1_arr)

        c0 = float(params.get("c0", 0.0))
        c2 = float(params.get("c2", 0.0))
        c4 = float(params.get("c4", 0.0))
        s0 = float(params.get("s0", 0.0))
        s2 = float(params.get("s2", 0.0))

        beta_q_arr = np.asarray(
            params.get("beta_q", [0.0] * n_q), dtype=float
        )
        bq_nabla2_arr = np.asarray(
            params.get("bq_nabla2", [0.0] * n_q), dtype=float
        )

        pieces = []
        for bq1, beta_q, bq_nabla2 in zip(bq1_arr, beta_q_arr, bq_nabla2_arr):
            for ell in self.ells:
                pieces.append(
                    self._pole(
                        b1, float(bq1), float(beta_q), float(bq_nabla2),
                        c0, c2, c4, s0, s2, ell,
                    )
                )
        return np.concatenate(pieces)
