"""Analytic template emulator for density-split × galaxy EFT multipoles.

For fixed cosmology, P(k, mu) is polynomial in mu (degree ≤ 4), so the
Legendre multipole projection is exact and analytic.  Each MCMC step reduces
to O(n_templates × nk) floating-point operations rather than a 2-D
Gauss-Legendre quadrature over (nk × nmu) points.
"""

import numpy as np

from .cosmology import get_linear_power, get_growth_rate
from .kernels import gaussian_kernel, tophat_kernel

_VALID_MODES = ("tree", "eft_ct", "eft", "one_loop")
_VALID_DS_MODELS = ("baseline", "rsd_selection", "phenomenological")

# Analytic Legendre moments
#   _M[ell][n] = (2*ell+1)/2 * int_{-1}^{1} L_ell(mu) * mu^n dmu
# Non-zero entries for even n = 0, 2, 4, 6 and ell = 0, 2, 4:
_M = {
    0: {0: 1.0,  2: 1.0 / 3.0,  4: 1.0 / 5.0,  6: 1.0 / 7.0},
    2: {0: 0.0,  2: 2.0 / 3.0,  4: 4.0 / 7.0,  6: 10.0 / 21.0},
    4: {0: 0.0,  2: 0.0,        4: 8.0 / 35.0, 6: 24.0 / 77.0},
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
        EFT mode: ``'tree'``, ``'eft_ct'`` (default), or
        ``'eft'``.
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
        mode="eft_ct",
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
        self._wk = wk

        A = plin * wk
        self._T_A = A                           # Plin * W_R
        self._T_k2A = k ** 2 * A               # k^2 * Plin * W_R
        self._T_k2R2A = (k * self.R) ** 2 * A  # (kR)^2 * Plin * W_R
        self._T_k2 = k ** 2                    # k^2 (stochastic shape)

        if self.mode in ("one_loop",):
            from .galaxy_models import _compute_loop_templates
            def _plin_func(kk):
                from .cosmology import get_linear_power as _glp
                return _glp(cosmo, np.asarray(kk, dtype=float), self.z)
            loops = _compute_loop_templates(k, _plin_func)
            self._set_loop_templates(loops)

    # ------------------------------------------------------------------
    # Cosmology update
    # ------------------------------------------------------------------

    def _set_loop_templates(self, loops: dict) -> None:
        """Store W_R-multiplied loop templates from a loops dict."""
        wk = self._wk
        self._T_p22    = loops["p22"]    * wk
        self._T_p13    = loops["p13"]    * wk
        self._T_I12    = loops["I12"]    * wk
        self._T_J12    = loops["J12"]    * wk
        self._T_Ib3nl  = loops["Ib3nl"] * wk
        self._T_I12_v  = loops["I12_v"] * wk
        self._T_J12_v  = loops["J12_v"] * wk
        self._T_p22_dt = loops["p22_dt"] * wk
        self._T_p13_dt = loops["p13_dt"] * wk

    def update_cosmology(self, plin: np.ndarray, f: float,
                         loop_arrays: dict = None) -> None:
        """Recompute templates for a new (plin, f) without reinitialising.

        ~0.01 ms per call. Intended for use in cosmology-varying MCMC loops.

        Parameters
        ----------
        plin : np.ndarray, shape (nk,)
        f : float
        loop_arrays : dict, optional
            Required when mode is 'one_loop'.
        """
        A = np.asarray(plin, dtype=float) * self._wk
        self._T_A     = A
        self._T_k2A   = self.k ** 2 * A
        self._T_k2R2A = (self.k * self.R) ** 2 * A
        self.f        = float(f)
        if self.mode in ("one_loop",) and loop_arrays is not None:
            self._set_loop_templates(loop_arrays)

    # ------------------------------------------------------------------
    # Core analytic projection
    # ------------------------------------------------------------------

    def _pole(self, b1, bq1, beta_q, bq_nabla2, c0, c2, c4, s0, s2, ell,
              b2=0.0, bs2=0.0, b3nl=0.0, sigma_fog=0.0):
        """Analytic Legendre multipole P_ell(k) for one quantile.

        Parameters
        ----------
        b1, bq1, beta_q, bq_nabla2 : float
            Bias parameters (bq_nabla2 and beta_q default to 0).
        c0, c2, c4, s0, s2 : float
            Galaxy EFT and stochastic parameters (all default to 0).
        b2, bs2, b3nl : float
            Higher-order bias (one_loop mode; default 0).
        sigma_fog : float
            FoG coefficient [(Mpc/h)^2] (default 0).
        ell : int

        Returns
        -------
        np.ndarray, shape (nk,)
        """
        M0 = _M[ell][0]
        M2 = _M[ell][2]
        M4 = _M[ell][4]
        M6 = _M[ell][6]
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

        cm6 = np.zeros_like(A)

        # ---- EFT counterterms ----
        if self.mode != "tree":
            # Galaxy counterterm: -k^2 * P_{DS×lin}(k,mu) * (c0 + c2*mu^2 + c4*mu^4)
            # P_{DS×lin} carries the DS angular structure (no galaxy bias factor):
            #   baseline:        bq1 * A
            #   rsd_selection:   bq1 * A * (1 + f*mu^2)
            #   phenomenological: A * (bq1 + beta_q*f*mu^2)
            # Expanding the product gives the mu-polynomial coefficients below.
            if self.ds_model == "baseline":
                cm0 = cm0 - c0 * bq1 * k2A
                cm2 = cm2 - c2 * bq1 * k2A
                cm4 = cm4 - c4 * bq1 * k2A
            elif self.ds_model == "rsd_selection":
                # P_{DS×lin} * CT_shape = bq1*A*(1+f*mu^2)*(c0+c2*mu^2+c4*mu^4)
                cm0 = cm0 - c0 * bq1 * k2A
                cm2 = cm2 - (c2 + c0 * f) * bq1 * k2A
                cm4 = cm4 - (c4 + c2 * f) * bq1 * k2A
                cm6 = cm6 - c4 * f * bq1 * k2A
            else:  # phenomenological
                # P_{DS×lin} * CT_shape = A*(bq1+beta_q*f*mu^2)*(c0+c2*mu^2+c4*mu^4)
                cm0 = cm0 - bq1 * c0 * k2A
                cm2 = cm2 - (bq1 * c2 + beta_q * f * c0) * k2A
                cm4 = cm4 - (bq1 * c4 + beta_q * f * c2) * k2A
                cm6 = cm6 - beta_q * f * c4 * k2A

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

        # ---- One-loop contributions ----
        if self.mode in ("one_loop",):
            P_dd_loop = self._T_p22 + self._T_p13   # W_R already multiplied in
            P_dt_loop = self._T_p22_dt + self._T_p13_dt

            # Galaxy-matter cross-spectrum loop: mu^0 and mu^2
            Pgm_loop_l0 = b1 * P_dd_loop
            Pgm_loop_l2 = f * P_dt_loop

            if self.mode == "one_loop":
                Pgm_loop_l0 = (Pgm_loop_l0
                               + b2 * self._T_I12 + bs2 * self._T_J12
                               + 2.0 * b3nl * self._T_Ib3nl)
                Pgm_loop_l2 = Pgm_loop_l2 + f * (b2 * self._T_I12_v + bs2 * self._T_J12_v)

            # Multiply by DS_factor = (bq1 + beta_q*f*mu^2)
            # cm0 += bq1 * Pgm_loop_l0
            # cm2 += bq1 * Pgm_loop_l2 + beta_q*f * Pgm_loop_l0
            # cm4 += beta_q*f * Pgm_loop_l2
            if self.ds_model == "baseline":
                cm0 = cm0 + bq1 * Pgm_loop_l0
                cm2 = cm2 + bq1 * Pgm_loop_l2
            elif self.ds_model == "rsd_selection":
                cm0 = cm0 + bq1 * Pgm_loop_l0
                cm2 = cm2 + bq1 * Pgm_loop_l2 + bq1 * f * Pgm_loop_l0
                cm4 = cm4 + bq1 * f * Pgm_loop_l2
            else:  # phenomenological
                cm0 = cm0 + bq1 * Pgm_loop_l0
                cm2 = cm2 + bq1 * Pgm_loop_l2 + beta_q * f * Pgm_loop_l0
                cm4 = cm4 + beta_q * f * Pgm_loop_l2

        # ---- FoG for cross-spectrum (one_loop/eft_full modes with sigma_fog) ----
        if sigma_fog != 0.0 and self.mode != "tree":
            # -sigma_fog * k^2 * DS_factor(mu) * (b1*mu^2 + f*mu^4) * Plin*W_R
            # Expanding by DS_factor(mu) = bq1 + beta_q*f*mu^2:
            #   mu^2: bq1*b1
            #   mu^4: bq1*f + beta_q*f*b1
            #   mu^6: beta_q*f^2
            if self.ds_model == "baseline":
                cm2 = cm2 - sigma_fog * bq1 * b1 * k2A
                cm4 = cm4 - sigma_fog * bq1 * f  * k2A
            elif self.ds_model == "rsd_selection":
                # DS_factor = bq1*(1+f*mu^2)
                cm2 = cm2 - sigma_fog * bq1 * b1 * k2A
                cm4 = cm4 - sigma_fog * bq1 * (f + f * b1) * k2A
                cm6 = cm6 - sigma_fog * bq1 * f ** 2 * k2A
            else:  # phenomenological
                cm2 = cm2 - sigma_fog * bq1 * b1 * k2A
                cm4 = cm4 - sigma_fog * (bq1 * f + beta_q * f * b1) * k2A
                cm6 = cm6 - sigma_fog * beta_q * f ** 2 * k2A

        # ---- Stochastic terms ----
        if self.mode == "eft":
            cm0 = cm0 + s0 + s2 * k2

        if self.mode == "one_loop":
            cm0 = cm0 + s0
            cm2 = cm2 + s2 * k2

        return M0 * cm0 + M2 * cm2 + M4 * cm4 + M6 * cm6

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
        b1        = float(params["b1"])
        bq1_arr   = np.asarray(params["bq1"], dtype=float)
        n_q       = len(bq1_arr)

        c0        = float(params.get("c0", 0.0))
        c2        = float(params.get("c2", 0.0))
        c4        = float(params.get("c4", 0.0))
        s0        = float(params.get("s0", 0.0))
        s2        = float(params.get("s2", 0.0))
        b2        = float(params.get("b2", 0.0))
        bs2       = float(params.get("bs2", 0.0))
        b3nl      = float(params.get("b3nl", 0.0))
        sigma_fog = float(params.get("sigma_fog", 0.0))

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
                        b2=b2, bs2=bs2, b3nl=b3nl, sigma_fog=sigma_fog,
                    )
                )
        return np.concatenate(pieces)

    @property
    def linear_param_names(self):
        """List of linear parameter names for the current mode (shared across quantiles)."""
        if self.mode == "tree":
            return []
        elif self.mode == "eft_ct":
            return ["c0"]
        elif self.mode == "eft":
            return ["c0", "s0"]
        elif self.mode == "one_loop":
            return ["c0", "c2", "c4", "s0", "s2", "b3nl"]
        return []

    def _template_pole(self, b1, ell, bq1, beta_q, b2=0.0, bs2=0.0):
        """Return template vectors for each linear parameter at one quantile/multipole.

        Returns dict mapping linear param name -> np.ndarray shape (nk,).
        """
        M0 = _M[ell][0]
        M2 = _M[ell][2]
        M4 = _M[ell][4]
        M6 = _M[ell][6]
        f   = self.f
        k2A = self._T_k2A
        k2  = self._T_k2
        nk  = len(self.k)

        templates = {}
        if self.mode == "tree":
            return templates

        # c0: -k^2 * P_{DS×lin} * c0
        # P_{DS×lin} mu-polynomial:
        #   baseline:          bq1 * A  -> mu^0: bq1, mu^2: 0
        #   rsd_selection:     bq1*A*(1+f*mu^2) -> mu^0: bq1, mu^2: bq1*f
        #   phenomenological:  A*(bq1+beta_q*f*mu^2) -> mu^0: bq1, mu^2: beta_q*f
        if self.ds_model == "baseline":
            templates["c0"] = M0 * (-bq1 * k2A)
        elif self.ds_model == "rsd_selection":
            templates["c0"] = M0 * (-bq1 * k2A) + M2 * (-bq1 * f * k2A)
        else:  # phenomenological
            templates["c0"] = M0 * (-bq1 * k2A) + M2 * (-beta_q * f * k2A)

        if self.mode in ("one_loop",):
            if self.ds_model == "baseline":
                templates["c2"] = M2 * (-bq1 * k2A)
                templates["c4"] = M4 * (-bq1 * k2A)
            elif self.ds_model == "rsd_selection":
                templates["c2"] = M2 * (-bq1 * k2A) + M4 * (-bq1 * f * k2A)
                templates["c4"] = M4 * (-bq1 * k2A) + M6 * (-bq1 * f * k2A)
            else:  # phenomenological
                templates["c2"] = M2 * (-bq1 * k2A) + M4 * (-beta_q * f * k2A)
                templates["c4"] = M4 * (-bq1 * k2A) + M6 * (-beta_q * f * k2A)

        if self.mode in ("eft", "one_loop"):
            templates["s0"] = M0 * np.ones(nk)
            templates["s2"] = M2 * k2

        if self.mode == "one_loop":
            # b3nl: 2*b3nl*Ib3nl * W_R enters at Pgm_l0, then multiplied by DS_factor mu^0
            # Contribution: bq1 * 2 * Ib3nl * W_R at mu^0
            templates["b3nl"] = M0 * (2.0 * bq1 * self._T_Ib3nl)

        # per-quantile bq_nabla2 template: (kR)^2 * tree_normed(bq1=1, beta_q=0)
        # tree_normed for baseline/phenomenological: A*(b1+f*mu^2), bq1=1 in the expression
        # This is stored separately since it's per-quantile
        if self.mode != "tree":
            A = self._T_A
            k2R2A = self._T_k2R2A
            if self.ds_model == "rsd_selection":
                templates["bq_nabla2"] = (M0 * b1 + M2 * f * (1.0 + b1) + M4 * f**2) * k2R2A
            else:  # baseline and phenomenological share tree_normed shape
                templates["bq_nabla2"] = (M0 * b1 + M2 * f) * k2R2A

        return templates

    def predict_decomposed(self, params: dict):
        """Decompose theory into nonlinear piece m and template matrix T.

        Returns
        -------
        m : np.ndarray, shape (n_q * n_ells * nk,)
            Model with all linear params set to 0.
        T : np.ndarray, shape (n_q * n_ells * nk, n_shared + n_q)
            Template matrix. First n_shared columns are shared linear params
            (c0, c2, ..., b3nl); last n_q columns are per-quantile bq_nabla2.
        """
        b1        = float(params["b1"])
        bq1_arr   = np.asarray(params["bq1"], dtype=float)
        n_q       = len(bq1_arr)
        b2        = float(params.get("b2", 0.0))
        bs2       = float(params.get("bs2", 0.0))
        sigma_fog = float(params.get("sigma_fog", 0.0))

        beta_q_arr = np.asarray(
            params.get("beta_q", [0.0] * n_q), dtype=float
        )

        shared_lin = self.linear_param_names  # e.g. ["c0","c2","c4","s0","s2","b3nl"]
        n_shared   = len(shared_lin)
        nk         = len(self.k)
        n_ell      = len(self.ells)
        n_tot      = n_q * n_ell * nk
        n_lin      = n_shared + n_q  # per-quantile bq_nabla2

        # m: model with all linear params = 0
        zero_params = dict(params)
        for name in shared_lin:
            zero_params[name] = 0.0
        zero_params["bq_nabla2"] = [0.0] * n_q
        m = self.predict(zero_params)

        # T: template matrix
        T = np.zeros((n_tot, n_lin))
        for i_q, (bq1, beta_q) in enumerate(zip(bq1_arr, beta_q_arr)):
            for i_ell, ell in enumerate(self.ells):
                row_start = (i_q * n_ell + i_ell) * nk
                row_end   = row_start + nk
                tpl = self._template_pole(b1, ell, float(bq1), float(beta_q),
                                          b2=b2, bs2=bs2)
                for j, name in enumerate(shared_lin):
                    if name in tpl:
                        T[row_start:row_end, j] = tpl[name]
                # per-quantile bq_nabla2 column
                if "bq_nabla2" in tpl:
                    col = n_shared + i_q
                    T[row_start:row_end, col] = tpl["bq_nabla2"]

        return m, T
