"""Analytic template emulator for galaxy auto-power P_gg multipoles.

For fixed cosmology, P_gg(k, mu) is polynomial in mu (degree ≤ 6), so the
Legendre multipole projection is exact and analytic.  Each MCMC step reduces
to O(n_templates × nk) floating-point operations rather than a 2-D
Gauss-Legendre quadrature over (nk × nmu) points.
"""

import numpy as np

from .cosmology import get_linear_power, get_growth_rate

_VALID_MODES = ("tree_only", "eft_lite", "eft_full", "one_loop", "one_loop_matter_only")
_EMULATOR_UNSUPPORTED_MODES = ()

# Analytic Legendre moments (same as emulator.py)
#   _M[ell][n] = (2*ell+1)/2 * int_{-1}^{1} L_ell(mu) * mu^n dmu
_M = {
    0: {0: 1.0,  2: 1.0 / 3.0,  4: 1.0 / 5.0,  6: 1.0 / 7.0},
    2: {0: 0.0,  2: 2.0 / 3.0,  4: 4.0 / 7.0,  6: 10.0 / 21.0},
    4: {0: 0.0,  2: 0.0,        4: 8.0 / 35.0, 6: 24.0 / 77.0},
}


class GalaxyTemplateEmulator:
    """Analytic template emulator for galaxy auto-power P_gg multipoles.

    Precomputes k-space templates at construction time (once per cosmology).
    ``predict`` then evaluates theory multipoles via pure linear algebra.

    Parameters
    ----------
    cosmo : cosmoprimo.Cosmology
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    ells : tuple of int
        Multipole orders to compute (default ``(0, 2, 4)``).
    z : float
        Redshift (default ``0.5``).
    space : str
        ``'redshift'`` (default) or ``'real'``.
    mode : str
        EFT mode: ``'tree_only'``, ``'eft_lite'`` (default), or ``'eft_full'``.
    """

    def __init__(
        self,
        cosmo,
        k,
        ells=(0, 2, 4),
        z=0.5,
        space="redshift",
        mode="eft_lite",
    ):
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
        self.space = space
        self.mode = mode

        self._precompute(cosmo)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _precompute(self, cosmo):
        """Compute cosmology-dependent templates (called once at init)."""
        k = self.k
        plin = get_linear_power(cosmo, k, self.z)
        self.f = get_growth_rate(cosmo, self.z) if self.space == "redshift" else 0.0

        self._T_Plin = plin             # P_lin
        self._T_k2Plin = k ** 2 * plin  # k^2 * P_lin
        self._T_k2 = k ** 2             # k^2 (stochastic shape)

        if self.mode in ("one_loop", "one_loop_matter_only"):
            from .galaxy_models import _compute_loop_templates
            def plin_func(kk):
                return get_linear_power(cosmo, np.asarray(kk, dtype=float), self.z)
            loops = _compute_loop_templates(k, plin_func)
            self._T_p22 = loops["p22"]
            self._T_p13 = loops["p13"]
            self._T_I12 = loops["I12"]
            self._T_J12 = loops["J12"]
            self._T_I22 = loops["I22"]
            self._T_I2K = loops["I2K"]
            self._T_J22 = loops["J22"]
            self._T_p22_dt = loops["p22_dt"]
            self._T_p22_tt = loops["p22_tt"]
            self._T_p13_dt = loops["p13_dt"]
            self._T_p13_tt = loops["p13_tt"]

    # ------------------------------------------------------------------
    # Cosmology update
    # ------------------------------------------------------------------

    def update_cosmology(self, plin: np.ndarray, f: float, loop_arrays=None) -> None:
        """Recompute templates for a new (plin, f) without reinitialising.

        ~0.01 ms per call. Intended for use in cosmology-varying MCMC loops.

        Parameters
        ----------
        plin : np.ndarray, shape (nk,)
        f : float
        loop_arrays : dict, optional
            Required when mode='one_loop'. Keys: 'p22', 'p13', 'I12', 'J12',
            'I22', 'I2K', 'J22', 'p22_dt', 'p22_tt', 'p13_dt', 'p13_tt',
            each shape (nk,).
        """
        plin = np.asarray(plin, dtype=float)
        self._T_Plin = plin
        self._T_k2Plin = self.k ** 2 * plin
        self.f = float(f)
        if self.mode in ("one_loop", "one_loop_matter_only") and loop_arrays is not None:
            self._T_p22 = loop_arrays["p22"]
            self._T_p13 = loop_arrays["p13"]
            self._T_I12 = loop_arrays["I12"]
            self._T_J12 = loop_arrays["J12"]
            self._T_I22 = loop_arrays["I22"]
            self._T_I2K = loop_arrays["I2K"]
            self._T_J22 = loop_arrays["J22"]
            self._T_p22_dt = loop_arrays["p22_dt"]
            self._T_p22_tt = loop_arrays["p22_tt"]
            self._T_p13_dt = loop_arrays["p13_dt"]
            self._T_p13_tt = loop_arrays["p13_tt"]

    # ------------------------------------------------------------------
    # Core analytic projection
    # ------------------------------------------------------------------

    def _pole(self, b1, c0, c2, c4, s0, s2, ell, b2=0.0, bs2=0.0):
        """Analytic Legendre multipole P_ell(k) for the galaxy auto-spectrum.

        Parameters
        ----------
        b1 : float
            Linear galaxy bias.
        c0, c2, c4 : float
            Galaxy EFT counterterm coefficients (eft_lite/eft_full; default 0).
        s0, s2 : float
            Stochastic amplitude parameters (eft_full; default 0).
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
        A = self._T_Plin
        k2A = self._T_k2Plin
        k2 = self._T_k2

        # ---- Tree-level: (b1 + f*mu^2)^2 * P_lin ----
        # = (b1^2 + 2*b1*f*mu^2 + f^2*mu^4) * P_lin
        cm0 = b1 ** 2 * A
        cm2 = 2.0 * b1 * f * A
        cm4 = f ** 2 * A
        cm6 = np.zeros_like(A)

        # ---- EFT counterterm ----
        # -2*k^2*(c0+c2*mu^2+c4*mu^4)*(b1+f*mu^2)*P_lin
        # Expanding:
        #   mu^0: -2*b1*c0
        #   mu^2: -2*(b1*c2 + f*c0)
        #   mu^4: -2*(b1*c4 + f*c2)
        #   mu^6: -2*f*c4
        if self.mode != "tree_only":
            cm0 = cm0 - 2.0 * b1 * c0 * k2A
            cm2 = cm2 - 2.0 * (b1 * c2 + f * c0) * k2A
            cm4 = cm4 - 2.0 * (b1 * c4 + f * c2) * k2A
            cm6 = cm6 - 2.0 * f * c4 * k2A

        # ---- One-loop matter-only corrections ----
        if self.mode == "one_loop_matter_only":
            p_loop_dd = b1 ** 2 * (self._T_p22 + self._T_p13)
            cm0 = cm0 + p_loop_dd
            P_dt_loop = self._T_p22_dt + self._T_p13_dt
            P_tt_loop = self._T_p22_tt + self._T_p13_tt
            cm2 = cm2 + 2.0 * b1 * f * P_dt_loop
            cm4 = cm4 + f ** 2 * P_tt_loop

        # ---- One-loop corrections (one_loop mode) ----
        if self.mode == "one_loop":
            p_loop_dd = b1 ** 2 * (self._T_p22 + self._T_p13)
            p_loop_bias = (
                2.0 * b1 * b2   * self._T_I12
                + 2.0 * b1 * bs2  * self._T_J12
                + b2 ** 2          * self._T_I22
                + 2.0 * b2 * bs2  * self._T_I2K
                + bs2 ** 2         * self._T_J22
            )
            cm0 = cm0 + p_loop_dd + p_loop_bias
            # Velocity loop contributions: density×velocity (mu^2) and velocity auto (mu^4)
            P_dt_loop = self._T_p22_dt + self._T_p13_dt
            P_tt_loop = self._T_p22_tt + self._T_p13_tt
            cm2 = cm2 + 2.0 * b1 * f * P_dt_loop
            cm4 = cm4 + f ** 2 * P_tt_loop

        # ---- Stochastic terms (eft_full and one_loop) ----
        if self.mode in ("eft_full", "one_loop", "one_loop_matter_only"):
            cm0 = cm0 + s0 + s2 * k2

        return M0 * cm0 + M2 * cm2 + M4 * cm4 + M6 * cm6

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, params: dict) -> np.ndarray:
        """Evaluate theory multipoles for all requested ells.

        Parameters
        ----------
        params : dict
            ``'b1'`` : float
                Linear galaxy bias.
            ``'c0'``, ``'c2'``, ``'c4'`` : float, optional
                EFT counterterm coefficients (default 0).
            ``'s0'``, ``'s2'`` : float, optional
                Stochastic amplitude parameters (eft_full; default 0).

        Returns
        -------
        np.ndarray, shape (n_ells * nk,)
            Flat data vector ordered as ``[ell0_k0..kN, ell2_k0..kN, ...]``.
        """
        b1  = float(params["b1"])
        c0  = float(params.get("c0", 0.0))
        c2  = float(params.get("c2", 0.0))
        c4  = float(params.get("c4", 0.0))
        s0  = float(params.get("s0", 0.0))
        s2  = float(params.get("s2", 0.0))
        b2  = float(params.get("b2", 0.0))
        bs2 = float(params.get("bs2", 0.0))

        pieces = []
        for ell in self.ells:
            pieces.append(self._pole(b1, c0, c2, c4, s0, s2, ell, b2=b2, bs2=bs2))
        return np.concatenate(pieces)
