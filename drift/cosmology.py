"""Cosmology utilities: wraps cosmoprimo to provide P_lin(k,z) and f(z)."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_DEFAULT_PARAMS = {
    "h": 0.6766,
    "Omega_m": 0.3111,
    "Omega_b": 0.049,
    "sigma8": 0.8102,
    "n_s": 0.9665,
    "engine": "class",
}


def get_cosmology(params: dict = None):
    """Return a cosmoprimo Cosmology object.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters. Keys: h, Omega_m, Omega_b, sigma8, n_s, engine.
        Defaults to Planck 2018-like values.

    Returns
    -------
    cosmoprimo.Cosmology
    """
    from cosmoprimo import Cosmology

    p = dict(_DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    engine = p.pop("engine", "class")
    return Cosmology(engine=engine, **p)


def get_linear_power(cosmo, k: np.ndarray, z: float) -> np.ndarray:
    """Return the linear matter power spectrum P_lin(k, z).

    Parameters
    ----------
    cosmo : cosmoprimo.Cosmology
    k : array_like
        Wavenumbers in h/Mpc.
    z : float
        Redshift.

    Returns
    -------
    np.ndarray, shape (len(k),)
        P_lin in (Mpc/h)^3.
    """
    k = np.asarray(k, dtype=float)
    pk_interp = cosmo.get_fourier().pk_interpolator(non_linear=False, of="delta_m")
    return pk_interp(k, z)


def get_growth_rate(cosmo, z: float) -> float:
    """Return the linear growth rate f = d ln D / d ln a at redshift z.

    Parameters
    ----------
    cosmo : cosmoprimo.Cosmology
    z : float

    Returns
    -------
    float
    """
    return float(cosmo.get_background().growth_rate(z))


class LinearPowerGrid:
    """Precomputed grid of P_lin(k) and f(z) over (sigma8, Omega_m).

    Evaluates via cubic spline interpolation — ~0.01 ms per call.

    Parameters
    ----------
    k : array_like, shape (nk,)
    z : float
    sigma8_range : tuple (min, max, n), default (0.6, 1.0, 20)
    omega_m_range : tuple (min, max, n), default (0.2, 0.5, 20)
    fixed_params : dict, optional
        Fixed parameters (h, Omega_b, n_s). Defaults to Planck 2018.
    """

    def __init__(self, k, z,
                 sigma8_range=(0.6, 1.2, 20),
                 omega_m_range=(0.2, 0.5, 20),
                 fixed_params=None):
        k = np.asarray(k, dtype=float)
        s8_vals = np.linspace(*sigma8_range)
        om_vals = np.linspace(*omega_m_range)

        plin_grid = np.empty((len(s8_vals), len(om_vals), len(k)))
        f_grid    = np.empty((len(s8_vals), len(om_vals)))
        for i, s8 in enumerate(s8_vals):
            for j, om in enumerate(om_vals):
                p = dict(fixed_params or {})
                p.update({"sigma8": float(s8), "Omega_m": float(om)})
                cosmo = get_cosmology(p)
                plin_grid[i, j] = get_linear_power(cosmo, k, z)
                f_grid[i, j]    = get_growth_rate(cosmo, z)

        self._plin_interp = RegularGridInterpolator(
            (s8_vals, om_vals), plin_grid, method="cubic",
            bounds_error=True,
        )
        self._f_interp = RegularGridInterpolator(
            (s8_vals, om_vals), f_grid, method="cubic",
            bounds_error=True,
        )
        self._s8_range = (sigma8_range[0], sigma8_range[1])
        self._om_range = (omega_m_range[0], omega_m_range[1])

    def predict(self, sigma8: float, omega_m: float):
        """Return (plin, f) at the given cosmological parameters.

        Raises ValueError if parameters are outside the grid bounds.
        """
        pt = np.array([[sigma8, omega_m]])
        try:
            plin = self._plin_interp(pt)[0]
            f    = float(self._f_interp(pt)[0])
        except ValueError as exc:
            raise ValueError(
                f"Parameters (sigma8={sigma8}, Omega_m={omega_m}) are outside "
                f"the grid bounds sigma8={self._s8_range}, "
                f"Omega_m={self._om_range}."
            ) from exc
        return plin, f


class OneLoopPowerGrid(LinearPowerGrid):
    """Like LinearPowerGrid but also precomputes one-loop integrals on the grid.

    Uses a coarser default grid (10×10) to limit initialisation time.
    ``predict()`` returns ``(plin, f, loop_arrays)`` where ``loop_arrays`` is
    a dict with keys 'p22', 'p13', 'I12', 'J12', 'I22', 'I2K', 'J22',
    'p22_dt', 'p22_tt', 'p13_dt', 'p13_tt'.

    Parameters
    ----------
    k : array_like, shape (nk,)
    z : float
    sigma8_range : tuple (min, max, n), default (0.6, 1.2, 10)
    omega_m_range : tuple (min, max, n), default (0.2, 0.5, 10)
    fixed_params : dict, optional
    """

    _LOOP_KEYS = ("p22", "p13", "I12", "J12", "I22", "I2K", "J22",
                   "I12_v", "J12_v", "Ib3nl",
                   "p22_dt", "p22_tt", "p13_dt", "p13_tt")

    def __init__(self, k, z,
                 sigma8_range=(0.6, 1.2, 10),
                 omega_m_range=(0.2, 0.5, 10),
                 fixed_params=None):
        # Build plin/f grids via parent
        super().__init__(k, z, sigma8_range, omega_m_range, fixed_params)

        from .galaxy_models import _compute_loop_templates

        k = np.asarray(k, dtype=float)
        s8_vals = np.linspace(*sigma8_range)
        om_vals = np.linspace(*omega_m_range)

        loop_grids = {key: np.empty((len(s8_vals), len(om_vals), len(k)))
                      for key in self._LOOP_KEYS}

        for i, s8 in enumerate(s8_vals):
            for j, om in enumerate(om_vals):
                p = dict(fixed_params or {})
                p.update({"sigma8": float(s8), "Omega_m": float(om)})
                cosmo = get_cosmology(p)
                def plin_func(kk, _cosmo=cosmo):
                    return get_linear_power(_cosmo, np.asarray(kk, dtype=float), z)
                loops = _compute_loop_templates(k, plin_func)
                for key in self._LOOP_KEYS:
                    loop_grids[key][i, j] = loops[key]

        self._loop_interps = {
            key: RegularGridInterpolator(
                (s8_vals, om_vals), loop_grids[key], method="cubic",
                bounds_error=True,
            )
            for key in self._LOOP_KEYS
        }

    def predict(self, sigma8: float, omega_m: float):
        """Return (plin, f, loop_arrays) at the given cosmological parameters."""
        plin, f = super().predict(sigma8, omega_m)
        pt = np.array([[sigma8, omega_m]])
        try:
            loop_arrays = {key: self._loop_interps[key](pt)[0]
                           for key in self._LOOP_KEYS}
        except ValueError as exc:
            raise ValueError(
                f"Parameters (sigma8={sigma8}, Omega_m={omega_m}) are outside "
                f"the grid bounds sigma8={self._s8_range}, "
                f"Omega_m={self._om_range}."
            ) from exc
        return plin, f, loop_arrays
