"""Cosmology utilities: wraps cosmoprimo to provide P_lin(k,z) and f(z)."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_DEFAULT_PARAMS = {
    "h": 0.6766,
    "omega_cdm": 0.1200,
    "omega_b": 0.02243,
    "sigma8": 0.8102,
    "n_s": 0.9665,
    "engine": "class",
}

DEFAULT_COSMO_RANGES = {
    "sigma8":    (0.6, 1.0, 5),
    "omega_cdm": (0.10, 0.14, 5),
    "omega_b":   (0.019, 0.026, 5),
    "h":         (0.55, 0.80, 5),
    "n_s":       (0.90, 1.05, 5),
}

# Names of cosmological parameters (excludes "engine")
ALL_COSMO_NAMES = ("sigma8", "omega_cdm", "omega_b", "h", "n_s")


def get_cosmology(params: dict = None):
    """Return a cosmoprimo Cosmology object.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters. Accepts physical densities (omega_cdm, omega_b)
        or fractional densities (Omega_m, Omega_b). If both forms are present,
        the fractional form takes precedence (the physical default is removed).
        Defaults to Planck 2018-like values.

    Returns
    -------
    cosmoprimo.Cosmology
    """
    from cosmoprimo import Cosmology

    p = dict(_DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    # Resolve conflicts: if user passes Omega_m/Omega_b, drop omega_cdm/omega_b
    if "Omega_m" in p and "omega_cdm" in p:
        del p["omega_cdm"]
    if "Omega_b" in p and "omega_b" in p:
        del p["omega_b"]

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
    """Precomputed grid of P_lin(k) and f(z) over arbitrary cosmological axes.

    Evaluates via interpolation — ~0.01 ms per call.

    Parameters
    ----------
    k : array_like, shape (nk,)
    z : float
    cosmo_ranges : dict
        {param_name: (min, max, n)} for each axis to interpolate.
        E.g. {"sigma8": (0.6, 1.0, 5), "omega_cdm": (0.10, 0.14, 5)}
    fixed_params : dict, optional
        {param_name: value} for cosmological params held constant.
        Defaults come from _DEFAULT_PARAMS for any param not in cosmo_ranges.
    """

    def __init__(self, k, z, cosmo_ranges=None, fixed_params=None):
        if cosmo_ranges is None:
            cosmo_ranges = {
                "sigma8": DEFAULT_COSMO_RANGES["sigma8"],
                "omega_cdm": DEFAULT_COSMO_RANGES["omega_cdm"],
            }

        k = np.asarray(k, dtype=float)
        self._axis_names = list(cosmo_ranges.keys())
        self._axis_values = [np.linspace(*cosmo_ranges[name])
                             for name in self._axis_names]
        shape = tuple(len(v) for v in self._axis_values)

        # Fixed params: start from defaults, override with user-supplied.
        # Skip defaults that would conflict with user-provided axis names
        # (e.g. don't default omega_cdm if Omega_m is an axis, and vice versa).
        _CONFLICTS = {
            "omega_cdm": "Omega_m", "Omega_m": "omega_cdm",
            "omega_b": "Omega_b", "Omega_b": "omega_b",
        }
        range_names = set(cosmo_ranges.keys())
        self._fixed_params = {}
        for name in ALL_COSMO_NAMES:
            if name in range_names:
                continue
            # Skip if a conflicting param is being varied
            if name in _CONFLICTS and _CONFLICTS[name] in range_names:
                continue
            self._fixed_params[name] = _DEFAULT_PARAMS[name]
        if fixed_params is not None:
            self._fixed_params.update(fixed_params)

        n_total = int(np.prod(shape))
        method = "linear" if all(len(v) <= 5 for v in self._axis_values) else "cubic"

        plin_grid = np.empty(shape + (len(k),))
        f_grid = np.empty(shape)

        for count, idx in enumerate(np.ndindex(*shape)):
            # Merge varying + fixed params
            p = dict(self._fixed_params)
            for ax_i, name in enumerate(self._axis_names):
                p[name] = float(self._axis_values[ax_i][idx[ax_i]])

            cosmo = get_cosmology(p)
            plin_grid[idx] = get_linear_power(cosmo, k, z)
            f_grid[idx] = get_growth_rate(cosmo, z)

            if n_total > 1 and ((count + 1) % max(1, n_total // 10) == 0 or count == 0):
                print(f"  LinearPowerGrid: {count + 1}/{n_total} points")

        axes = tuple(self._axis_values)
        self._plin_interp = RegularGridInterpolator(
            axes, plin_grid, method=method, bounds_error=True,
        )
        self._f_interp = RegularGridInterpolator(
            axes, f_grid, method=method, bounds_error=True,
        )

    def predict(self, **cosmo_params):
        """Return (plin, f) at the given cosmological parameters.

        Parameters
        ----------
        **cosmo_params : float
            Must include all axis names from cosmo_ranges.

        Raises ValueError if parameters are outside the grid bounds.
        """
        pt = np.array([[cosmo_params[name] for name in self._axis_names]])
        try:
            plin = self._plin_interp(pt)[0]
            f = float(self._f_interp(pt)[0])
        except ValueError as exc:
            bounds_str = ", ".join(
                f"{name}=[{self._axis_values[i][0]:.4f}, {self._axis_values[i][-1]:.4f}]"
                for i, name in enumerate(self._axis_names)
            )
            given_str = ", ".join(
                f"{name}={cosmo_params.get(name, '?')}"
                for name in self._axis_names
            )
            raise ValueError(
                f"Parameters ({given_str}) are outside the grid bounds {bounds_str}."
            ) from exc
        return plin, f


class OneLoopPowerGrid(LinearPowerGrid):
    """Like LinearPowerGrid but also precomputes one-loop integrals on the grid.

    ``predict()`` returns ``(plin, f, loop_arrays)`` where ``loop_arrays`` is
    a dict with keys 'p22', 'p13', 'I12', 'J12', 'I22', 'I2K', 'J22',
    'p22_dt', 'p22_tt', 'p13_dt', 'p13_tt'.

    Parameters
    ----------
    k : array_like, shape (nk,)
    z : float
    cosmo_ranges : dict, optional
    fixed_params : dict, optional
    """

    _LOOP_KEYS = ("p22", "p13", "I12", "J12", "I22", "I2K", "J22",
                   "I12_v", "J12_v", "Ib3nl",
                   "p22_dt", "p22_tt", "p13_dt", "p13_tt")

    def __init__(self, k, z, cosmo_ranges=None, fixed_params=None):
        super().__init__(k, z, cosmo_ranges, fixed_params)

        from ..theory.galaxy import _compute_loop_templates

        k = np.asarray(k, dtype=float)
        shape = tuple(len(v) for v in self._axis_values)
        n_total = int(np.prod(shape))
        method = "linear" if all(len(v) <= 5 for v in self._axis_values) else "cubic"

        loop_grids = {key: np.empty(shape + (len(k),))
                      for key in self._LOOP_KEYS}

        for count, idx in enumerate(np.ndindex(*shape)):
            p = dict(self._fixed_params)
            for ax_i, name in enumerate(self._axis_names):
                p[name] = float(self._axis_values[ax_i][idx[ax_i]])

            cosmo = get_cosmology(p)

            def plin_func(kk, _cosmo=cosmo):
                return get_linear_power(_cosmo, np.asarray(kk, dtype=float), z)

            loops = _compute_loop_templates(k, plin_func)
            for key in self._LOOP_KEYS:
                loop_grids[key][idx] = loops[key]

            if n_total > 1 and ((count + 1) % max(1, n_total // 10) == 0 or count == 0):
                print(f"  OneLoopPowerGrid: {count + 1}/{n_total} points")

        axes = tuple(self._axis_values)
        self._loop_interps = {
            key: RegularGridInterpolator(
                axes, loop_grids[key], method=method, bounds_error=True,
            )
            for key in self._LOOP_KEYS
        }

    def predict(self, **cosmo_params):
        """Return (plin, f, loop_arrays) at the given cosmological parameters."""
        plin, f = super().predict(**cosmo_params)
        pt = np.array([[cosmo_params[name] for name in self._axis_names]])
        try:
            loop_arrays = {key: self._loop_interps[key](pt)[0]
                           for key in self._LOOP_KEYS}
        except ValueError as exc:
            bounds_str = ", ".join(
                f"{name}=[{self._axis_values[i][0]:.4f}, {self._axis_values[i][-1]:.4f}]"
                for i, name in enumerate(self._axis_names)
            )
            raise ValueError(
                f"Parameters outside grid bounds {bounds_str}."
            ) from exc
        return plin, f, loop_arrays
