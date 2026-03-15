"""Cosmology utilities: wraps cosmoprimo to provide P_lin(k,z) and f(z)."""

import numpy as np

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
