"""Legendre multipole projection utilities."""

import numpy as np
from cosmoprimo.fftlog import PowerToCorrelation
from scipy.special import legendre as scipy_legendre


def legendre(ell: int, mu: np.ndarray) -> np.ndarray:
    """Evaluate the Legendre polynomial P_ell(mu).

    Parameters
    ----------
    ell : int
        Multipole order (0, 2, 4, ...).
    mu : array_like
        Evaluation points in [-1, 1].

    Returns
    -------
    np.ndarray
    """
    mu = np.asarray(mu, dtype=float)
    return scipy_legendre(ell)(mu)


def _gauss_legendre_grid(n: int = 200):
    """Return n-point Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def _validate_fftlog_k_grid(k: np.ndarray, rtol: float = 1e-4) -> np.ndarray:
    """Validate that k is strictly increasing and close to log-spaced."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1:
        raise ValueError("k must be a one-dimensional array.")
    if k.size < 2:
        raise ValueError("k must contain at least two points.")
    if np.any(k <= 0.0):
        raise ValueError("k must be strictly positive for FFTLog transforms.")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("k must be strictly increasing for FFTLog transforms.")

    log_diffs = np.diff(np.log(k))
    if not np.allclose(log_diffs, log_diffs[0], rtol=rtol, atol=0.0):
        raise ValueError("k must be uniformly log-spaced for FFTLog transforms.")
    return k


def project_multipole(
    k: np.ndarray,
    p_of_mu_func,
    ell: int,
    mu_grid=None,
) -> np.ndarray:
    """Project P(k, mu) onto Legendre multipole P_ell(k).

    P_ell(k) = (2*ell+1)/2 * integral_{-1}^{1} dmu P(k,mu) L_ell(mu)

    Uses Gauss-Legendre quadrature by default.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers.
    p_of_mu_func : callable
        Function p_of_mu_func(k, mu) -> array of shape (nk, nmu).
    ell : int
        Multipole order.
    mu_grid : tuple (mu_nodes, weights) or None
        If None, uses 200-point Gauss-Legendre grid.

    Returns
    -------
    np.ndarray, shape (nk,)
    """
    k = np.asarray(k, dtype=float)

    if mu_grid is None:
        mu_nodes, weights = _gauss_legendre_grid(200)
    else:
        mu_nodes, weights = mu_grid

    # P(k, mu): shape (nk, nmu)
    pkmu = p_of_mu_func(k, mu_nodes)

    # Legendre polynomial evaluated on quadrature nodes: shape (nmu,)
    leg = legendre(ell, mu_nodes)

    # Numerical integration: sum_j w_j * P(k, mu_j) * L_ell(mu_j)
    integrand = pkmu * leg[np.newaxis, :]  # (nk, nmu)
    integral = integrand @ weights          # (nk,)

    return (2 * ell + 1) / 2.0 * integral


def compute_multipoles(
    k: np.ndarray,
    model_callable,
    ells=(0, 2, 4),
    mu_grid=None,
    **model_kwargs,
) -> dict:
    """Compute Legendre multipoles for a given model.

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    model_callable : callable
        Function model_callable(k, mu, **model_kwargs) -> array (nk, nmu).
    ells : tuple of int
        Multipole orders to compute.
    mu_grid : tuple or None
        Gauss-Legendre grid (nodes, weights). Defaults to 200-point grid.
    **model_kwargs
        Extra keyword arguments passed to model_callable.

    Returns
    -------
    dict
        {ell: np.ndarray of shape (nk,)} for each ell in ells.
    """
    k = np.asarray(k, dtype=float)

    if mu_grid is None:
        mu_grid = _gauss_legendre_grid(200)

    def p_of_mu(kk, mu):
        return model_callable(kk, mu, **model_kwargs)

    return {
        ell: project_multipole(k, p_of_mu, ell, mu_grid=mu_grid)
        for ell in ells
    }


def power_to_correlation_multipoles(
    k: np.ndarray,
    poles: dict,
    ells=(0, 2, 4),
    q=1.0,
    extrap="log",
    **fftlog_kwargs,
) -> tuple:
    """Transform power-spectrum multipoles into correlation-function multipoles."""
    k = _validate_fftlog_k_grid(k)
    ells = tuple(ells)

    missing = [ell for ell in ells if ell not in poles]
    if missing:
        raise ValueError(f"Missing power-spectrum multipoles for ell={missing}.")

    pk = np.vstack([np.asarray(poles[ell], dtype=float) for ell in ells])
    if pk.shape[1] != k.size:
        raise ValueError("Each multipole array must have the same length as k.")

    s, xi = PowerToCorrelation(k, ell=ells, q=q, **fftlog_kwargs)(pk, extrap=extrap)
    if np.ndim(s) == 2:
        s = np.asarray(s[0], dtype=float)
    else:
        s = np.asarray(s, dtype=float)
    xi = np.asarray(xi, dtype=float)
    return s, {ell: xi[i] for i, ell in enumerate(ells)}


def compute_correlation_multipoles(
    k: np.ndarray,
    model_callable,
    ells=(0, 2, 4),
    mu_grid=None,
    q=1.0,
    extrap="log",
    fftlog_kwargs=None,
    **model_kwargs,
) -> tuple:
    """Project P(k, mu) to multipoles and transform them to xi_ell(s)."""
    if fftlog_kwargs is None:
        fftlog_kwargs = {}
    poles = compute_multipoles(
        k,
        model_callable,
        ells=ells,
        mu_grid=mu_grid,
        **model_kwargs,
    )
    return power_to_correlation_multipoles(
        k,
        poles,
        ells=ells,
        q=q,
        extrap=extrap,
        **fftlog_kwargs,
    )
