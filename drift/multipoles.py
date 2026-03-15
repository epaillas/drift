"""Legendre multipole projection utilities."""

import numpy as np
from scipy.special import legendre as scipy_legendre
from scipy.integrate import quad


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
