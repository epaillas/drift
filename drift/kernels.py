"""Smoothing kernels W_R(k) for density-split statistics."""

import numpy as np


def gaussian_kernel(k: np.ndarray, R: float) -> np.ndarray:
    """Gaussian smoothing kernel (cross-spectrum convention, one W_R factor).

    W_R(k) = exp[-(kR)^2 / 2]

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    R : float
        Smoothing radius in Mpc/h.

    Returns
    -------
    np.ndarray
    """
    k = np.asarray(k, dtype=float)
    return np.exp(-0.5 * (k * R) ** 2)


def tophat_kernel(k: np.ndarray, R: float) -> np.ndarray:
    """Real-space top-hat smoothing kernel.

    W_R(k) = 3 [sin(kR) - kR cos(kR)] / (kR)^3

    The k=0 limit (= 1) is handled analytically.

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    R : float
        Smoothing radius in Mpc/h.

    Returns
    -------
    np.ndarray
    """
    k = np.asarray(k, dtype=float)
    x = k * R
    # Avoid division by zero; use Taylor expansion for small x
    result = np.ones_like(x)
    mask = np.abs(x) > 1e-6
    xm = x[mask]
    result[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / xm ** 3
    return result
