"""IR resummation: wiggle/no-wiggle decomposition and BAO damping."""

import numpy as np


def eisenstein_hu_nowiggle(k, h, Omega_m, Omega_b, T_cmb=2.7255):
    """Eisenstein & Hu (1998) no-wiggle transfer function T(k).

    Implements Eq. 29-31 of Eisenstein & Hu (1998, ApJ 496, 605).
    Returns the *transfer function* T(k), not the power spectrum.

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Total matter density parameter.
    Omega_b : float
        Baryon density parameter.
    T_cmb : float
        CMB temperature in Kelvin (default 2.7255).

    Returns
    -------
    np.ndarray
        Transfer function T(k), same shape as k.
    """
    k = np.asarray(k, dtype=float)
    theta = T_cmb / 2.7
    omega_m = Omega_m * h ** 2
    omega_b = Omega_b * h ** 2
    f_b = omega_b / omega_m

    # Sound horizon (Eq. 26)
    s = 44.5 * np.log(9.83 / omega_m) / np.sqrt(1.0 + 10.0 * omega_b ** 0.75)

    # Effective shape parameter (Eq. 31)
    alpha_gamma = (
        1.0
        - 0.328 * np.log(431.0 * omega_m) * f_b
        + 0.38 * np.log(22.3 * omega_m) * f_b ** 2
    )
    gamma_eff = Omega_m * h * (
        alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s) ** 4)
    )

    # No-wiggle transfer function (Eq. 29)
    q = k * theta ** 2 / gamma_eff
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T_nw = L0 / (L0 + C0 * q ** 2)

    return T_nw


def split_wiggle_nowiggle(k, plin, h, Omega_m, Omega_b, n_s, poly_order=4):
    """Split P_lin into smooth (no-wiggle) and wiggly parts.

    Uses the Eisenstein-Hu no-wiggle transfer function with a polynomial
    correction in log-space to match the broad-band shape of P_lin.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    plin : array_like, shape (nk,)
        Linear power spectrum P_lin(k).
    h, Omega_m, Omega_b, n_s : float
        Cosmological parameters.
    poly_order : int
        Order of the log-space polynomial correction (default 4).

    Returns
    -------
    pnw : np.ndarray, shape (nk,)
        No-wiggle (smooth) power spectrum.
    pw : np.ndarray, shape (nk,)
        Wiggly residual: pw = plin - pnw.
    """
    k = np.asarray(k, dtype=float)
    plin = np.asarray(plin, dtype=float)

    T_nw = eisenstein_hu_nowiggle(k, h, Omega_m, Omega_b)
    pnw_shape = T_nw ** 2 * k ** n_s

    # Fit polynomial correction in log-space to capture broad-band mismatch
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            pnw_shape > 0, np.log(plin / pnw_shape), 0.0
        )
    mask = np.isfinite(log_ratio) & (pnw_shape > 0) & (plin > 0)
    if np.sum(mask) > poly_order + 1:
        coeffs = np.polyfit(np.log(k[mask]), log_ratio[mask], poly_order)
        log_correction = np.polyval(coeffs, np.log(k))
    else:
        log_correction = np.zeros_like(k)

    pnw = pnw_shape * np.exp(log_correction)
    pw = plin - pnw

    return pnw, pw


def compute_sigma_dd(plin_func, q_min=1e-4, q_max=10.0, n_q=512):
    """Compute the 1D displacement variance (per component).

    sigma_dd = (1 / 6pi^2) int_0^inf dq P_lin(q)

    Parameters
    ----------
    plin_func : callable
        plin_func(q) -> P_lin(q).
    q_min, q_max : float
        Integration limits in h/Mpc.
    n_q : int
        Number of log-spaced quadrature points.

    Returns
    -------
    float
        Displacement variance in (Mpc/h)^2.
    """
    q = np.geomspace(q_min, q_max, n_q)
    plin_q = plin_func(q)
    return float(np.trapz(plin_q, q) / (6.0 * np.pi ** 2))


def ir_damping(k, mu, f, sigma_dd):
    """Compute the IR resummation damping factor for the BAO feature.

    D(k, mu) = exp(-k^2 Sigma^2_tot(mu))

    where Sigma^2_tot(mu) = sigma_dd * [1 + f(2+f) mu^2],
    with Sigma_perp = sigma_dd, Sigma_par = (1+f)^2 * sigma_dd.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    f : float
        Linear growth rate.
    sigma_dd : float
        1D displacement variance in (Mpc/h)^2.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
        Damping factor D(k, mu).
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    sigma_tot_sq = sigma_dd * (1.0 + f * (2.0 + f) * mu ** 2)  # (nmu,)
    return np.exp(-k[:, np.newaxis] ** 2 * sigma_tot_sq[np.newaxis, :])
