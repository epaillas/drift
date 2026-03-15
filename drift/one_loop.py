"""One-loop matter power spectrum: P22 and P13 via SPT."""

import numpy as np


def F2_kernel(k1: float, k2: float, cos_theta: float) -> float:
    """Standard SPT F2 kernel.

    Parameters
    ----------
    k1, k2 : float
        Magnitudes of the two wavevectors.
    cos_theta : float
        Cosine of the angle between them.

    Returns
    -------
    float
    """
    if k1 == 0.0 or k2 == 0.0:
        return 0.0
    return (
        5.0 / 7.0
        + 0.5 * (k1 / k2 + k2 / k1) * cos_theta
        + 2.0 / 7.0 * cos_theta ** 2
    )


def _I13_angular(k_val: float, q_arr: np.ndarray) -> np.ndarray:
    """Analytic angular integral over the symmetrized F3 kernel for P13.

    Implements the standard result (Bernardeau et al. 2002, eq. A17):

        I13(k, q) = 1/252 * [12/x^2 - 158 + 100*x^2 - 42*x^4
                             + 3/x^3 * (x^2-1)^3 * (7*x^2+2) * ln|(1+x)/(1-x)|]

    where x = q / k. A Taylor expansion around x -> 1 (q -> k) is used
    when |x - 1| < 1e-6 to avoid the log singularity.

    Parameters
    ----------
    k_val : float
        Reference wavenumber.
    q_arr : np.ndarray, shape (nq,)
        Radial integration variable array.

    Returns
    -------
    np.ndarray, shape (nq,)
    """
    x = q_arr / k_val
    result = np.zeros_like(x)

    regular = np.abs(x - 1.0) >= 1e-6

    # Regular evaluation
    xr = x[regular]
    term_poly = 12.0 / xr**2 - 158.0 + 100.0 * xr**2 - 42.0 * xr**4
    log_arg = np.abs((1.0 + xr) / (1.0 - xr))
    term_log = (3.0 / xr**3) * (xr**2 - 1.0)**3 * (7.0 * xr**2 + 2.0) * np.log(log_arg)
    result[regular] = (term_poly + term_log) / 252.0

    # Taylor expansion around x = 1 (q -> k): result ~ -16/21 + higher
    xs = x[~regular]
    dx = xs - 1.0
    # Leading terms of the Taylor series around x=1
    result[~regular] = -16.0 / 21.0 + (16.0 / 7.0) * dx - (4.0 / 7.0) * dx**2

    return result


def compute_P22(
    k,
    plin_func,
    q_min: float = 1e-4,
    q_max: float = 10.0,
    n_q: int = 128,
    n_mu: int = 128,
) -> np.ndarray:
    """Compute P22(k) via explicit 2D integral over (q, mu_q).

    Uses a log-spaced q grid and linear mu grid. The change of variable
    dq -> d(ln q) absorbs the Jacobian (int q^3 d(ln q) ...).

    Parameters
    ----------
    k : array_like, shape (nk,)
        Output wavenumbers in h/Mpc.
    plin_func : callable
        plin_func(k_arr) -> P_lin array, shape (nk,).
    q_min, q_max : float
        Integration limits in h/Mpc.
    n_q : int
        Number of log-spaced q points.
    n_mu : int
        Number of linear mu_q points in [-1, 1].

    Returns
    -------
    np.ndarray, shape (nk,)
    """
    k = np.asarray(k, dtype=float)
    q_arr = np.geomspace(q_min, q_max, n_q)         # (nq,)
    mu_arr = np.linspace(-1.0, 1.0, n_mu)           # (nmu,)
    ln_q = np.log(q_arr)

    plin_q = plin_func(q_arr)                        # (nq,)

    result = np.zeros(len(k))

    for i, ki in enumerate(k):
        # k2 magnitude via law of cosines: |k - q|
        # k2^2 = ki^2 + q^2 - 2*ki*q*mu
        q2d = q_arr[:, np.newaxis]          # (nq, 1)
        mu2d = mu_arr[np.newaxis, :]        # (1, nmu)

        k2_sq = ki**2 + q2d**2 - 2.0 * ki * q2d * mu2d   # (nq, nmu)
        k2_sq = np.maximum(k2_sq, 0.0)
        k2 = np.sqrt(k2_sq)                                # (nq, nmu)

        # cos_theta between q and k2 = k - q
        # cos(q, k2) = (ki * mu - q) / k2; handle k2 -> 0
        with np.errstate(invalid="ignore", divide="ignore"):
            cos12 = np.where(k2 > 1e-10, (ki * mu2d - q2d) / k2, 0.0)

        f2 = (
            5.0 / 7.0
            + 0.5 * (q2d / k2 + k2 / q2d) * cos12
            + 2.0 / 7.0 * cos12**2
        )
        # Mask k2 -> 0 (P_lin(0) = 0, so integrand vanishes)
        f2 = np.where(k2 > 1e-10, f2, 0.0)

        plin_k2 = plin_func(k2.ravel()).reshape(k2.shape)   # (nq, nmu)

        # Integrand: 2 * q^3 * plin(q) * plin(k2) * F2^2  (in ln q space)
        integrand = 2.0 * q2d**3 * plin_q[:, np.newaxis] * plin_k2 * f2**2

        # Integrate over mu first (trapezoidal), then over ln q
        mu_integral = np.trapz(integrand, mu_arr, axis=1)   # (nq,)
        result[i] = np.trapz(mu_integral, ln_q)

    # Prefactor: azimuthal integral of d^3q/(2pi)^3 gives 2pi/(2pi)^3 = 1/(2pi)^2 = 1/(4pi^2).
    # The leading factor of 2 in the SPT P22 definition is already in the integrand above.
    result *= 1.0 / (4.0 * np.pi**2)
    return result


def compute_P13(
    k,
    plin_func,
    q_min: float = 1e-4,
    q_max: float = 10.0,
    n_q: int = 256,
) -> np.ndarray:
    """Compute 2*P13(k) via 1D radial integral using analytic F3 angular average.

    2*P13(k) = 6 * P_lin(k) * int_0^inf dq/(4*pi^2) q^2 P_lin(q) * I13(k, q)

    Derivation: d^3q/(2pi)^3 = q^2 dq dmu / (4*pi^2) after azimuthal integration.
    I13(k,q) = int_{-1}^{1} dmu F3^(s)(k,q,-q) is the analytic angular average.

    The integration is performed over ln(q) with np.trapz.
    Sign convention: 2*P13 is typically negative at intermediate k.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Output wavenumbers in h/Mpc.
    plin_func : callable
        plin_func(k_arr) -> P_lin array.
    q_min, q_max : float
        Integration limits in h/Mpc.
    n_q : int
        Number of log-spaced q points.

    Returns
    -------
    np.ndarray, shape (nk,)
        Values of 2*P13(k).
    """
    k = np.asarray(k, dtype=float)
    q_arr = np.geomspace(q_min, q_max, n_q)   # (nq,)
    ln_q = np.log(q_arr)
    plin_q = plin_func(q_arr)                  # (nq,)

    result = np.zeros(len(k))
    for i, ki in enumerate(k):
        plin_ki = float(plin_func(np.array([ki]))[0])
        I13 = _I13_angular(ki, q_arr)                 # (nq,)
        # integrand in ln q: q^3 * plin(q) * I13(k,q)
        integrand = q_arr**3 * plin_q * I13
        integral = np.trapz(integrand, ln_q)
        result[i] = 6.0 * plin_ki * integral / (4.0 * np.pi**2)

    return result


def compute_one_loop_matter(
    k,
    plin_func,
    q_min: float = 1e-4,
    q_max: float = 10.0,
    n_q_22: int = 128,
    n_mu_22: int = 128,
    n_q_13: int = 256,
) -> dict:
    """Compute the one-loop matter power spectrum.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Output wavenumbers in h/Mpc.
    plin_func : callable
        plin_func(k_arr) -> P_lin array.
    q_min, q_max : float
        Integration limits in h/Mpc.
    n_q_22, n_mu_22 : int
        Grid sizes for the P22 2D integral.
    n_q_13 : int
        Grid size for the P13 1D integral.

    Returns
    -------
    dict with keys 'plin', 'p22', 'p13', 'p1loop'
        p1loop = plin + p22 + p13 (with p13 = 2*P13 in SPT notation).
    """
    k = np.asarray(k, dtype=float)
    plin = plin_func(k)
    p22 = compute_P22(k, plin_func, q_min=q_min, q_max=q_max, n_q=n_q_22, n_mu=n_mu_22)
    p13 = compute_P13(k, plin_func, q_min=q_min, q_max=q_max, n_q=n_q_13)
    return {
        "plin": plin,
        "p22": p22,
        "p13": p13,
        "p1loop": plin + p22 + p13,
    }
