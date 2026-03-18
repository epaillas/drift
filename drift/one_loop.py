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

    Implements the *EFT-renormalized* version of the standard result
    (Bernardeau et al. 2002, eq. A17).  The raw kernel approaches
    -122/315 as x = q/k -> infinity, generating a UV-sensitive
    contribution proportional to P_lin(k) that is absorbed by
    renormalization of the linear bias b1.  We subtract this constant
    so that the loop integral yields the finite, scale-dependent
    one-loop correction only.

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

    near_one = np.abs(x - 1.0) < 1e-6
    large_x = x > 50.0
    regular = ~near_one & ~large_x

    # Regular evaluation (subtract UV constant via modified polynomial)
    xr = x[regular]
    # Original poly: 12/x^2 - 158 + 100*x^2 - 42*x^4
    # Add 252 * 122/315 = 488/5 = 97.6 to subtract the UV asymptotic
    term_poly = 12.0 / xr**2 - 302.0 / 5.0 + 100.0 * xr**2 - 42.0 * xr**4
    log_arg = np.abs((1.0 + xr) / (1.0 - xr))
    term_log = (3.0 / xr**3) * (xr**2 - 1.0)**3 * (7.0 * xr**2 + 2.0) * np.log(log_arg)
    result[regular] = (term_poly + term_log) / 252.0

    # Taylor expansion around x = 1 (q -> k)
    xs = x[near_one]
    dx = xs - 1.0
    # Renormalized leading term: -16/21 + 122/315 = -118/315
    result[near_one] = -118.0 / 315.0 + (16.0 / 7.0) * dx - (4.0 / 7.0) * dx**2

    # Asymptotic expansion for large x (avoids catastrophic cancellation)
    # I13_reg(x) = (96/(5*x^2) - 160/(21*x^4) + ...) / 252
    xl = x[large_x]
    result[large_x] = (96.0 / (5.0 * xl**2) - 160.0 / (21.0 * xl**4)) / 252.0

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
        Number of Gauss-Legendre quadrature points in [-1, 1].

    Returns
    -------
    np.ndarray, shape (nk,)
    """
    k = np.asarray(k, dtype=float)
    q_arr = np.geomspace(q_min, q_max, n_q)         # (nq,)
    mu_arr, mu_weights = np.polynomial.legendre.leggauss(n_mu)  # (nmu,)
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

        # Integrate over mu first (Gauss-Legendre), then over ln q
        mu_integral = integrand @ mu_weights                 # (nq,)
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


def _G2_spt(k1: float, k2: float, cos_theta: float) -> float:
    """SPT G2 kernel for the velocity divergence field.

    Differs from the density F2 kernel: uses coefficients 3/7 and 4/7
    instead of 5/7 and 2/7.
    """
    if k1 == 0.0 or k2 == 0.0:
        return 0.0
    return 3.0 / 7.0 + 0.5 * (k1 / k2 + k2 / k1) * cos_theta + 4.0 / 7.0 * cos_theta ** 2


def _I13_G_angular(k_val: float, q_arr: np.ndarray) -> np.ndarray:
    """Analytic angular integral over the symmetrized G3 kernel for P13_tt.

    EFT-renormalized: the raw kernel approaches -1/5 as x -> infinity.
    This UV constant is subtracted so that the loop integral yields
    only the finite, scale-dependent correction.

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

    near_one = np.abs(x - 1.0) < 1e-6
    large_x = x > 50.0
    regular = ~near_one & ~large_x

    # Regular evaluation (subtract UV constant via modified polynomial)
    xr = x[regular]
    # Original poly: 12/x^2 - 82 + 4*x^2 - 6*x^4
    # Add 504 * 1/5 = 100.8 to subtract the UV asymptotic
    term_poly = 12.0 / xr ** 2 + 18.8 + 4.0 * xr ** 2 - 6.0 * xr ** 4
    log_arg = np.abs((1.0 + xr) / (1.0 - xr))
    term_log = (3.0 / xr ** 3) * (xr ** 2 - 1.0) ** 3 * (xr ** 2 + 2.0) * np.log(log_arg)
    result[regular] = (term_poly + term_log) / 504.0

    # Taylor expansion around x = 1
    xs = x[~near_one & ~large_x]  # not used; near_one handled below
    xs = x[near_one]
    dx = xs - 1.0
    # Renormalized leading term: -72/504 + 1/5 = -1/7 + 1/5 = 2/35
    result[near_one] = (28.8 - 40.0 * dx + 4.0 * dx ** 2) / 504.0

    # Asymptotic expansion for large x
    # I13_G_reg(x) = (1248/(35*x^2) - 608/(105*x^4) + ...) / 504
    xl = x[large_x]
    result[large_x] = (1248.0 / (35.0 * xl**2) - 608.0 / (105.0 * xl**4)) / 504.0

    return result


def compute_Pdt_Ptt(
    k,
    plin_func,
    p13_dd,
    q_min: float = 1e-4,
    q_max: float = 10.0,
    n_q: int = 128,
    n_mu: int = 128,
    n_q_13: int = 256,
) -> dict:
    """Compute P22_dt, P22_tt, P13_dt, P13_tt for redshift-space one-loop.

    The redshift-space P_gg decomposes as:
        P_dd: density auto (isotropic)
        P_dt: density × velocity (∝ mu^2)
        P_tt: velocity auto (∝ mu^4)

    P22_dt and P22_tt use F2*G2_spt and G2_spt^2 kernels respectively.
    P13_tt uses the G3 angular average (``_I13_G_angular``).
    P13_dt uses the identity: P13_dt = (P13_dd + P13_tt) / 2.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Output wavenumbers in h/Mpc.
    plin_func : callable
        plin_func(k_arr) -> P_lin array, shape (nk,).
    p13_dd : np.ndarray, shape (nk,)
        The density P13 array (from ``compute_P13``), needed for P13_dt.
    q_min, q_max : float
        Integration limits in h/Mpc.
    n_q, n_mu : int
        Grid sizes for the P22 2D integrals.
    n_q_13 : int
        Grid size for the P13_tt 1D integral.

    Returns
    -------
    dict with keys 'p22_dt', 'p22_tt', 'p13_dt', 'p13_tt', each shape (nk,).
    """
    k = np.asarray(k, dtype=float)
    q_arr = np.geomspace(q_min, q_max, n_q)
    mu_arr, mu_weights = np.polynomial.legendre.leggauss(n_mu)
    ln_q = np.log(q_arr)

    plin_q = plin_func(q_arr)

    p22_dt = np.zeros(len(k))
    p22_tt = np.zeros(len(k))

    for i, ki in enumerate(k):
        q2d = q_arr[:, np.newaxis]
        mu2d = mu_arr[np.newaxis, :]

        k2_sq = ki ** 2 + q2d ** 2 - 2.0 * ki * q2d * mu2d
        k2_sq = np.maximum(k2_sq, 0.0)
        k2 = np.sqrt(k2_sq)

        with np.errstate(invalid="ignore", divide="ignore"):
            cos12 = np.where(k2 > 1e-10, (ki * mu2d - q2d) / k2, 0.0)

        # F2 kernel (density)
        with np.errstate(invalid="ignore", divide="ignore"):
            f2 = 5.0 / 7.0 + 0.5 * (q2d / k2 + k2 / q2d) * cos12 + 2.0 / 7.0 * cos12 ** 2
        f2 = np.where(k2 > 1e-10, f2, 0.0)

        # G2_spt kernel (velocity divergence)
        with np.errstate(invalid="ignore", divide="ignore"):
            g2s = 3.0 / 7.0 + 0.5 * (q2d / k2 + k2 / q2d) * cos12 + 4.0 / 7.0 * cos12 ** 2
        g2s = np.where(k2 > 1e-10, g2s, 0.0)

        plin_k2 = plin_func(k2.ravel()).reshape(k2.shape)

        base = q2d ** 3 * plin_q[:, np.newaxis] * plin_k2

        ig_dt = 2.0 * base * f2 * g2s
        ig_tt = 2.0 * base * g2s ** 2

        def _integrate(ig):
            mu_int = ig @ mu_weights
            return np.trapz(mu_int, ln_q)

        p22_dt[i] = _integrate(ig_dt)
        p22_tt[i] = _integrate(ig_tt)

    prefactor = 1.0 / (4.0 * np.pi ** 2)
    p22_dt *= prefactor
    p22_tt *= prefactor

    # P13_tt via 1D integral with G3 angular kernel
    q_arr_13 = np.geomspace(q_min, q_max, n_q_13)
    ln_q_13 = np.log(q_arr_13)
    plin_q_13 = plin_func(q_arr_13)

    p13_tt = np.zeros(len(k))
    for i, ki in enumerate(k):
        plin_ki = float(plin_func(np.array([ki]))[0])
        I13_G = _I13_G_angular(ki, q_arr_13)
        integrand = q_arr_13 ** 3 * plin_q_13 * I13_G
        integral = np.trapz(integrand, ln_q_13)
        p13_tt[i] = 6.0 * plin_ki * integral / (4.0 * np.pi ** 2)

    # P13_dt = (P13_dd + P13_tt) / 2  [SPT symmetry identity]
    p13_dt = 0.5 * (np.asarray(p13_dd, dtype=float) + p13_tt)

    return {
        "p22_dt": p22_dt,
        "p22_tt": p22_tt,
        "p13_dt": p13_dt,
        "p13_tt": p13_tt,
    }


def G2_kernel(cos_theta: float) -> float:
    """Tidal bias kernel (traceless tidal tensor).

    Parameters
    ----------
    cos_theta : float
        Cosine of the angle between the two wavevectors.

    Returns
    -------
    float
    """
    return cos_theta ** 2 - 1.0 / 3.0


def compute_bias_loops(
    k,
    plin_func,
    q_min: float = 1e-4,
    q_max: float = 10.0,
    n_q: int = 128,
    n_mu: int = 128,
) -> dict:
    """Compute the 5 one-loop bias integrals in a single 2D pass.

    All integrals have the form::

        X(k) = int d^3q/(2pi)^3  K(q, k-q)  P_lin(q) P_lin(|k-q|)

    using the same (q, mu_q) grid as ``compute_P22``.

    Integrals returned
    ------------------
    I12 : kernel = F2(q, k-q, cos12)          [b1 × b2 cross]
    J12 : kernel = F2(q, k-q, cos12) * G2(cos12)  [b1 × bs2 cross]
    I22 : kernel = 1/2                        [b2^2 auto, trivial]
    I2K : kernel = G2(cos12) / 2              [b2 × bs2 cross]
    J22 : kernel = G2(cos12)^2 / 2            [bs2^2 auto]

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
        Number of Gauss-Legendre quadrature points in [-1, 1].

    Returns
    -------
    dict with keys 'I12', 'J12', 'I22', 'I2K', 'J22', each shape (nk,).
    """
    k = np.asarray(k, dtype=float)
    q_arr = np.geomspace(q_min, q_max, n_q)      # (nq,)
    mu_arr, mu_weights = np.polynomial.legendre.leggauss(n_mu)  # (nmu,)
    ln_q = np.log(q_arr)

    plin_q = plin_func(q_arr)                      # (nq,)

    I12 = np.zeros(len(k))
    J12 = np.zeros(len(k))
    I22 = np.zeros(len(k))
    I2K = np.zeros(len(k))
    J22 = np.zeros(len(k))

    for i, ki in enumerate(k):
        q2d = q_arr[:, np.newaxis]          # (nq, 1)
        mu2d = mu_arr[np.newaxis, :]        # (1, nmu)

        k2_sq = ki ** 2 + q2d ** 2 - 2.0 * ki * q2d * mu2d  # (nq, nmu)
        k2_sq = np.maximum(k2_sq, 0.0)
        k2 = np.sqrt(k2_sq)

        with np.errstate(invalid="ignore", divide="ignore"):
            cos12 = np.where(k2 > 1e-10, (ki * mu2d - q2d) / k2, 0.0)

        # F2 kernel (same as compute_P22)
        f2 = (
            5.0 / 7.0
            + 0.5 * (q2d / k2 + k2 / q2d) * cos12
            + 2.0 / 7.0 * cos12 ** 2
        )
        f2 = np.where(k2 > 1e-10, f2, 0.0)

        # G2 (tidal) kernel
        g2 = cos12 ** 2 - 1.0 / 3.0                          # (nq, nmu)

        plin_k2 = plin_func(k2.ravel()).reshape(k2.shape)     # (nq, nmu)

        base = q2d ** 3 * plin_q[:, np.newaxis] * plin_k2    # (nq, nmu)

        # Accumulate five integrands
        ig_I12 = base * f2                                      # kernel = F2
        ig_J12 = base * f2 * g2                                 # kernel = F2 * G2
        ig_I22 = 0.5 * base                                     # kernel = 1/2
        ig_I2K = 0.5 * base * g2                                # kernel = G2 / 2
        ig_J22 = 0.5 * base * g2 ** 2                           # kernel = G2^2 / 2

        def _integrate(ig):
            mu_int = ig @ mu_weights                           # (nq,)
            return np.trapz(mu_int, ln_q)

        I12[i] = _integrate(ig_I12)
        J12[i] = _integrate(ig_J12)
        I22[i] = _integrate(ig_I22)
        I2K[i] = _integrate(ig_I2K)
        J22[i] = _integrate(ig_J22)

    prefactor = 1.0 / (4.0 * np.pi ** 2)
    return {
        "I12": I12 * prefactor,
        "J12": J12 * prefactor,
        "I22": I22 * prefactor,
        "I2K": I2K * prefactor,
        "J22": J22 * prefactor,
    }


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
