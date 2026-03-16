"""Tree-level and EFT galaxy auto-power spectrum P(k, mu)."""

import numpy as np

from .cosmology import get_linear_power, get_growth_rate
from .eft_bias import GalaxyEFTParams
from .one_loop import compute_P22, compute_P13, compute_bias_loops, compute_Pdt_Ptt

_VALID_MODES = ("tree_only", "eft_lite", "eft_full", "one_loop")


def _compute_loop_templates(k, plin_func):
    """Compute all one-loop arrays needed for the one_loop mode.

    Returns
    -------
    dict with keys 'p22', 'p13', 'I12', 'J12', 'I22', 'I2K', 'J22',
    'p22_dt', 'p22_tt', 'p13_dt', 'p13_tt', each shape (nk,).
    """
    p22 = compute_P22(k, plin_func)
    p13 = compute_P13(k, plin_func)
    bias = compute_bias_loops(k, plin_func)
    vel = compute_Pdt_Ptt(k, plin_func, p13_dd=p13)
    return {"p22": p22, "p13": p13, **bias, **vel}


def pgg_mu(k, mu, z, cosmo, b1, space="redshift"):
    """Tree-level Kaiser galaxy auto-power spectrum P_gg(k, mu).

    In redshift space: P_gg = (b1 + f*mu^2)^2 * P_lin(k)
    In real space:     P_gg = b1^2 * P_lin(k)

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    mu : array_like, shape (nmu,)
        Cosine of angle to line of sight.
    z : float
        Redshift.
    cosmo : cosmoprimo.Cosmology
    b1 : float
        Linear galaxy bias.
    space : str
        'redshift' (default) or 'real'.

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    plin = get_linear_power(cosmo, k, z)   # (nk,)

    if space == "real":
        return (b1 ** 2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))

    f = get_growth_rate(cosmo, z)
    kaiser = b1 + f * mu ** 2   # (nmu,)
    return plin[:, np.newaxis] * kaiser[np.newaxis, :] ** 2


def pgg_eft_mu(k, mu, z, cosmo, gal_params, space="redshift", mode="eft_lite"):
    """EFT galaxy auto-power spectrum P_gg(k, mu).

    Modes
    -----
    tree_only : same as ``pgg_mu``
    eft_lite  : tree + galaxy EFT counterterm
        P_gg = (b1+f*mu^2)^2 * P_lin
               - 2*k^2*(c0+c2*mu^2+c4*mu^4)*(b1+f*mu^2)*P_lin
    eft_full  : eft_lite + stochastic term  s0 + s2*k^2

    Parameters
    ----------
    k : array_like, shape (nk,)
    mu : array_like, shape (nmu,)
    z : float
    cosmo : cosmoprimo.Cosmology
    gal_params : GalaxyEFTParams
    space : str
    mode : str

    Returns
    -------
    np.ndarray, shape (nk, nmu)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {_VALID_MODES}.")

    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    plin = get_linear_power(cosmo, k, z)   # (nk,)

    b1 = gal_params.b1

    if space == "real":
        if mode == "one_loop":
            def plin_func(kk):
                return get_linear_power(cosmo, np.asarray(kk, dtype=float), z)

            loops = _compute_loop_templates(k, plin_func)
            b2  = gal_params.b2
            bs2 = gal_params.bs2
            p_loop_bias = (
                2.0 * b1 * b2   * loops["I12"]
                + 2.0 * b1 * bs2 * loops["J12"]
                + b2 ** 2        * loops["I22"]
                + 2.0 * b2 * bs2 * loops["I2K"]
                + bs2 ** 2       * loops["J22"]
            )
            P_real = b1 ** 2 * (plin + loops["p22"] + loops["p13"]) + p_loop_bias
            P = P_real[:, np.newaxis] * np.ones((1, len(mu)))
        else:
            P = (b1 ** 2 * plin)[:, np.newaxis] * np.ones((1, len(mu)))
        if mode in ("eft_full", "one_loop"):
            P = P + (gal_params.s0 + gal_params.s2 * k ** 2)[:, np.newaxis]
        return P

    f = get_growth_rate(cosmo, z)

    if mode == "tree_only":
        return pgg_mu(k, mu, z, cosmo, b1, space=space)

    kaiser = b1 + f * mu ** 2   # (nmu,)
    P = plin[:, np.newaxis] * kaiser[np.newaxis, :] ** 2

    if mode in ("eft_lite", "eft_full"):
        c0 = gal_params.c0
        c2 = gal_params.c2
        c4 = gal_params.c4
        ct_shape = c0 + c2 * mu ** 2 + c4 * mu ** 4   # (nmu,)
        counterterm = (
            -2.0 * k ** 2 * plin
        )[:, np.newaxis] * ct_shape[np.newaxis, :] * kaiser[np.newaxis, :]
        P = P + counterterm

    if mode == "eft_full":
        P = P + (gal_params.s0 + gal_params.s2 * k ** 2)[:, np.newaxis]

    if mode == "one_loop":
        b2  = gal_params.b2
        bs2 = gal_params.bs2

        def plin_func(kk):
            return get_linear_power(cosmo, np.asarray(kk, dtype=float), z)

        loops = _compute_loop_templates(k, plin_func)
        p22, p13 = loops["p22"], loops["p13"]
        p22_dt, p22_tt = loops["p22_dt"], loops["p22_tt"]
        p13_dt, p13_tt = loops["p13_dt"], loops["p13_tt"]

        p_loop_bias = (
            2.0 * b1 * b2   * loops["I12"]
            + 2.0 * b1 * bs2 * loops["J12"]
            + b2 ** 2        * loops["I22"]
            + 2.0 * b2 * bs2 * loops["I2K"]
            + bs2 ** 2       * loops["J22"]
        )  # (nk,)

        # Density auto loop (isotropic, mu^0)
        P_dd_loop = b1 ** 2 * (p22 + p13)          # (nk,)
        # Density × velocity loop (∝ mu^2)
        P_dt_loop = p22_dt + p13_dt                 # (nk,)
        # Velocity auto loop (∝ mu^4)
        P_tt_loop = p22_tt + p13_tt                 # (nk,)

        P = P + P_dd_loop[:, np.newaxis] + p_loop_bias[:, np.newaxis]
        P = P + (2.0 * b1 * f) * mu[np.newaxis, :] ** 2 * P_dt_loop[:, np.newaxis]
        P = P + f ** 2 * mu[np.newaxis, :] ** 4 * P_tt_loop[:, np.newaxis]

        # EFT counterterm
        c0 = gal_params.c0
        c2 = gal_params.c2
        c4 = gal_params.c4
        ct_shape = c0 + c2 * mu ** 2 + c4 * mu ** 4
        counterterm = (
            -2.0 * k ** 2 * plin
        )[:, np.newaxis] * ct_shape[np.newaxis, :] * kaiser[np.newaxis, :]
        P = P + counterterm

        # Stochastic term
        P = P + (gal_params.s0 + gal_params.s2 * k ** 2)[:, np.newaxis]

    return P
