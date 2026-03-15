"""Tests for drift.multipoles."""

import numpy as np
import pytest
from scipy.special import legendre as scipy_legendre

from drift.multipoles import legendre, project_multipole, compute_multipoles


# ---- legendre ----------------------------------------------------------------

def test_legendre_ell0():
    mu = np.linspace(-1, 1, 20)
    np.testing.assert_allclose(legendre(0, mu), np.ones(20))


def test_legendre_ell2():
    mu = np.linspace(-1, 1, 20)
    expected = scipy_legendre(2)(mu)
    np.testing.assert_allclose(legendre(2, mu), expected)


def test_legendre_ell4():
    mu = np.linspace(-1, 1, 20)
    expected = scipy_legendre(4)(mu)
    np.testing.assert_allclose(legendre(4, mu), expected)


# ---- project_multipole -------------------------------------------------------

def test_constant_only_monopole():
    """A constant P(k,mu)=C contributes only to ell=0."""
    k = np.array([0.1, 0.2, 0.3])

    def p_const(kk, mu):
        return np.ones((len(kk), len(mu)))

    p0 = project_multipole(k, p_const, ell=0)
    p2 = project_multipole(k, p_const, ell=2)
    p4 = project_multipole(k, p_const, ell=4)

    np.testing.assert_allclose(p0, np.ones(3), rtol=1e-6)
    np.testing.assert_allclose(p2, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(p4, np.zeros(3), atol=1e-10)


def test_mu2_projection():
    """P(k,mu) = mu^2: analytical results for multipoles.

    mu^2 = (1/3) P_0 + (2/3) * (1/2)(3mu^2-1) correction...
    Actually: mu^2 = 1/3 * L_0 + 2/3 * (L_2 + 1/3*L_0)... let me derive:
    L_0 = 1, L_2 = (3mu^2-1)/2
    => mu^2 = (2*L_2 + L_0) / 3 + 1/3 - 1/3 = (2*L_2 + 1)/3

    P_0 = 1/2 * integral_{-1}^{1} mu^2 * 1 dmu = 1/2 * [mu^3/3]_{-1}^{1} = 1/2 * 2/3 = 1/3
    P_2 = 5/2 * integral_{-1}^{1} mu^2 * (3mu^2-1)/2 dmu
        = 5/4 * integral_{-1}^{1} (3mu^4 - mu^2) dmu
        = 5/4 * [3*2/5 - 2/3] = 5/4 * [6/5 - 2/3] = 5/4 * [18/15 - 10/15]
        = 5/4 * 8/15 = 40/60 = 2/3
    """
    k = np.array([0.1, 0.2])

    def p_mu2(kk, mu):
        return np.ones((len(kk), 1)) * mu[np.newaxis, :] ** 2

    p0 = project_multipole(k, p_mu2, ell=0)
    p2 = project_multipole(k, p_mu2, ell=2)

    np.testing.assert_allclose(p0, np.full(2, 1.0 / 3.0), rtol=1e-5)
    np.testing.assert_allclose(p2, np.full(2, 2.0 / 3.0), rtol=1e-5)


# ---- compute_multipoles ------------------------------------------------------

def test_compute_multipoles_keys():
    k = np.array([0.1, 0.2])

    def model(kk, mu):
        return np.ones((len(kk), len(mu)))

    result = compute_multipoles(k, model, ells=(0, 2, 4))
    assert set(result.keys()) == {0, 2, 4}
    for v in result.values():
        assert v.shape == (2,)
