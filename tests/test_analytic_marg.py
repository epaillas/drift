"""Tests for analytic marginalization of linear nuisance parameters."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.utils.cosmology import get_cosmology
from drift.emulators.galaxy import GalaxyTemplateEmulator
from drift.analytic_marginalization import MarginalizedLikelihood


@pytest.fixture
def cosmo():
    return get_cosmology()


@pytest.fixture
def k():
    return np.linspace(0.01, 0.3, 30)


# --------------------------------------------------------------------------
# Test 1: linear_param_names returns correct sets for each mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode, expected", [
    ("tree", []),
    ("eft_ct", ["c0"]),
    ("eft", ["c0", "s0"]),
    ("one_loop", ["c0", "c2", "c4", "s0", "s2", "b3nl"]),
])
def test_linear_param_names(cosmo, k, mode, expected):
    emu = GalaxyTemplateEmulator(cosmo, k, ells=(0, 2), z=0.5, space="redshift", mode=mode)
    assert emu.linear_param_names == expected


# --------------------------------------------------------------------------
# Test 2: Decomposition identity: m + T @ alpha == predict(full_params)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["eft_ct", "eft", "one_loop"])
def test_decomposition_identity(cosmo, k, mode):
    rng = np.random.default_rng(42)
    emu = GalaxyTemplateEmulator(cosmo, k, ells=(0, 2), z=0.5, space="redshift", mode=mode)
    lin_names = emu.linear_param_names

    for _ in range(5):
        params = {"b1": rng.uniform(0.5, 4.0)}
        if mode == "one_loop":
            params["b2"] = rng.uniform(-5, 5)
            params["bs2"] = rng.uniform(-5, 5)

        # Random linear params
        alpha = {}
        for name in lin_names:
            if name.startswith("c"):
                alpha[name] = rng.uniform(-20, 20)
            elif name.startswith("s"):
                alpha[name] = rng.uniform(-500, 500)
            elif name == "b3nl":
                alpha[name] = rng.uniform(-5, 5)

        # Full prediction with all params
        full_params = dict(params)
        full_params.update(alpha)
        pred_full = emu.predict(full_params)

        # Decomposed prediction
        m, T = emu.predict_decomposed(params)
        alpha_vec = np.array([alpha[name] for name in lin_names])
        pred_decomp = m + T @ alpha_vec

        np.testing.assert_allclose(pred_decomp, pred_full, rtol=1e-10,
                                   err_msg=f"Decomposition identity failed for mode={mode}")


# --------------------------------------------------------------------------
# Test 3: Decomposition identity in real space
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["eft", "one_loop"])
def test_decomposition_identity_real_space(cosmo, k, mode):
    rng = np.random.default_rng(123)
    emu = GalaxyTemplateEmulator(cosmo, k, ells=(0, 2, 4), z=0.5, space="real", mode=mode)
    lin_names = emu.linear_param_names

    params = {"b1": 2.0}
    if mode == "one_loop":
        params["b2"] = 0.5
        params["bs2"] = -0.5

    alpha = {}
    for name in lin_names:
        if name.startswith("c"):
            alpha[name] = rng.uniform(-20, 20)
        elif name.startswith("s"):
            alpha[name] = rng.uniform(-500, 500)
        elif name == "b3nl":
            alpha[name] = rng.uniform(-5, 5)

    full_params = dict(params)
    full_params.update(alpha)
    pred_full = emu.predict(full_params)

    m, T = emu.predict_decomposed(params)
    alpha_vec = np.array([alpha[name] for name in lin_names])
    pred_decomp = m + T @ alpha_vec

    np.testing.assert_allclose(pred_decomp, pred_full, rtol=1e-10)


# --------------------------------------------------------------------------
# Test 4: Parameter recovery from synthetic data
# --------------------------------------------------------------------------
def test_parameter_recovery(cosmo, k):
    """Generate synthetic data with known params, check bestfit_linear_params recovers truth."""
    mode = "one_loop"
    ells = (0, 2)
    emu = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, space="redshift", mode=mode)
    lin_names = emu.linear_param_names

    true_params = {
        "b1": 2.0, "b2": 0.5, "bs2": -0.5,
        "c0": 5.0, "c2": 2.0, "c4": 0.0,
        "s0": 100.0, "s2": 0.0, "b3nl": 0.1,
    }
    data = emu.predict(true_params)
    true_alpha = np.array([true_params[name] for name in lin_names])

    # Small noise
    ndata = len(data)
    noise_level = 0.001 * np.abs(data).mean()
    cov = np.eye(ndata) * noise_level ** 2
    precision = np.eye(ndata) / noise_level ** 2

    prior_sigmas = np.array([100.0, 100.0, 100.0, 5000.0, 500.0, 20.0])
    marg_like = MarginalizedLikelihood(data, precision, prior_sigmas)

    nl_params = {"b1": true_params["b1"], "b2": true_params["b2"], "bs2": true_params["bs2"]}
    m, T = emu.predict_decomposed(nl_params)
    recovered = marg_like.bestfit_linear_params(m, T)

    # c0 and s0 are well-constrained; c2/c4/s2/b3nl can be degenerate so use looser tol
    np.testing.assert_allclose(recovered, true_alpha, atol=1.0,
                               err_msg="Failed to recover linear params from noiseless data")
    # Check the well-constrained ones more tightly
    for j, name in enumerate(lin_names):
        if name in ("c0", "s0", "b3nl"):
            np.testing.assert_allclose(recovered[j], true_alpha[j], atol=0.15,
                                       err_msg=f"Failed to recover {name}")


# --------------------------------------------------------------------------
# Test 5: Marginalized likelihood vs brute-force for eft_lite (1D case)
# --------------------------------------------------------------------------
def test_marginalized_vs_bruteforce(cosmo, k):
    """For eft_lite (only c0 is linear), verify against numerical integration."""
    from scipy.integrate import quad

    mode = "eft_ct"
    ells = (0, 2)
    emu = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=0.5, space="redshift", mode=mode)

    true_params = {"b1": 2.0, "c0": 5.0}
    data = emu.predict(true_params)

    ndata = len(data)
    # Use large noise to avoid numerical underflow in brute-force integral
    noise = 5000.0
    precision = np.eye(ndata) / noise ** 2
    prior_sigma = np.array([100.0])

    marg_like = MarginalizedLikelihood(data, precision, prior_sigma)

    # Compare difference in marginalized log-L between two b1 values
    # (this cancels the normalization constant)
    m1, T1 = emu.predict_decomposed({"b1": 2.0})
    m2, T2 = emu.predict_decomposed({"b1": 1.8})
    log_L_marg1 = marg_like(m1, T1)
    log_L_marg2 = marg_like(m2, T2)

    # Brute-force: integrate over c0 numerically using log-sum-exp trick
    def log_integrand(c0, m, T):
        pred = m + T.ravel() * c0
        diff = data - pred
        chi2 = diff @ precision @ diff
        log_prior = -0.5 * (c0 / 100.0) ** 2
        return -0.5 * chi2 + log_prior

    # Use a grid-based log-sum-exp integration
    c0_grid = np.linspace(-500, 500, 10000)
    dc0 = c0_grid[1] - c0_grid[0]

    log_vals1 = np.array([log_integrand(c0, m1, T1) for c0 in c0_grid])
    max1 = log_vals1.max()
    log_integral1 = max1 + np.log(np.sum(np.exp(log_vals1 - max1)) * dc0)

    log_vals2 = np.array([log_integrand(c0, m2, T2) for c0 in c0_grid])
    max2 = log_vals2.max()
    log_integral2 = max2 + np.log(np.sum(np.exp(log_vals2 - max2)) * dc0)

    diff_marg = log_L_marg1 - log_L_marg2
    diff_brute = log_integral1 - log_integral2

    np.testing.assert_allclose(diff_marg, diff_brute, atol=0.01,
                               err_msg="Marginalized likelihood disagrees with brute-force integration")


# --------------------------------------------------------------------------
# Test 6: tree_only mode has no linear params (decomposition is trivial)
# --------------------------------------------------------------------------
def test_tree_only_decomposition(cosmo, k):
    emu = GalaxyTemplateEmulator(cosmo, k, ells=(0, 2), z=0.5, space="redshift", mode="tree")
    m, T = emu.predict_decomposed({"b1": 2.0})
    assert T.shape[1] == 0
    pred = emu.predict({"b1": 2.0})
    np.testing.assert_allclose(m, pred, rtol=1e-12)
