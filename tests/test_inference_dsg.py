"""Tests for the synthetic DSG inference helpers and consistency checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.analytic_marginalization import MarginalizedLikelihood
from drift.utils.cosmology import get_cosmology
from drift.io import build_diagonal_covariance
from drift.synthetic import make_synthetic_dsg
from scripts.inference_dsg import (
    DS_MODEL,
    ELLS,
    KERNEL,
    QUANTILES,
    R,
    SPACE,
    Z,
    _build_data_mask,
    _build_params,
    _build_params_marginalized,
    _build_synthetic_truth,
    _build_truth_lookup,
    _build_truth_theta,
    _synthetic_linear_truth,
    _validate_synthetic_truth,
    make_eft_theory_model,
    make_eft_theory_model_marginalized,
)


@pytest.fixture
def cosmo():
    return get_cosmology()


@pytest.fixture
def k():
    return np.linspace(0.01, 0.3, 30)


def test_synthetic_truth_matches_quantiles():
    truth = _build_synthetic_truth(QUANTILES, DS_MODEL, "eft_ct")
    assert len(truth["bq1"]) == len(QUANTILES)
    assert len(truth["bq_nabla2"]) == len(QUANTILES)


def test_validate_synthetic_truth_raises_for_length_mismatch():
    bad_truth = {"b1": 2.0, "bq1": [0.5, 1.0, -1.0]}
    with pytest.raises(ValueError, match="expected 2"):
        _validate_synthetic_truth(bad_truth, QUANTILES, DS_MODEL, "tree")


def test_synthetic_vector_length_matches_mask(cosmo, k):
    truth = _build_synthetic_truth(QUANTILES, DS_MODEL, "tree")
    data_y, _ = make_synthetic_dsg(
        k, ells=ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
        ds_model=DS_MODEL, mode="tree", true_params=truth, cosmo=cosmo,
    )
    mask = _build_data_mask(k, ELLS, QUANTILES, {ell: 0.2 for ell in ELLS}, kmin=0.0)
    assert len(data_y) == len(mask)
    assert data_y[mask].shape[0] == mask.sum()


def test_tree_synthetic_truth_reproduces_masked_theory(cosmo, k):
    truth = _build_synthetic_truth(QUANTILES, DS_MODEL, "tree")
    truth_lookup = _build_truth_lookup(truth, QUANTILES, DS_MODEL, "tree")
    param_names, bounds, _ = _build_params(DS_MODEL, "tree", QUANTILES)
    theta = _build_truth_theta(param_names, truth_lookup, fixed_cosmo={})

    theory_fn = make_eft_theory_model(
        cosmo, k, ells=ELLS, quantiles=QUANTILES, ds_model=DS_MODEL, mode="tree",
    )
    data_y, _ = make_synthetic_dsg(
        k, ells=ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
        ds_model=DS_MODEL, mode="tree", true_params=truth, cosmo=cosmo,
    )
    mask = _build_data_mask(k, ELLS, QUANTILES, {ell: 0.2 for ell in ELLS}, kmin=0.0)

    np.testing.assert_allclose(theory_fn(theta)[mask], data_y[mask], rtol=1e-10)


def test_marginalized_eft_ct_recovers_linear_truth(cosmo, k):
    mode = "eft_ct"
    truth = _build_synthetic_truth(QUANTILES, DS_MODEL, mode)
    truth_lookup = _build_truth_lookup(truth, QUANTILES, DS_MODEL, mode)
    param_names, bounds = _build_params_marginalized(DS_MODEL, mode, QUANTILES)
    theta_nl = _build_truth_theta(param_names, truth_lookup, fixed_cosmo={})

    decomposed_fn = make_eft_theory_model_marginalized(
        cosmo, k, ells=ELLS, quantiles=QUANTILES, ds_model=DS_MODEL, mode=mode,
    )
    data_y, _ = make_synthetic_dsg(
        k, ells=ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
        ds_model=DS_MODEL, mode=mode, true_params=truth, cosmo=cosmo,
    )

    mask = _build_data_mask(k, ELLS, QUANTILES, {ell: 0.2 for ell in ELLS}, kmin=0.0)
    data_y = data_y[mask]
    m, T = decomposed_fn(theta_nl)
    m = m[mask]
    T = T[mask]

    linear_param_names = ["c0"] + [f"bq_nabla2_{q}" for q in QUANTILES]
    alpha_truth = _synthetic_linear_truth(linear_param_names, truth_lookup, QUANTILES)

    np.testing.assert_allclose(m + T @ alpha_truth, data_y, rtol=1e-10)

    _, precision = build_diagonal_covariance(data_y, rescale=64.0)
    prior_sigmas = np.array([50.0] + [4.0] * len(QUANTILES))
    marg_like = MarginalizedLikelihood(data_y, precision, prior_sigmas)
    recovered = marg_like.bestfit_linear_params(m, T)

    np.testing.assert_allclose(recovered, alpha_truth, atol=1e-3)
