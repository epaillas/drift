"""Tests for joint P_gg + DS×g inference helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.analytic_marginalization import MarginalizedLikelihood
from drift.cosmology import get_cosmology
from drift.io import diagonal_covariance
from drift.synthetic import make_synthetic_dsg, make_synthetic_pgg
from scripts.inference_dsg import (
    DS_MODEL,
    ELLS as DSG_ELLS,
    KERNEL,
    QUANTILES,
    R,
    SPACE,
    Z,
    _build_data_mask as _build_dsg_data_mask,
    _build_params as _build_dsg_params,
    _build_params_marginalized as _build_dsg_params_marginalized,
)
from scripts.inference_joint import (
    MODEL_MODE,
    PGG_ELLS,
    _assemble_joint_templates,
    _build_joint_linear_param_names,
    _build_joint_params,
    _build_joint_params_marginalized,
    _build_joint_truth,
    _build_joint_truth_lookup,
    _build_pgg_mask,
    _combine_block_covariances,
    _joint_linear_truth,
    make_joint_theory_model,
    make_joint_theory_model_marginalized,
)


@pytest.fixture
def cosmo():
    return get_cosmology()


@pytest.fixture
def k():
    return np.linspace(0.01, 0.3, 30)


def _pgg_param_names(model_mode, use_marg):
    names, *_ = _build_joint_params_marginalized(model_mode, (), DS_MODEL) if use_marg else _build_joint_params(model_mode, (), DS_MODEL)
    return [
        name for name in names
        if not name.startswith("bq1_")
        and not name.startswith("beta_q_")
        and not name.startswith("bq_nabla2_")
    ]


def _dsg_param_names(model_mode, use_marg):
    if use_marg:
        return _build_dsg_params_marginalized(DS_MODEL, model_mode, QUANTILES)[0]
    return _build_dsg_params(DS_MODEL, model_mode, QUANTILES)[0]


def test_joint_params_include_shared_and_dsg_specific_terms():
    param_names, _ = _build_joint_params("one_loop", QUANTILES, DS_MODEL)

    assert "b1" in param_names
    assert "b2" in param_names
    assert "bs2" in param_names
    assert "b3nl" in param_names
    assert "c0" in param_names
    assert "c2" in param_names
    assert "c4" in param_names
    assert "s0" in param_names
    assert "s2" in param_names
    assert "bq1_1" in param_names
    assert "bq1_5" in param_names
    assert "bq_nabla2_1" in param_names
    assert "bq_nabla2_5" in param_names


def test_joint_tree_synthetic_truth_reproduces_masked_theory(cosmo, k):
    mode = "tree"
    truth = _build_joint_truth(mode, DS_MODEL, QUANTILES)
    truth_lookup = _build_joint_truth_lookup(truth, QUANTILES, DS_MODEL, mode)
    param_names, _ = _build_joint_params(mode, QUANTILES, DS_MODEL)
    theta = np.asarray([truth_lookup[name] for name in param_names], dtype=float)

    theory_fn = make_joint_theory_model(
        cosmo, k, k, mode, mode, DS_MODEL,
        _pgg_param_names(mode, use_marg=False),
        _dsg_param_names(mode, use_marg=False),
    )
    data_pgg, _ = make_synthetic_pgg(
        k, ells=PGG_ELLS, z=Z, space=SPACE, mode=mode, true_params=truth, cosmo=cosmo,
    )
    data_dsg, _ = make_synthetic_dsg(
        k, ells=DSG_ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
        ds_model=DS_MODEL, mode=mode, true_params=truth, cosmo=cosmo,
    )

    kmax_dict = {ell: 0.2 for ell in sorted(set(PGG_ELLS) | set(DSG_ELLS))}
    pgg_mask = _build_pgg_mask(k, PGG_ELLS, kmax_dict, kmin=0.0)
    dsg_mask = _build_dsg_data_mask(k, DSG_ELLS, QUANTILES, kmax_dict, kmin=0.0)
    expected = np.concatenate([data_pgg[pgg_mask], data_dsg[dsg_mask]])
    predicted = theory_fn(theta)
    masked = np.concatenate([predicted[:len(data_pgg)][pgg_mask], predicted[len(data_pgg):][dsg_mask]])

    np.testing.assert_allclose(masked, expected, rtol=1e-10)


def test_joint_marginalized_eft_ct_recovers_linear_truth(cosmo, k):
    mode = "eft_ct"
    truth = _build_joint_truth(mode, DS_MODEL, QUANTILES)
    truth_lookup = _build_joint_truth_lookup(truth, QUANTILES, DS_MODEL, mode)
    param_names, _ = _build_joint_params_marginalized(mode, QUANTILES, DS_MODEL)
    theta_nl = np.asarray([truth_lookup[name] for name in param_names], dtype=float)

    decomposed_fn = make_joint_theory_model_marginalized(
        cosmo, k, k, mode, DS_MODEL,
        _pgg_param_names(mode, use_marg=True),
        _dsg_param_names(mode, use_marg=True),
        _build_joint_linear_param_names(mode, QUANTILES),
    )
    data_pgg, _ = make_synthetic_pgg(
        k, ells=PGG_ELLS, z=Z, space=SPACE, mode=mode, true_params=truth, cosmo=cosmo,
    )
    data_dsg, _ = make_synthetic_dsg(
        k, ells=DSG_ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
        ds_model=DS_MODEL, mode=mode, true_params=truth, cosmo=cosmo,
    )

    kmax_dict = {ell: 0.2 for ell in sorted(set(PGG_ELLS) | set(DSG_ELLS))}
    pgg_mask = _build_pgg_mask(k, PGG_ELLS, kmax_dict, kmin=0.0)
    dsg_mask = _build_dsg_data_mask(k, DSG_ELLS, QUANTILES, kmax_dict, kmin=0.0)
    data_y = np.concatenate([data_pgg[pgg_mask], data_dsg[dsg_mask]])

    m, T = decomposed_fn(theta_nl)
    n_pgg = len(data_pgg)
    m = np.concatenate([m[:n_pgg][pgg_mask], m[n_pgg:][dsg_mask]])
    T = np.vstack([T[:n_pgg][pgg_mask], T[n_pgg:][dsg_mask]])

    linear_param_names = _build_joint_linear_param_names(mode, QUANTILES)
    alpha_truth = _joint_linear_truth(linear_param_names, truth_lookup)

    np.testing.assert_allclose(m + T @ alpha_truth, data_y, rtol=1e-10)

    _, precision = diagonal_covariance(data_y, rescale=64.0)
    prior_sigmas = np.array([50.0] + [4.0] * len(QUANTILES))
    marg_like = MarginalizedLikelihood(data_y, precision, prior_sigmas)
    recovered = marg_like.bestfit_linear_params(m, T)

    np.testing.assert_allclose(recovered, alpha_truth, atol=1e-3)


def test_block_covariance_assembly_shapes_and_zero_off_diagonal():
    cov_a = np.array([[2.0, 0.5], [0.5, 1.0]])
    prec_a = np.linalg.inv(cov_a)
    cov_b = np.array([[3.0]])
    prec_b = np.linalg.inv(cov_b)

    cov, precision = _combine_block_covariances(cov_a, prec_a, cov_b, prec_b)

    assert cov.shape == (3, 3)
    assert precision.shape == (3, 3)
    np.testing.assert_allclose(cov[:2, :2], cov_a)
    np.testing.assert_allclose(cov[2:, 2:], cov_b)
    np.testing.assert_allclose(cov[:2, 2:], 0.0)
    np.testing.assert_allclose(cov[2:, :2], 0.0)
    np.testing.assert_allclose(precision[:2, 2:], 0.0)
    np.testing.assert_allclose(precision[2:, :2], 0.0)
