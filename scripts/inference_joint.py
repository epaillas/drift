"""Joint P_gg + DS×g parameter inference with PocoMC.

Fits the galaxy auto-power spectrum multipoles and density-split × galaxy
multipoles simultaneously using a shared cosmology and galaxy EFT parameter set.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from pocomc import Prior, Sampler
from scipy.stats import uniform as sp_uniform

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from drift.analytic_marginalization import MarginalizedLikelihood
from drift.utils.cosmology import (
    ALL_COSMO_NAMES,
    DEFAULT_COSMO_RANGES,
    LinearPowerGrid,
    OneLoopPowerGrid,
    _DEFAULT_PARAMS,
    get_cosmology,
)
from drift.emulators.galaxy import GalaxyTemplateEmulator
from drift.io import diagonal_covariance, load_pgg_measurements, load_measurements, mock_covariance, taylor_cache_key
from drift.synthetic import make_synthetic_dsg, make_synthetic_pgg
from scripts.inference_dsg import (
    DS_MODEL,
    ELLS as DSG_ELLS,
    KERNEL,
    MEAS_PATH as DSG_MEAS_PATH,
    QUANTILES,
    R,
    SPACE,
    Z,
    _build_data_mask as _build_dsg_data_mask,
    _build_synthetic_truth as _build_dsg_synthetic_truth,
    _build_truth_lookup as _build_dsg_truth_lookup,
    _build_truth_theta as _build_truth_theta,
    _parse_kmax,
    _validate_synthetic_truth,
    make_direct_theory_model as make_dsg_direct_theory_model,
    make_eft_theory_model as make_dsg_theory_model,
    make_eft_theory_model_marginalized as make_dsg_theory_model_marginalized,
    make_log_likelihood,
    resolve_cosmo_params,
)
from scripts.inference_pgg import (
    COV_DIR as PGG_COV_DIR,
    ELLS as PGG_ELLS,
    MEAS_PATH as PGG_MEAS_PATH,
    _resolve_pgg_covariance,
    make_direct_theory_model as make_pgg_direct_theory_model,
    make_eft_theory_model as make_pgg_theory_model,
    make_eft_theory_model_marginalized as make_pgg_theory_model_marginalized,
)


MODEL_MODE = "one_loop"
OUTPUT_ROOT = Path(__file__).parents[1] / "outputs" / "inference_joint"
VARY_COSMO = False


def _shared_linear_param_names(model_mode):
    return GalaxyTemplateEmulator(
        get_cosmology(),
        np.linspace(0.01, 0.3, 4),
        ells=PGG_ELLS,
        z=Z,
        space=SPACE,
        mode=model_mode,
    ).linear_param_names


def _build_joint_params(model_mode, quantiles, ds_model,
                        free_cosmo_names=None, cosmo_bounds=None):
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"] + [f"bq1_{q}" for q in quantiles]
    bounds_list = (list(cosmo_bounds) if len(cosmo_bounds) else []) + [[0.5, 4.0]] + [[-4.0, 4.0]] * len(quantiles)
    bounds = np.array(bounds_list, dtype=float)

    if model_mode != "tree":
        param_names.append("sigma_fog")
        bounds = np.vstack([bounds, [[0.0, 30.0]]])

    shared_linear = []
    if model_mode in ("eft_ct", "eft", "one_loop"):
        shared_linear.append(("c0", [-50.0, 50.0]))
    if model_mode in ("eft", "one_loop"):
        shared_linear.append(("s0", [-5000.0, 5000.0]))
    if model_mode == "one_loop":
        shared_linear.extend([
            ("c2", [-50.0, 50.0]),
            ("c4", [-50.0, 50.0]),
            ("s2", [-500.0, 500.0]),
        ])
    for name, bound in shared_linear:
        param_names.append(name)
        bounds = np.vstack([bounds, [bound]])

    if model_mode == "one_loop":
        param_names += ["b2", "bs2"]
        bounds = np.vstack([bounds, [[-10.0, 10.0], [-10.0, 10.0]]])
        param_names += ["b3nl"]
        bounds = np.vstack([bounds, [[-10.0, 10.0]]])

    if ds_model == "phenomenological":
        param_names += [f"beta_q_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-2.0, 2.0] for _ in quantiles]])

    if model_mode in ("eft_ct", "eft", "one_loop"):
        param_names += [f"bq_nabla2_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-4.0, 4.0] for _ in quantiles]])

    return param_names, bounds


def _build_joint_params_marginalized(model_mode, quantiles, ds_model,
                                     free_cosmo_names=None, cosmo_bounds=None):
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"] + [f"bq1_{q}" for q in quantiles]
    bounds_list = (list(cosmo_bounds) if len(cosmo_bounds) else []) + [[0.5, 4.0]] + [[-4.0, 4.0]] * len(quantiles)
    bounds = np.array(bounds_list, dtype=float)

    if model_mode != "tree":
        param_names.append("sigma_fog")
        bounds = np.vstack([bounds, [[0.0, 30.0]]])

    if model_mode == "one_loop":
        param_names += ["b2", "bs2"]
        bounds = np.vstack([bounds, [[-10.0, 10.0], [-10.0, 10.0]]])

    if ds_model == "phenomenological":
        param_names += [f"beta_q_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-2.0, 2.0] for _ in quantiles]])

    return param_names, bounds


def _unpack_joint_theta(theta, quantiles, model_mode, ds_model,
                        free_cosmo_names=None, fixed_cosmo=None):
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}

    idx = 0
    cosmo_vals = {}
    for name in free_cosmo_names:
        cosmo_vals[name] = float(theta[idx])
        idx += 1
    cosmo_vals.update(fixed_cosmo)

    b1 = float(theta[idx])
    idx += 1
    bq1_vals = [float(x) for x in theta[idx:idx + len(quantiles)]]
    idx += len(quantiles)

    sigma_fog = 0.0
    if model_mode != "tree":
        sigma_fog = float(theta[idx])
        idx += 1

    c0 = c2 = c4 = s0 = s2 = b2 = bs2 = b3nl = 0.0
    if model_mode in ("eft_ct", "eft", "one_loop"):
        c0 = float(theta[idx])
        idx += 1
    if model_mode in ("eft", "one_loop"):
        s0 = float(theta[idx])
        idx += 1
    if model_mode == "one_loop":
        c2 = float(theta[idx])
        idx += 1
        c4 = float(theta[idx])
        idx += 1
        s2 = float(theta[idx])
        idx += 1

    if model_mode == "one_loop":
        b2 = float(theta[idx])
        idx += 1
        bs2 = float(theta[idx])
        idx += 1
        b3nl = float(theta[idx])
        idx += 1

    if ds_model == "phenomenological":
        beta_q_vals = [float(x) for x in theta[idx:idx + len(quantiles)]]
        idx += len(quantiles)
    else:
        beta_q_vals = [0.0] * len(quantiles)

    if model_mode in ("eft_ct", "eft", "one_loop"):
        bq_nabla2_vals = [float(x) for x in theta[idx:idx + len(quantiles)]]
        idx += len(quantiles)
    else:
        bq_nabla2_vals = [0.0] * len(quantiles)

    if idx != len(theta):
        raise ValueError(f"Theta length mismatch: consumed {idx}, received {len(theta)}.")

    return {
        "cosmo": cosmo_vals,
        "b1": b1,
        "bq1_vals": bq1_vals,
        "sigma_fog": sigma_fog,
        "c0": c0,
        "c2": c2,
        "c4": c4,
        "s0": s0,
        "s2": s2,
        "b2": b2,
        "bs2": bs2,
        "b3nl": b3nl,
        "beta_q_vals": beta_q_vals,
        "bq_nabla2_vals": bq_nabla2_vals,
    }


def _unpack_joint_theta_marginalized(theta, quantiles, model_mode, ds_model,
                                     free_cosmo_names=None, fixed_cosmo=None):
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}

    idx = 0
    cosmo_vals = {}
    for name in free_cosmo_names:
        cosmo_vals[name] = float(theta[idx])
        idx += 1
    cosmo_vals.update(fixed_cosmo)

    b1 = float(theta[idx])
    idx += 1
    bq1_vals = [float(x) for x in theta[idx:idx + len(quantiles)]]
    idx += len(quantiles)

    sigma_fog = 0.0
    if model_mode != "tree":
        sigma_fog = float(theta[idx])
        idx += 1

    b2 = bs2 = 0.0
    if model_mode == "one_loop":
        b2 = float(theta[idx])
        idx += 1
        bs2 = float(theta[idx])
        idx += 1

    if ds_model == "phenomenological":
        beta_q_vals = [float(x) for x in theta[idx:idx + len(quantiles)]]
        idx += len(quantiles)
    else:
        beta_q_vals = [0.0] * len(quantiles)

    if idx != len(theta):
        raise ValueError(f"Marginalized theta length mismatch: consumed {idx}, received {len(theta)}.")

    return {
        "cosmo": cosmo_vals,
        "b1": b1,
        "bq1_vals": bq1_vals,
        "sigma_fog": sigma_fog,
        "c0": 0.0,
        "c2": 0.0,
        "c4": 0.0,
        "s0": 0.0,
        "s2": 0.0,
        "b2": b2,
        "bs2": bs2,
        "b3nl": 0.0,
        "beta_q_vals": beta_q_vals,
        "bq_nabla2_vals": [0.0] * len(quantiles),
    }


def _joint_to_pgg_theta(theta, quantiles, model_mode, ds_model,
                        pgg_param_names, free_cosmo_names=None, fixed_cosmo=None,
                        marginalized=False):
    unpack = _unpack_joint_theta_marginalized if marginalized else _unpack_joint_theta
    joint = unpack(theta, quantiles, model_mode, ds_model,
                   free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo)
    lookup = dict(joint["cosmo"])
    lookup.update({
        "b1": joint["b1"],
        "sigma_fog": joint["sigma_fog"],
        "c0": joint["c0"],
        "c2": joint["c2"],
        "c4": joint["c4"],
        "s0": joint["s0"],
        "s2": joint["s2"],
        "b2": joint["b2"],
        "bs2": joint["bs2"],
        "b3nl": joint["b3nl"],
    })
    return np.asarray([lookup[name] for name in pgg_param_names], dtype=float)


def _joint_to_dsg_theta(theta, quantiles, model_mode, ds_model,
                        dsg_param_names, free_cosmo_names=None, fixed_cosmo=None,
                        marginalized=False):
    unpack = _unpack_joint_theta_marginalized if marginalized else _unpack_joint_theta
    joint = unpack(theta, quantiles, model_mode, ds_model,
                   free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo)
    lookup = dict(joint["cosmo"])
    lookup["b1"] = joint["b1"]
    lookup["sigma_fog"] = joint["sigma_fog"]
    lookup["c0"] = joint["c0"]
    lookup["c2"] = joint["c2"]
    lookup["c4"] = joint["c4"]
    lookup["s0"] = joint["s0"]
    lookup["s2"] = joint["s2"]
    lookup["b2"] = joint["b2"]
    lookup["bs2"] = joint["bs2"]
    lookup["b3nl"] = joint["b3nl"]
    for i, q in enumerate(quantiles):
        lookup[f"bq1_{q}"] = joint["bq1_vals"][i]
        lookup[f"beta_q_{q}"] = joint["beta_q_vals"][i]
        lookup[f"bq_nabla2_{q}"] = joint["bq_nabla2_vals"][i]
    return np.asarray([lookup[name] for name in dsg_param_names], dtype=float)


def _build_joint_truth(model_mode, ds_model, quantiles):
    truth = _build_dsg_synthetic_truth(quantiles, ds_model, model_mode)
    if "sigma_fog" not in truth:
        truth["sigma_fog"] = 0.0
    return truth


def _build_joint_truth_lookup(true_params, quantiles, ds_model, model_mode, cosmo_params=None):
    truth = _build_dsg_truth_lookup(
        true_params, quantiles, ds_model, model_mode, cosmo_params=cosmo_params,
    )
    for name in ("sigma_fog", "c0", "c2", "c4", "s0", "s2", "b2", "bs2", "b3nl"):
        truth.setdefault(name, float(true_params.get(name, 0.0)))
    return truth


def _joint_linear_truth(linear_param_names, truth_lookup):
    alpha = []
    for name in linear_param_names:
        if name.startswith("bq_nabla2_"):
            alpha.append(float(truth_lookup[name]))
        else:
            alpha.append(float(truth_lookup.get(name, 0.0)))
    return np.asarray(alpha, dtype=float)


def _combine_block_covariances(cov_a, precision_a, cov_b, precision_b):
    n_a = cov_a.shape[0]
    n_b = cov_b.shape[0]
    cov = np.zeros((n_a + n_b, n_a + n_b))
    precision = np.zeros_like(cov)
    cov[:n_a, :n_a] = cov_a
    cov[n_a:, n_a:] = cov_b
    precision[:n_a, :n_a] = precision_a
    precision[n_a:, n_a:] = precision_b
    return cov, precision


def _build_joint_linear_param_names(model_mode, quantiles):
    names = list(_shared_linear_param_names(model_mode))
    if model_mode != "tree":
        names += [f"bq_nabla2_{q}" for q in quantiles]
    return names


def _assemble_joint_templates(m_pgg, T_pgg, pgg_linear_names,
                              m_dsg, T_dsg, dsg_linear_names,
                              joint_linear_names):
    n_pgg = m_pgg.shape[0]
    n_dsg = m_dsg.shape[0]
    n_lin = len(joint_linear_names)
    m = np.concatenate([m_pgg, m_dsg])
    T = np.zeros((n_pgg + n_dsg, n_lin))
    name_to_col = {name: i for i, name in enumerate(joint_linear_names)}

    for j, name in enumerate(pgg_linear_names):
        T[:n_pgg, name_to_col[name]] = T_pgg[:, j]
    for j, name in enumerate(dsg_linear_names):
        T[n_pgg:, name_to_col[name]] = T_dsg[:, j]
    return m, T


def _build_pgg_mask(k, ells, kmax_dict, kmin=0.0):
    return np.concatenate([((k >= kmin) & (k <= kmax_dict.get(ell, np.inf))) for ell in ells])


def make_joint_theory_model(cosmo, k_pgg, k_dsg, pgg_mode, dsg_mode, ds_model,
                            pgg_param_names, dsg_param_names, free_cosmo_names=None,
                            fixed_cosmo=None, no_emulator=False,
                            pgg_cosmo_grid=None, dsg_cosmo_grid=None):
    pgg_builder = make_pgg_direct_theory_model if no_emulator else make_pgg_theory_model
    dsg_builder = make_dsg_direct_theory_model if no_emulator else make_dsg_theory_model

    if no_emulator:
        pgg_theory = pgg_builder(
            cosmo, k_pgg, ells=PGG_ELLS, mode=pgg_mode,
            free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
        )
        dsg_theory = dsg_builder(
            cosmo, k_dsg, ells=DSG_ELLS, quantiles=QUANTILES,
            ds_model=ds_model, mode=dsg_mode,
            free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
        )
    else:
        pgg_theory = pgg_builder(
            cosmo, k_pgg, ells=PGG_ELLS, mode=pgg_mode,
            free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            cosmo_grid=pgg_cosmo_grid,
        )
        dsg_theory = dsg_builder(
            cosmo, k_dsg, ells=DSG_ELLS, quantiles=QUANTILES,
            ds_model=ds_model, mode=dsg_mode,
            free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            cosmo_grid=dsg_cosmo_grid,
        )

    def theory(theta):
        pgg_theta = _joint_to_pgg_theta(
            theta, QUANTILES, pgg_mode, ds_model,
            pgg_param_names, free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
        )
        dsg_theta = _joint_to_dsg_theta(
            theta, QUANTILES, dsg_mode, ds_model,
            dsg_param_names, free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
        )
        return np.concatenate([pgg_theory(pgg_theta), dsg_theory(dsg_theta)])

    return theory


def make_joint_theory_model_marginalized(cosmo, k_pgg, k_dsg, model_mode, ds_model,
                                         pgg_param_names, dsg_param_names,
                                         joint_linear_names, free_cosmo_names=None,
                                         fixed_cosmo=None, pgg_cosmo_grid=None,
                                         dsg_cosmo_grid=None):
    pgg_decomp = make_pgg_theory_model_marginalized(
        cosmo, k_pgg, ells=PGG_ELLS, mode=model_mode, cosmo_grid=pgg_cosmo_grid,
        free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
    )
    dsg_decomp = make_dsg_theory_model_marginalized(
        cosmo, k_dsg, ells=DSG_ELLS, quantiles=QUANTILES,
        ds_model=ds_model, mode=model_mode, cosmo_grid=dsg_cosmo_grid,
        free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
    )
    pgg_linear_names = list(_shared_linear_param_names(model_mode))
    dsg_linear_names = pgg_linear_names + ([f"bq_nabla2_{q}" for q in QUANTILES] if model_mode != "tree" else [])

    def decomposed(theta):
        pgg_theta = _joint_to_pgg_theta(
            theta, QUANTILES, model_mode, ds_model,
            pgg_param_names, free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            marginalized=True,
        )
        dsg_theta = _joint_to_dsg_theta(
            theta, QUANTILES, model_mode, ds_model,
            dsg_param_names, free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            marginalized=True,
        )
        m_pgg, T_pgg = pgg_decomp(pgg_theta)
        m_dsg, T_dsg = dsg_decomp(dsg_theta)
        return _assemble_joint_templates(
            m_pgg, T_pgg, pgg_linear_names,
            m_dsg, T_dsg, dsg_linear_names,
            joint_linear_names,
        )

    return decomposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-emulator", action="store_true",
                        help="Use direct GL quadrature instead of the analytic template emulator.")
    parser.add_argument("--vary-cosmo", action="store_true", default=VARY_COSMO,
                        help="Jointly infer cosmological parameters alongside bias parameters.")
    parser.add_argument("--fix-cosmo", nargs="+", default=None, metavar="PARAM[=VALUE]",
                        help="Fix cosmological parameters when --vary-cosmo is set.")
    parser.add_argument("--mode", choices=["tree", "eft_ct", "eft", "one_loop"],
                        default=MODEL_MODE, help=f"EFT model mode (default: {MODEL_MODE}).")
    parser.add_argument("--kmax", nargs="+", default=None, metavar="[ELL:]VALUE",
                        help="Maximum k to include. Single float or 'ell:value' pairs.")
    parser.add_argument("--kmin", type=float, default=0.01, metavar="VALUE",
                        help="Minimum k to include (default: 0.01).")
    parser.add_argument("--rebin", type=int, default=5, metavar="N",
                        help="Keep every Nth k-bin when loading measurements (default: 5).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic noiseless joint data vector instead of measurements.")
    parser.add_argument("--synthetic-mode", choices=["tree", "eft_ct", "eft", "one_loop"],
                        default=None, metavar="MODE",
                        help="Model mode for synthetic data generation (default: same as --mode).")
    parser.add_argument("--taylor", action="store_true",
                        help="Wrap theory model in a Taylor expansion emulator.")
    parser.add_argument("--taylor-order", type=int, default=4, metavar="N")
    parser.add_argument("--taylor-step", type=float, default=0.01, metavar="FRAC")
    parser.add_argument("--diag-cov", action="store_true",
                        help="Use diagonal covariance instead of per-statistic covariance.")
    parser.add_argument("--analytic-cov", action="store_true",
                        help="Use analytic cubic-box covariance for the P_gg block.")
    parser.add_argument("--analytic-cov-terms", type=str, default="gaussian", metavar="TERMS",
                        help="Analytic P_gg covariance terms: 'gaussian' or 'gaussian+effective_cng'.")
    parser.add_argument("--cov-rescale", type=float, default=64.0, metavar="FACTOR")
    parser.add_argument("--box-volume", type=float, default=None, metavar="V",
                        help="Box volume in (Mpc/h)^3 for analytic P_gg covariance.")
    parser.add_argument("--number-density", type=float, default=None, metavar="N",
                        help="Galaxy number density in (h/Mpc)^3 for analytic P_gg covariance.")
    parser.add_argument("--shot-noise", type=float, default=None, metavar="P0",
                        help="Constant shot-noise power in (Mpc/h)^3 for analytic P_gg covariance.")
    parser.add_argument("--cng-amplitude", type=float, default=0.0, metavar="A",
                        help="Amplitude of the effective connected P_gg covariance term.")
    parser.add_argument("--cng-coherence", type=float, default=0.35, metavar="SIGMA",
                        help="Log-k coherence length of the effective connected P_gg covariance term.")
    parser.add_argument("--analytic-marg", action="store_true",
                        help="Analytically marginalize over linear nuisance parameters.")
    parser.add_argument("--marg-prior-sigma", nargs="+", default=None, metavar="PARAM=VALUE",
                        help="Override prior sigmas for analytic marginalization.")
    args = parser.parse_args()

    model_mode = args.mode
    synthetic_mode = args.synthetic_mode or model_mode
    free_cosmo_names, fixed_cosmo, cosmo_bounds = resolve_cosmo_params(args.vary_cosmo, args.fix_cosmo)
    use_marg = args.analytic_marg

    if use_marg:
        param_names, bounds = _build_joint_params_marginalized(
            model_mode, QUANTILES, DS_MODEL,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )
        marg_sigma_overrides = {}
        if args.marg_prior_sigma:
            for tok in args.marg_prior_sigma:
                name, val = tok.split("=", 1)
                marg_sigma_overrides[name.strip()] = float(val)
    else:
        param_names, bounds = _build_joint_params(
            model_mode, QUANTILES, DS_MODEL,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )

    output_dir = OUTPUT_ROOT / SPACE / DS_MODEL / model_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vary_cosmo:
        print(f"Free cosmo params: {free_cosmo_names}")
        print(f"Fixed cosmo params: {fixed_cosmo}")

    if args.synthetic:
        if args.vary_cosmo and synthetic_mode == "tree":
            warnings.warn(
                "Synthetic joint truth recovery with mode='tree' and free cosmology is not expected "
                "to be identifiable. Use fixed cosmology or a higher-order model.",
                stacklevel=2,
            )
        true_params = _build_joint_truth(synthetic_mode, DS_MODEL, QUANTILES)
        _validate_synthetic_truth(true_params, QUANTILES, DS_MODEL, synthetic_mode)
        k_pgg = np.linspace(0.01, 0.3, 30)
        k_dsg = k_pgg.copy()
        cosmo = get_cosmology()
        print(f"Generating synthetic joint data vector (mode={synthetic_mode}) ...")
        print(f"  True parameters: {true_params}")
        data_pgg, _ = make_synthetic_pgg(
            k_pgg, ells=PGG_ELLS, z=Z, space=SPACE, mode=synthetic_mode,
            true_params=true_params, cosmo=cosmo,
        )
        data_dsg, _ = make_synthetic_dsg(
            k_dsg, ells=DSG_ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
            ds_model=DS_MODEL, mode=synthetic_mode, true_params=true_params, cosmo=cosmo,
        )
        truth_lookup = _build_joint_truth_lookup(
            true_params, QUANTILES, DS_MODEL, synthetic_mode,
            cosmo_params={name: _DEFAULT_PARAMS[name] for name in ALL_COSMO_NAMES},
        )
    else:
        print(f"Loading P_gg measurements from {PGG_MEAS_PATH} ...")
        k_pgg, poles_pgg = load_pgg_measurements(PGG_MEAS_PATH, ells=PGG_ELLS, rebin=args.rebin, kmin=args.kmin)
        data_pgg = np.concatenate([poles_pgg[ell] for ell in PGG_ELLS])
        print(f"Loading DS×g measurements from {DSG_MEAS_PATH} ...")
        k_dsg, multipoles_per_bin = load_measurements(
            DSG_MEAS_PATH, nquantiles=max(QUANTILES), ells=DSG_ELLS, rebin=args.rebin,
        )
        data_dsg = np.concatenate([
            multipoles_per_bin[f"DS{q}"][ell]
            for q in QUANTILES
            for ell in DSG_ELLS
        ])
        cosmo = get_cosmology()

    kmax_dict = _parse_kmax(args.kmax, tuple(sorted(set(PGG_ELLS) | set(DSG_ELLS))))
    pgg_mask = _build_pgg_mask(k_pgg, PGG_ELLS, kmax_dict, kmin=args.kmin)
    dsg_mask = _build_dsg_data_mask(k_dsg, DSG_ELLS, QUANTILES, kmax_dict, kmin=args.kmin)
    data_pgg_masked = data_pgg[pgg_mask]
    data_dsg_masked = data_dsg[dsg_mask]
    data_y_masked = np.concatenate([data_pgg_masked, data_dsg_masked])

    print(f"  P_gg masked data length: {len(data_pgg_masked)}")
    print(f"  DS×g masked data length: {len(data_dsg_masked)}")
    print(f"  Joint masked data length: {len(data_y_masked)}")

    pgg_param_names = _build_joint_params_marginalized(
        model_mode, (), DS_MODEL,
        free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
    )[0] if use_marg else _build_joint_params(
        model_mode, (), DS_MODEL,
        free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
    )[0]
    pgg_param_names = [name for name in pgg_param_names if not name.startswith("bq1_") and not name.startswith("beta_q_") and not name.startswith("bq_nabla2_")]

    from scripts.inference_dsg import _build_params as _build_dsg_params, _build_params_marginalized as _build_dsg_params_marginalized
    dsg_param_names = _build_dsg_params_marginalized(
        DS_MODEL, model_mode, QUANTILES,
        free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
    )[0] if use_marg else _build_dsg_params(
        DS_MODEL, model_mode, QUANTILES,
        free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
    )[0]

    joint_linear_names = _build_joint_linear_param_names(model_mode, QUANTILES)
    if use_marg:
        all_names, all_bounds = _build_joint_params(
            model_mode, QUANTILES, DS_MODEL,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )
        prior_sigmas = []
        for name in joint_linear_names:
            idx = all_names.index(name)
            lo, hi = all_bounds[idx]
            prior_sigmas.append(marg_sigma_overrides.get(name, (hi - lo) / 2.0))
        prior_sigmas = np.asarray(prior_sigmas, dtype=float)
        print(f"  Linear params (marginalized): {joint_linear_names}")

    pgg_cosmo_grid = None
    dsg_cosmo_grid = None
    if args.vary_cosmo and not args.no_emulator:
        cosmo_ranges = {name: DEFAULT_COSMO_RANGES[name] for name in free_cosmo_names}
        if model_mode == "one_loop":
            print(f"Precomputing OneLoopPowerGrid ({' x '.join(free_cosmo_names)}) ...")
            pgg_cosmo_grid = OneLoopPowerGrid(k_pgg, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)
            dsg_cosmo_grid = OneLoopPowerGrid(k_dsg, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)
        else:
            print(f"Precomputing LinearPowerGrid ({' x '.join(free_cosmo_names)}) ...")
            pgg_cosmo_grid = LinearPowerGrid(k_pgg, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)
            dsg_cosmo_grid = LinearPowerGrid(k_dsg, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo)
        print("  done.")

    if args.taylor:
        from drift.taylor import TaylorEmulator

        fiducial = {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(param_names, bounds)}
        cache_hash = taylor_cache_key(
            model_mode=model_mode, ds_model=DS_MODEL, space=SPACE, z=Z,
            pgg_ells=PGG_ELLS, dsg_ells=DSG_ELLS, quantiles=QUANTILES,
            kmax=str(kmax_dict), kmin=args.kmin, rebin=args.rebin,
            free_cosmo_names=str(free_cosmo_names), fixed_cosmo=str(sorted(fixed_cosmo.items())),
            taylor_order=args.taylor_order, taylor_step=args.taylor_step,
            fiducial=str(sorted(fiducial.items())), param_names=str(param_names),
            analytic_marg=use_marg,
        )
        cache_path = output_dir / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            if use_marg:
                base_decomp = make_joint_theory_model_marginalized(
                    cosmo, k_pgg, k_dsg, model_mode, DS_MODEL,
                    pgg_param_names, dsg_param_names, joint_linear_names,
                    free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
                    pgg_cosmo_grid=pgg_cosmo_grid, dsg_cosmo_grid=dsg_cosmo_grid,
                )

                def _dict_decomposed(params):
                    theta = np.asarray([params[name] for name in param_names], dtype=float)
                    m, T = base_decomp(theta)
                    m = np.concatenate([m[:len(data_pgg)][pgg_mask], m[len(data_pgg):][dsg_mask]])
                    T = np.vstack([T[:len(data_pgg)][pgg_mask], T[len(data_pgg):][dsg_mask]])
                    return np.concatenate([m, T.ravel()])
            else:
                base_theory = make_joint_theory_model(
                    cosmo, k_pgg, k_dsg, model_mode, model_mode, DS_MODEL,
                    pgg_param_names, dsg_param_names,
                    free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
                    no_emulator=True,
                )

                def _dict_decomposed(params):
                    theta = np.asarray([params[name] for name in param_names], dtype=float)
                    pred = base_theory(theta)
                    n_pgg = len(data_pgg)
                    return np.concatenate([pred[:n_pgg][pgg_mask], pred[n_pgg:][dsg_mask]])

            print(f"Building Taylor emulator (order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(_dict_decomposed, fiducial, order=args.taylor_order, step_sizes=args.taylor_step)
            taylor_emu.save_coefficients(cache_path)
            print(f"  Cached Taylor coefficients to {cache_path}")

        if use_marg:
            n_data = len(data_y_masked)
            n_lin = len(joint_linear_names)

            def decomposed_fn_masked(theta):
                out = taylor_emu.predict({name: theta[i] for i, name in enumerate(param_names)})
                return out[:n_data], out[n_data:].reshape(n_data, n_lin)
        else:
            theory_fn_masked = lambda theta: taylor_emu.predict({name: theta[i] for i, name in enumerate(param_names)})
    else:
        if use_marg:
            base_decomp = make_joint_theory_model_marginalized(
                cosmo, k_pgg, k_dsg, model_mode, DS_MODEL,
                pgg_param_names, dsg_param_names, joint_linear_names,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
                pgg_cosmo_grid=pgg_cosmo_grid, dsg_cosmo_grid=dsg_cosmo_grid,
            )

            def decomposed_fn_masked(theta):
                m, T = base_decomp(theta)
                n_pgg = len(data_pgg)
                m_pgg = m[:n_pgg][pgg_mask]
                m_dsg = m[n_pgg:][dsg_mask]
                T_pgg = T[:n_pgg][pgg_mask]
                T_dsg = T[n_pgg:][dsg_mask]
                return np.concatenate([m_pgg, m_dsg]), np.vstack([T_pgg, T_dsg])
        else:
            theory_fn = make_joint_theory_model(
                cosmo, k_pgg, k_dsg, model_mode, model_mode, DS_MODEL,
                pgg_param_names, dsg_param_names,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
                no_emulator=args.no_emulator,
                pgg_cosmo_grid=pgg_cosmo_grid, dsg_cosmo_grid=dsg_cosmo_grid,
            )

            def theory_fn_masked(theta):
                pred = theory_fn(theta)
                n_pgg = len(data_pgg)
                return np.concatenate([pred[:n_pgg][pgg_mask], pred[n_pgg:][dsg_mask]])

    if args.synthetic and not args.analytic_cov:
        cov_pgg, precision_pgg = diagonal_covariance(data_pgg_masked, rescale=args.cov_rescale)
        cov_dsg, precision_dsg = diagonal_covariance(data_dsg_masked, rescale=args.cov_rescale)
    elif args.diag_cov:
        cov_pgg, precision_pgg = diagonal_covariance(data_pgg_masked, rescale=args.cov_rescale)
        cov_dsg, precision_dsg = diagonal_covariance(data_dsg_masked, rescale=args.cov_rescale)
    else:
        fiducial_poles_pgg = (
            {ell: poles_pgg[ell] for ell in PGG_ELLS}
            if not args.synthetic else data_pgg
        )
        cov_pgg, precision_pgg = _resolve_pgg_covariance(
            args, k_pgg, data_pgg_masked, pgg_mask, fiducial_poles=fiducial_poles_pgg,
        )
        cov_dsg, precision_dsg = diagonal_covariance(data_dsg_masked, rescale=args.cov_rescale)
    cov, precision_matrix = _combine_block_covariances(cov_pgg, precision_pgg, cov_dsg, precision_dsg)

    if args.synthetic:
        truth_theta = _build_truth_theta(param_names, truth_lookup, fixed_cosmo)
        if use_marg:
            m_truth, T_truth = decomposed_fn_masked(truth_theta)
            alpha_truth = _joint_linear_truth(joint_linear_names, truth_lookup)
            pred_truth = m_truth + T_truth @ alpha_truth
        else:
            pred_truth = theory_fn_masked(truth_theta)
        if args.vary_cosmo and not args.no_emulator:
            print("  Reprojecting synthetic data onto the cosmology-grid theory basis.")
            data_y_masked = pred_truth.copy()
        if not np.allclose(pred_truth, data_y_masked, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Synthetic joint self-check failed: theory does not reproduce the injected data vector.")
        print(f"  Synthetic self-check chi2: {np.dot(data_y_masked - pred_truth, data_y_masked - pred_truth):.3e}")

    if use_marg:
        marg_like = MarginalizedLikelihood(data_y_masked, precision_matrix, prior_sigmas)

        def log_likelihood(theta):
            m, T = decomposed_fn_masked(theta)
            return marg_like(m, T)
    else:
        log_likelihood = make_log_likelihood(data_y_masked, precision_matrix, theory_fn_masked)

    prior = Prior([sp_uniform(loc=lo, scale=hi - lo) for lo, hi in bounds])

    print("Initialising PocoMC sampler ...")
    sampler = Sampler(
        prior=prior,
        likelihood=log_likelihood,
        n_dim=len(param_names),
        n_effective=512,
        n_active=256,
        output_dir=str(output_dir),
        random_state=42,
    )
    sampler.run(n_total=2000, n_evidence=0, progress=True)

    samples, weights, logl, logp = sampler.posterior()
    save_kwargs = {
        "samples": samples,
        "weights": weights,
        "logl": logl,
        "param_names": np.array(param_names),
        "fixed_cosmo": np.array([f"{k}={v}" for k, v in fixed_cosmo.items()]),
    }
    if use_marg:
        linear_samples = np.zeros((samples.shape[0], len(joint_linear_names)))
        for i in range(samples.shape[0]):
            m, T = decomposed_fn_masked(samples[i])
            linear_samples[i] = marg_like.bestfit_linear_params(m, T)
        save_kwargs["linear_param_names"] = np.array(joint_linear_names)
        save_kwargs["linear_samples"] = linear_samples
        save_kwargs["prior_sigmas"] = prior_sigmas

    chains_path = output_dir / "chains.npz"
    np.savez(chains_path, **save_kwargs)
    print(f"\nChains saved to {chains_path}")


if __name__ == "__main__":
    main()
