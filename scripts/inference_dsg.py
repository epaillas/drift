"""DSG parameter inference with PocoMC.

Recovers per-bin density biases bq[1..5] and galaxy tracer bias b1
from density-split × galaxy multipole measurements in outputs/dsg_measured.hdf5.
Cosmology is fixed to Planck 2018 by default; use --vary-cosmo to sample it.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

from drift.cosmology import (
    get_cosmology, LinearPowerGrid, OneLoopPowerGrid,
    get_linear_power, get_growth_rate,
    _DEFAULT_PARAMS, DEFAULT_COSMO_RANGES, ALL_COSMO_NAMES,
)
from drift.emulator import TemplateEmulator
from drift.galaxy_models import _compute_loop_templates
from drift.analytic_marginalization import MarginalizedLikelihood
from drift.io import load_measurements, diagonal_covariance, taylor_cache_key
from drift.synthetic import make_synthetic_dsg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACE      = "redshift"         # "redshift" | "real"
DS_MODEL   = "rsd_selection" # "baseline" | "rsd_selection" | "phenomenological"
MODEL_MODE = "tree"         # "tree" | "eft_ct" | "eft" | "one_loop"

_suffix   = "_real" if SPACE == "real" else ""
MEAS_PATH = Path(__file__).parents[1] / "outputs" / f"dsg_measured{_suffix}.hdf5"
OUTPUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "inference_dsg" / SPACE / DS_MODEL / MODEL_MODE
)

Z = 0.5
R = 10.0
KERNEL    = "gaussian"
ELLS      = (0, 2)
QUANTILES = (1, 5)

VARY_COSMO = False


# ---------------------------------------------------------------------------
# Cosmology CLI helpers (mirrored from inference_pgg.py)
# ---------------------------------------------------------------------------
def parse_fix_cosmo(tokens):
    """Parse --fix-cosmo tokens into a dict of {name: value}."""
    fixed = {}
    if tokens is None:
        return fixed
    for tok in tokens:
        if "=" in tok:
            name, val = tok.split("=", 1)
            fixed[name.strip()] = float(val)
        else:
            name = tok.strip()
            if name not in _DEFAULT_PARAMS:
                raise ValueError(f"Unknown cosmo parameter: {name}")
            fixed[name] = _DEFAULT_PARAMS[name]
    return fixed


def resolve_cosmo_params(vary_cosmo, fix_cosmo_tokens):
    """Return (free_cosmo_names, fixed_cosmo, cosmo_bounds)."""
    if not vary_cosmo:
        return [], {name: _DEFAULT_PARAMS[name] for name in ALL_COSMO_NAMES}, np.empty((0, 2))

    fixed_cosmo = parse_fix_cosmo(fix_cosmo_tokens)
    free_cosmo_names = [name for name in ALL_COSMO_NAMES if name not in fixed_cosmo]

    for name in ALL_COSMO_NAMES:
        if name not in fixed_cosmo and name not in free_cosmo_names:
            fixed_cosmo[name] = _DEFAULT_PARAMS[name]

    cosmo_bounds = np.array([
        [DEFAULT_COSMO_RANGES[name][0], DEFAULT_COSMO_RANGES[name][1]]
        for name in free_cosmo_names
    ])
    return free_cosmo_names, fixed_cosmo, cosmo_bounds


# ---------------------------------------------------------------------------
# k-masking helpers
# ---------------------------------------------------------------------------
def _parse_kmax(kmax_args, ells):
    """Parse --kmax CLI values into {ell: float} dict."""
    kmax_dict = {ell: np.inf for ell in ells}
    if kmax_args is None:
        return kmax_dict
    if len(kmax_args) == 1 and ":" not in kmax_args[0]:
        val = float(kmax_args[0])
        return {ell: val for ell in ells}
    for item in kmax_args:
        ell_str, val_str = item.split(":")
        kmax_dict[int(ell_str)] = float(val_str)
    return kmax_dict


def _build_data_mask(k, ells, quantiles, kmax_dict, kmin=0.0):
    """Build a flat boolean mask matching the data vector ordering."""
    segments = []
    for _q in quantiles:
        for ell in ells:
            kmax = kmax_dict.get(ell, np.inf)
            segments.append((k >= kmin) & (k <= kmax))
    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------
def _build_params(ds_model, model_mode, quantiles, free_cosmo_names=None,
                  cosmo_bounds=None):
    """Return (param_names, bounds, is_linear) for the given configuration."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"] + [f"bq1_{q}" for q in quantiles]
    bounds_list = (list(cosmo_bounds) if len(cosmo_bounds) else []) + [[0.5, 4.0]] + [[-4.0, 4.0]] * len(quantiles)
    bounds = np.array(bounds_list)
    is_linear = [False] * len(param_names)

    if model_mode != "tree":
        param_names.append("sigma_fog")
        bounds = np.vstack([bounds, [[0.0, 30.0]]])
        is_linear.append(False)

    if model_mode in ("eft_ct", "eft", "one_loop"):
        param_names += ["c0"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
        is_linear.append(True)

    if model_mode in ("eft", "one_loop"):
        param_names += ["s0"]
        bounds = np.vstack([bounds, [[-5000.0, 5000.0]]])
        is_linear.append(True)

    if model_mode in ("one_loop",):
        param_names += ["c2"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
        is_linear.append(True)
        param_names += ["c4"]
        bounds = np.vstack([bounds, [[-50.0, 50.0]]])
        is_linear.append(True)
        param_names += ["s2"]
        bounds = np.vstack([bounds, [[-500.0, 500.0]]])
        is_linear.append(True)

    if model_mode == "one_loop":
        param_names += ["b2", "bs2"]
        bounds = np.vstack([bounds, [[-10.0, 10.0], [-10.0, 10.0]]])
        is_linear += [False, False]
        param_names += ["b3nl"]
        bounds = np.vstack([bounds, [[-10.0, 10.0]]])
        is_linear.append(True)

    if ds_model == "phenomenological":
        param_names += [f"beta_q_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-2.0, 2.0] for _ in quantiles]])
        is_linear += [False] * len(quantiles)

    if model_mode in ("eft_ct", "eft", "one_loop"):
        param_names += [f"bq_nabla2_{q}" for q in quantiles]
        bounds = np.vstack([bounds, [[-4.0, 4.0] for _ in quantiles]])
        is_linear += [True] * len(quantiles)

    return param_names, bounds, is_linear


def _build_params_marginalized(ds_model, model_mode, quantiles,
                                free_cosmo_names=None, cosmo_bounds=None):
    """Return (param_names, bounds) for nonlinear params only (marginalized mode)."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"] + [f"bq1_{q}" for q in quantiles]
    bounds_list = (list(cosmo_bounds) if len(cosmo_bounds) else []) + [[0.5, 4.0]] + [[-4.0, 4.0]] * len(quantiles)
    bounds = np.array(bounds_list)

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


PARAM_NAMES, BOUNDS, _ = _build_params(DS_MODEL, MODEL_MODE, QUANTILES)


# ---------------------------------------------------------------------------
# Parameter unpacking
# ---------------------------------------------------------------------------
def _unpack_theta(theta, n_bq, mode, ds_model, quantiles,
                  free_cosmo_names=None, fixed_cosmo=None):
    """Unpack flat parameter vector into a dict."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}

    idx = 0
    cosmo_vals = {}
    for name in free_cosmo_names:
        cosmo_vals[name] = float(theta[idx]); idx += 1
    cosmo_vals.update(fixed_cosmo)

    b1 = float(theta[idx]); idx += 1
    bq1_vals = list(theta[idx:idx + n_bq]); idx += n_bq

    sigma_fog = 0.0
    if mode != "tree":
        sigma_fog = float(theta[idx]); idx += 1

    c0, s0, c2, c4, s2, b2, bs2, b3nl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if mode in ("eft_ct", "eft", "one_loop"):
        c0 = float(theta[idx]); idx += 1
    if mode in ("eft", "one_loop"):
        s0 = float(theta[idx]); idx += 1
    if mode in ("one_loop",):
        c2 = float(theta[idx]); idx += 1
        c4 = float(theta[idx]); idx += 1
        s2 = float(theta[idx]); idx += 1
    if mode == "one_loop":
        b2   = float(theta[idx]); idx += 1
        bs2  = float(theta[idx]); idx += 1
        b3nl = float(theta[idx]); idx += 1

    if ds_model == "phenomenological":
        beta_q_vals = list(theta[idx:idx + n_bq]); idx += n_bq
    else:
        beta_q_vals = [0.0] * n_bq

    if mode in ("eft_ct", "eft", "one_loop"):
        bq_nabla2_vals = list(theta[idx:idx + n_bq]); idx += n_bq
    else:
        bq_nabla2_vals = [0.0] * n_bq

    return dict(b1=b1, bq1_vals=bq1_vals, sigma_fog=sigma_fog,
                c0=c0, s0=s0, c2=c2, c4=c4, s2=s2,
                b2=b2, bs2=bs2, b3nl=b3nl,
                beta_q_vals=beta_q_vals, bq_nabla2_vals=bq_nabla2_vals,
                cosmo=cosmo_vals)


def _unpack_theta_marginalized(theta, n_bq, mode, ds_model, quantiles,
                                free_cosmo_names=None, fixed_cosmo=None):
    """Unpack flat parameter vector (nonlinear params only)."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}

    idx = 0
    cosmo_vals = {}
    for name in free_cosmo_names:
        cosmo_vals[name] = float(theta[idx]); idx += 1
    cosmo_vals.update(fixed_cosmo)

    b1 = float(theta[idx]); idx += 1
    bq1_vals = list(theta[idx:idx + n_bq]); idx += n_bq

    sigma_fog = 0.0
    if mode != "tree":
        sigma_fog = float(theta[idx]); idx += 1

    b2, bs2 = 0.0, 0.0
    if mode == "one_loop":
        b2  = float(theta[idx]); idx += 1
        bs2 = float(theta[idx]); idx += 1

    if ds_model == "phenomenological":
        beta_q_vals = list(theta[idx:idx + n_bq]); idx += n_bq
    else:
        beta_q_vals = [0.0] * n_bq

    return dict(b1=b1, bq1_vals=bq1_vals, sigma_fog=sigma_fog,
                b2=b2, bs2=bs2, beta_q_vals=beta_q_vals,
                cosmo=cosmo_vals)


# ---------------------------------------------------------------------------
# Theory models
# ---------------------------------------------------------------------------
def make_eft_theory_model(cosmo, k, ells, quantiles, ds_model, mode,
                          cosmo_grid=None, free_cosmo_names=None,
                          fixed_cosmo=None):
    """Return a callable theta -> flat_data_vector using TemplateEmulator."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    vary_cosmo = cosmo_grid is not None
    n_bq = len(quantiles)
    emulator = TemplateEmulator(
        cosmo, k, ells=ells, z=Z, R=R,
        kernel=KERNEL, space=SPACE,
        ds_model=ds_model, mode=mode,
    )

    def theory(theta):
        p = _unpack_theta(theta, n_bq, mode, ds_model, quantiles,
                          free_cosmo_names=free_cosmo_names,
                          fixed_cosmo=fixed_cosmo)
        if vary_cosmo:
            cosmo_kw = {name: p["cosmo"][name] for name in free_cosmo_names}
            if mode in ("one_loop",):
                plin, f, loop_arrays = cosmo_grid.predict(**cosmo_kw)
                emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
            else:
                plin, f = cosmo_grid.predict(**cosmo_kw)
                emulator.update_cosmology(plin, f)
        params = {
            "b1":        p["b1"],
            "bq1":       [float(v) for v in p["bq1_vals"]],
            "sigma_fog": p["sigma_fog"],
            "c0":        p["c0"],
            "c2":        p["c2"],
            "c4":        p["c4"],
            "s0":        p["s0"],
            "s2":        p["s2"],
            "b2":        p["b2"],
            "bs2":       p["bs2"],
            "b3nl":      p["b3nl"],
            "beta_q":    [float(v) for v in p["beta_q_vals"]],
            "bq_nabla2": [float(v) for v in p["bq_nabla2_vals"]],
        }
        return emulator.predict(params)

    return theory


def make_eft_theory_model_marginalized(cosmo, k, ells, quantiles, ds_model, mode,
                                        cosmo_grid=None, free_cosmo_names=None,
                                        fixed_cosmo=None):
    """Return a callable theta -> (m, T) for marginalized inference."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    vary_cosmo = len(free_cosmo_names) > 0
    use_grid   = cosmo_grid is not None
    n_bq = len(quantiles)
    emulator = TemplateEmulator(
        cosmo, k, ells=ells, z=Z, R=R,
        kernel=KERNEL, space=SPACE,
        ds_model=ds_model, mode=mode,
    )

    def decomposed_theory(theta):
        p = _unpack_theta_marginalized(theta, n_bq, mode, ds_model, quantiles,
                                       free_cosmo_names=free_cosmo_names,
                                       fixed_cosmo=fixed_cosmo)
        if vary_cosmo:
            if use_grid:
                cosmo_kw = {name: p["cosmo"][name] for name in free_cosmo_names}
                if mode in ("one_loop",):
                    plin, f, loop_arrays = cosmo_grid.predict(**cosmo_kw)
                    emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
                else:
                    plin, f = cosmo_grid.predict(**cosmo_kw)
                    emulator.update_cosmology(plin, f)
            else:
                eval_cosmo = get_cosmology(p["cosmo"])
                plin = get_linear_power(eval_cosmo, k, Z)
                f    = get_growth_rate(eval_cosmo, Z)
                if mode in ("one_loop",):
                    loop_arrays = _compute_loop_templates(
                        k, lambda kk: get_linear_power(eval_cosmo, np.asarray(kk), Z)
                    )
                    emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
                else:
                    emulator.update_cosmology(plin, f)
        params = {
            "b1":        p["b1"],
            "bq1":       [float(v) for v in p["bq1_vals"]],
            "sigma_fog": p["sigma_fog"],
            "b2":        p["b2"],
            "bs2":       p["bs2"],
            "beta_q":    [float(v) for v in p["beta_q_vals"]],
        }
        return emulator.predict_decomposed(params)

    return decomposed_theory


def make_direct_theory_model(cosmo, k, ells, quantiles, ds_model, mode,
                             free_cosmo_names=None, fixed_cosmo=None):
    """Return a callable theta -> flat_data_vector using direct GL quadrature."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    from drift.eft_bias import DSSplitBinEFT, GalaxyEFTParams
    from drift.eft_models import pqg_eft_mu
    from drift.multipoles import compute_multipoles

    vary_cosmo = len(free_cosmo_names) > 0
    n_bq = len(quantiles)

    def theory(theta):
        p = _unpack_theta(theta, n_bq, mode, ds_model, quantiles,
                          free_cosmo_names=free_cosmo_names,
                          fixed_cosmo=fixed_cosmo)
        eval_cosmo = (
            get_cosmology(p["cosmo"]) if vary_cosmo else cosmo
        )
        gal = GalaxyEFTParams(
            b1=p["b1"], sigma_fog=p["sigma_fog"],
            c0=p["c0"], c2=p["c2"], c4=p["c4"],
            s0=p["s0"], s2=p["s2"],
            b2=p["b2"], bs2=p["bs2"], b3nl=p["b3nl"],
        )
        vec = []
        for q, bq1, betaq, bq_nabla2 in zip(
            quantiles, p["bq1_vals"], p["beta_q_vals"], p["bq_nabla2_vals"]
        ):
            ds_bin = DSSplitBinEFT(
                label=f"DS{q}", bq1=float(bq1),
                beta_q=float(betaq), bq_nabla2=float(bq_nabla2),
            )

            def model(kk, mu, _ds=ds_bin, _gal=gal, _cosmo=eval_cosmo):
                return pqg_eft_mu(
                    kk, mu, z=Z, cosmo=_cosmo,
                    ds_params=_ds, gal_params=_gal,
                    R=R, kernel=KERNEL, space=SPACE,
                    ds_model=ds_model, mode=mode,
                )

            poles = compute_multipoles(k, model, ells=ells)
            for ell in ells:
                vec.append(poles[ell])
        return np.concatenate(vec)

    return theory


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------
def make_log_likelihood(data_y, precision_matrix, theory_fn):
    def log_likelihood(theta):
        pred = theory_fn(theta)
        diff = data_y - pred
        return -0.5 * diff @ precision_matrix @ diff
    return log_likelihood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-emulator",
        action="store_true",
        help="Use direct GL quadrature instead of the analytic template emulator.",
    )
    parser.add_argument(
        "--vary-cosmo",
        action="store_true",
        default=VARY_COSMO,
        help="Jointly infer cosmological parameters alongside bias parameters.",
    )
    parser.add_argument(
        "--fix-cosmo",
        nargs="+",
        default=None,
        metavar="PARAM[=VALUE]",
        help="Fix cosmological parameters when --vary-cosmo is set.",
    )
    parser.add_argument(
        "--mode",
        choices=["tree", "eft_ct", "eft", "one_loop"],
        default=MODEL_MODE,
        help=f"EFT model mode (default: {MODEL_MODE}).",
    )
    parser.add_argument(
        "--kmax",
        nargs="+",
        default=None,
        metavar="[ELL:]VALUE",
        help="Maximum k to include. Single float or 'ell:value' pairs.",
    )
    parser.add_argument(
        "--kmin",
        type=float,
        default=0.0,
        metavar="VALUE",
        help="Minimum k to include (default: 0).",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=5,
        metavar="N",
        help="Keep every Nth k-bin when loading measurements (default: 5).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic (noiseless) data vector instead of measurements.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["tree", "eft_ct", "eft", "one_loop"],
        default=None,
        metavar="MODE",
        help="Model mode for synthetic data generation (default: same as --mode).",
    )
    parser.add_argument(
        "--taylor",
        action="store_true",
        help="Wrap theory model in a Taylor expansion emulator.",
    )
    parser.add_argument(
        "--taylor-order",
        type=int,
        default=4,
        metavar="N",
    )
    parser.add_argument(
        "--taylor-step",
        type=float,
        default=0.01,
        metavar="FRAC",
    )
    parser.add_argument(
        "--diag-cov",
        action="store_true",
        help="Use diagonal covariance (always used for synthetic data).",
    )
    parser.add_argument(
        "--cov-rescale",
        type=float,
        default=64.0,
        metavar="FACTOR",
    )
    parser.add_argument(
        "--analytic-marg",
        action="store_true",
        help="Analytically marginalize over linear nuisance parameters.",
    )
    parser.add_argument(
        "--marg-prior-sigma",
        nargs="+",
        default=None,
        metavar="PARAM=VALUE",
        help="Override prior sigmas for analytic marginalization.",
    )
    args = parser.parse_args()

    model_mode = args.mode

    free_cosmo_names, fixed_cosmo, cosmo_bounds = resolve_cosmo_params(
        args.vary_cosmo, args.fix_cosmo,
    )

    use_marg = args.analytic_marg
    if use_marg:
        PARAM_NAMES, BOUNDS = _build_params_marginalized(
            DS_MODEL, model_mode, QUANTILES,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )
        marg_sigma_overrides = {}
        if args.marg_prior_sigma:
            for tok in args.marg_prior_sigma:
                name, val = tok.split("=", 1)
                marg_sigma_overrides[name.strip()] = float(val)
    else:
        PARAM_NAMES, BOUNDS, _ = _build_params(
            DS_MODEL, model_mode, QUANTILES,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )

    output_dir = (
        Path(__file__).parents[1] / "outputs" / "inference_dsg" / SPACE / DS_MODEL / model_mode
    )

    if args.vary_cosmo:
        print(f"Free cosmo params: {free_cosmo_names}")
        print(f"Fixed cosmo params: {fixed_cosmo}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load / generate data
    if args.synthetic:
        synthetic_mode = args.synthetic_mode or model_mode
        n_q = len(QUANTILES)
        if synthetic_mode == "tree":
            TRUE_PARAMS = {"b1": 2.0, "bq1": [0.5, 1.0, -1.0, -0.5],
                           "beta_q": [0.5]*n_q}
        elif synthetic_mode in ("eft_ct",):
            TRUE_PARAMS = {"b1": 2.0, "bq1": [0.5, 1.0, -1.0, -0.5],
                           "sigma_fog": 5.0, "c0": 5.0,
                           "beta_q": [0.5]*n_q, "bq_nabla2": [0.1]*n_q}
        elif synthetic_mode == "eft":
            TRUE_PARAMS = {"b1": 2.0, "bq1": [0.5, 1.0, -1.0, -0.5],
                           "sigma_fog": 5.0, "c0": 5.0, "s0": 100.0,
                           "beta_q": [0.5]*n_q, "bq_nabla2": [0.1]*n_q}
        else:  # one_loop / one_loop_matter_only
            TRUE_PARAMS = {"b1": 2.0, "bq1": [0.5, 1.0, -1.0, -0.5],
                           "sigma_fog": 5.0, "c0": 5.0, "c2": 2.0, "c4": 0.0,
                           "s0": 100.0, "s2": 0.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1,
                           "beta_q": [0.5]*n_q, "bq_nabla2": [0.1]*n_q}
        k = np.linspace(0.01, 0.3, 30)
        cosmo = get_cosmology()
        print(f"Generating synthetic DS×g data vector (mode={synthetic_mode}) ...")
        print(f"  True parameters: {TRUE_PARAMS}")
        data_y, _ = make_synthetic_dsg(
            k, ells=ELLS, z=Z, R=R, kernel=KERNEL, space=SPACE,
            ds_model=DS_MODEL, mode=synthetic_mode,
            true_params=TRUE_PARAMS, cosmo=cosmo,
        )
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")
        print(f"  Data vector length: {len(data_y)}")
    else:
        print(f"Loading measurements from {MEAS_PATH} ...")
        k, multipoles_per_bin = load_measurements(
            MEAS_PATH, nquantiles=max(QUANTILES), ells=ELLS, rebin=args.rebin,
        )
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")
        data_y = np.concatenate([
            multipoles_per_bin[f"DS{q}"][ell]
            for q in QUANTILES
            for ell in ELLS
        ])
        print(f"  Data vector length: {len(data_y)}")

    # Apply kmax/kmin masking
    kmax_dict = _parse_kmax(args.kmax, ELLS)
    mask = _build_data_mask(k, ELLS, QUANTILES, kmax_dict, kmin=args.kmin)
    print("  kmax cuts:")
    for ell in ELLS:
        kmax_val = kmax_dict.get(ell, np.inf)
        n_kept = ((k >= args.kmin) & (k <= kmax_val)).sum()
        print(f"    ell={ell}: kmax={kmax_val:.4g} h/Mpc  ({n_kept}/{len(k)} k-bins kept)")
    data_y_masked = data_y[mask]
    print(f"  Masked data vector length: {len(data_y_masked)}")

    # 2. Cosmology
    if not args.synthetic:
        cosmo = get_cosmology()

    # 3. Theory model
    print(f"  DS model: {DS_MODEL},  EFT mode: {model_mode}")
    if use_marg:
        print(f"  Analytic marginalization: ON (sampled params: {PARAM_NAMES})")

    # Determine linear param names for marginalized mode
    if use_marg:
        _tmp_emu = TemplateEmulator(cosmo, k, ells=ELLS, z=Z, R=R,
                                    kernel=KERNEL, space=SPACE,
                                    ds_model=DS_MODEL, mode=model_mode)
        shared_lin = _tmp_emu.linear_param_names
        n_q = len(QUANTILES)
        # per-quantile bq_nabla2 are also linear
        bq_nabla2_names = [f"bq_nabla2_{q}" for q in QUANTILES] if model_mode != "tree" else []
        linear_param_names = shared_lin + bq_nabla2_names

        all_names, all_bounds, _ = _build_params(
            DS_MODEL, model_mode, QUANTILES,
            free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )
        prior_sigmas = []
        for name in linear_param_names:
            idx = all_names.index(name)
            lo, hi = all_bounds[idx]
            sigma = marg_sigma_overrides.get(name, (hi - lo) / 2.0)
            prior_sigmas.append(sigma)
        prior_sigmas = np.array(prior_sigmas)
        print(f"  Linear params (marginalized): {linear_param_names}")
        print(f"  Prior sigmas: {dict(zip(linear_param_names, prior_sigmas))}")
        del _tmp_emu

    if args.taylor and use_marg:
        from drift.taylor import TaylorEmulator
        fiducial = {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(PARAM_NAMES, BOUNDS)}
        cache_hash = taylor_cache_key(
            ds_model=DS_MODEL, model_mode=model_mode, space=SPACE, z=Z,
            ells=ELLS, quantiles=QUANTILES, kmax=str(kmax_dict),
            kmin=args.kmin, rebin=args.rebin,
            free_cosmo_names=str(free_cosmo_names),
            fixed_cosmo=str(sorted(fixed_cosmo.items())),
            taylor_order=args.taylor_order, taylor_step=args.taylor_step,
            fiducial=str(sorted(fiducial.items())),
            param_names=str(PARAM_NAMES),
            analytic_marg=True,
        )
        cache_path = output_dir / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            decomposed_fn = make_eft_theory_model_marginalized(
                cosmo, k, ells=ELLS, quantiles=QUANTILES,
                ds_model=DS_MODEL, mode=model_mode,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            n_lin = len(linear_param_names)

            def _dict_decomposed(params):
                theta = np.array([params[name] for name in PARAM_NAMES])
                m, T = decomposed_fn(theta)
                m_masked = m[mask]
                T_masked = T[mask]
                return np.concatenate([m_masked, T_masked.ravel()])

            print(f"Building Taylor emulator for decomposed theory "
                  f"(order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(
                _dict_decomposed, fiducial,
                order=args.taylor_order, step_sizes=args.taylor_step,
            )
            taylor_emu.save_coefficients(cache_path)
            print(f"  Cached Taylor coefficients to {cache_path}")

        n_data_masked = mask.sum()
        n_lin = len(linear_param_names)

        def decomposed_fn_masked(theta):
            out = taylor_emu.predict(
                {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
            )
            m = out[:n_data_masked]
            T = out[n_data_masked:].reshape(n_data_masked, n_lin)
            return m, T

        print("  done.")

    elif args.taylor:
        from drift.taylor import TaylorEmulator
        fiducial = {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(PARAM_NAMES, BOUNDS)}
        cache_hash = taylor_cache_key(
            ds_model=DS_MODEL, model_mode=model_mode, space=SPACE, z=Z,
            ells=ELLS, quantiles=QUANTILES, kmax=str(kmax_dict),
            kmin=args.kmin, rebin=args.rebin,
            free_cosmo_names=str(free_cosmo_names),
            fixed_cosmo=str(sorted(fixed_cosmo.items())),
            taylor_order=args.taylor_order, taylor_step=args.taylor_step,
            fiducial=str(sorted(fiducial.items())),
            param_names=str(PARAM_NAMES),
        )
        cache_path = output_dir / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            print("Using direct GL quadrature ...")
            base_theory_fn = make_direct_theory_model(
                cosmo, k, ells=ELLS, quantiles=QUANTILES,
                ds_model=DS_MODEL, mode=model_mode,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            base_theory_masked = lambda theta: base_theory_fn(theta)[mask]

            def _dict_theory(params):
                theta = np.array([params[name] for name in PARAM_NAMES])
                return base_theory_masked(theta)

            print(f"Building Taylor emulator (order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(
                _dict_theory, fiducial,
                order=args.taylor_order, step_sizes=args.taylor_step,
            )
            taylor_emu.save_coefficients(cache_path)
            print(f"  Cached Taylor coefficients to {cache_path}")

        theory_fn_masked = lambda theta: taylor_emu.predict(
            {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
        )
        print("  done.")

    else:
        cosmo_grid = None
        if args.vary_cosmo and not args.no_emulator:
            cosmo_ranges = {name: DEFAULT_COSMO_RANGES[name] for name in free_cosmo_names}
            if model_mode in ("one_loop",):
                print(f"Precomputing OneLoopPowerGrid ({' x '.join(free_cosmo_names)}) ...")
                cosmo_grid = OneLoopPowerGrid(
                    k, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo,
                )
            else:
                print(f"Precomputing LinearPowerGrid ({' x '.join(free_cosmo_names)}) ...")
                cosmo_grid = LinearPowerGrid(
                    k, z=Z, cosmo_ranges=cosmo_ranges, fixed_params=fixed_cosmo,
                )
            print("  done.")

        if use_marg:
            print("Building template emulator (marginalized) ...")
            decomposed_fn_raw = make_eft_theory_model_marginalized(
                cosmo, k, ells=ELLS, quantiles=QUANTILES,
                ds_model=DS_MODEL, mode=model_mode, cosmo_grid=cosmo_grid,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            def decomposed_fn_masked(theta):
                m, T = decomposed_fn_raw(theta)
                return m[mask], T[mask]
        elif args.no_emulator:
            print("Using direct GL quadrature ...")
            theory_fn = make_direct_theory_model(
                cosmo, k, ells=ELLS, quantiles=QUANTILES,
                ds_model=DS_MODEL, mode=model_mode,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
        else:
            print("Building template emulator ...")
            theory_fn = make_eft_theory_model(
                cosmo, k, ells=ELLS, quantiles=QUANTILES,
                ds_model=DS_MODEL, mode=model_mode, cosmo_grid=cosmo_grid,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )

        if not use_marg:
            print("  done.")
            theory_fn_masked = lambda theta: theory_fn(theta)[mask]
        else:
            print("  done.")

    cov, precision_matrix = diagonal_covariance(data_y_masked, rescale=args.cov_rescale)

    # 4. Log-likelihood
    if use_marg:
        marg_like = MarginalizedLikelihood(data_y_masked, precision_matrix, prior_sigmas)
        def log_likelihood(theta):
            m, T = decomposed_fn_masked(theta)
            return marg_like(m, T)
    else:
        log_likelihood = make_log_likelihood(data_y_masked, precision_matrix, theory_fn_masked)

    # 5. Prior
    dists = [sp_uniform(loc=lo, scale=hi - lo) for lo, hi in BOUNDS]
    prior = Prior(dists)

    # 6. Sampler
    print("Initialising PocoMC sampler ...")
    sampler = Sampler(
        prior=prior,
        likelihood=log_likelihood,
        n_dim=len(PARAM_NAMES),
        n_effective=512,
        n_active=256,
        output_dir=str(output_dir),
        random_state=42,
    )
    sampler.run(n_total=2000, n_evidence=0, progress=True)

    # 7. Save results
    samples, weights, logl, logp = sampler.posterior()
    chains_path = output_dir / "chains.npz"

    save_kwargs = dict(
        samples=samples,
        weights=weights,
        logl=logl,
        param_names=np.array(PARAM_NAMES),
    )

    if use_marg:
        n_samples = samples.shape[0]
        n_linear  = len(linear_param_names)
        linear_samples = np.zeros((n_samples, n_linear))
        for i in range(n_samples):
            m, T = decomposed_fn_masked(samples[i])
            linear_samples[i] = marg_like.bestfit_linear_params(m, T)
        save_kwargs["linear_param_names"] = np.array(linear_param_names)
        save_kwargs["linear_samples"]     = linear_samples
        save_kwargs["prior_sigmas"]       = prior_sigmas

    np.savez(chains_path, **save_kwargs)
    print(f"\nChains saved to {chains_path}")

    # 8. Print summary
    if args.synthetic:
        print(f"\nTrue parameters: {TRUE_PARAMS}")
    print("\nParameter estimates (posterior mean ± std):")
    header = f"  {'name':>14}  {'mean':>8}  {'std':>8}"
    if args.synthetic:
        header += f"  {'true':>8}"
    print(header)
    for i, name in enumerate(PARAM_NAMES):
        mean = np.average(samples[:, i], weights=weights)
        var  = np.average((samples[:, i] - mean) ** 2, weights=weights)
        std  = np.sqrt(var)
        if args.synthetic:
            true_val = TRUE_PARAMS.get(name, 0.0)
            print(f"  {name:>14}  {mean:>8.3f}  {std:>8.3f}  {true_val:>8.3f}")
        else:
            print(f"  {name:>14}  {mean:>8.3f}  {std:>8.3f}")

    if use_marg:
        print("\nRecovered linear parameters (posterior mean ± std):")
        header = f"  {'name':>14}  {'mean':>8}  {'std':>8}"
        if args.synthetic:
            header += f"  {'true':>8}"
        print(header)
        for j, name in enumerate(linear_param_names):
            mean = np.average(linear_samples[:, j], weights=weights)
            var  = np.average((linear_samples[:, j] - mean) ** 2, weights=weights)
            std  = np.sqrt(var)
            if args.synthetic:
                true_val = TRUE_PARAMS.get(name, 0.0)
                print(f"  {name:>14}  {mean:>8.3f}  {std:>8.3f}  {true_val:>8.3f}")
            else:
                print(f"  {name:>14}  {mean:>8.3f}  {std:>8.3f}")


if __name__ == "__main__":
    main()
