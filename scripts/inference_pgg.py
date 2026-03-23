"""Galaxy auto-power P_gg multipole fitting with PocoMC.

Recovers galaxy bias b1 (and optionally EFT/stochastic nuisances and cosmology)
from galaxy auto-power spectrum multipole measurements.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from scipy.stats import uniform as sp_uniform
import pocomc
from pocomc import Sampler, Prior

from drift.cosmology import (
    get_cosmology, LinearPowerGrid, OneLoopPowerGrid,
    get_linear_power, get_growth_rate,
    _DEFAULT_PARAMS, DEFAULT_COSMO_RANGES, ALL_COSMO_NAMES,
)
from drift.galaxy_emulator import GalaxyTemplateEmulator
from drift.galaxy_models import _compute_loop_templates
from drift.analytic_marginalization import MarginalizedLikelihood
from drift.io import (
    analytic_pgg_covariance,
    load_pgg_measurements,
    mock_covariance,
    diagonal_covariance,
    taylor_cache_key,
)
from drift.synthetic import make_synthetic_pgg

# Import shared utilities from inference_dsg
from inference_dsg import make_log_likelihood

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACE      = "redshift"  # "redshift" | "real"
MODEL_MODE = "one_loop"  # "tree" | "eft_ct" | "eft" | "one_loop"

MEAS_PATH  = Path(__file__).parents[1] / "outputs" / "hods" / "mesh2_spectrum_poles_c000_hod007.h5"
COV_DIR    = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance"
OUTPUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "inference_pgg" / SPACE / MODEL_MODE
)

Z    = 0.5
ELLS = (0, 2, 4)

VARY_COSMO = False


def _validate_analytic_covariance_inputs(args):
    """Validate CLI arguments for the analytic covariance path."""
    if args.box_volume is None:
        raise ValueError("--analytic-cov requires --box-volume.")
    has_nd = args.number_density is not None
    has_sn = args.shot_noise is not None
    if has_nd == has_sn:
        raise ValueError(
            "--analytic-cov requires exactly one of --number-density or --shot-noise."
        )


def _resolve_pgg_covariance(args, k, data_y_masked, mask, fiducial_poles=None):
    """Return (cov, precision) for the selected P_gg covariance source."""
    synthetic = getattr(args, "synthetic", False)
    diag_cov = getattr(args, "diag_cov", False)
    analytic_cov = getattr(args, "analytic_cov", False)
    cov_rescale = getattr(args, "cov_rescale", 1.0)
    rebin = getattr(args, "rebin", 13)

    if synthetic and not analytic_cov:
        return diagonal_covariance(data_y_masked, rescale=cov_rescale)
    if diag_cov:
        return diagonal_covariance(data_y_masked, rescale=cov_rescale)
    if analytic_cov:
        _validate_analytic_covariance_inputs(args)
        poles = fiducial_poles if fiducial_poles is not None else data_y_masked
        cov, precision = analytic_pgg_covariance(
            k,
            poles,
            ELLS,
            volume=getattr(args, "box_volume", None),
            number_density=getattr(args, "number_density", None),
            shot_noise=getattr(args, "shot_noise", None),
            mask=mask,
            rescale=cov_rescale,
            terms=getattr(args, "analytic_cov_terms", "gaussian"),
            cng_amplitude=getattr(args, "cng_amplitude", 0.0),
            cng_coherence=getattr(args, "cng_coherence", 0.35),
        )
        print(f"Using analytic cubic-box covariance with shape {cov.shape}")
        return cov, precision
    print(f"Estimating covariance from mocks in {COV_DIR} ...")
    cov, precision = mock_covariance(
        COV_DIR, "pgg", ELLS, k_data=k, mask=mask,
        rescale=cov_rescale, rebin=rebin,
    )
    print(f"  Covariance matrix shape: {cov.shape}")
    return cov, precision


# ---------------------------------------------------------------------------
# Cosmology CLI helpers
# ---------------------------------------------------------------------------
def parse_fix_cosmo(tokens):
    """Parse --fix-cosmo tokens into a dict of {name: value}.

    Each token is either 'name=value' or just 'name' (uses Planck default).
    """
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
    """Return (free_cosmo_names, fixed_cosmo, cosmo_bounds).

    free_cosmo_names : list of str — params to vary in MCMC
    fixed_cosmo : dict — {name: value} for params held constant
    cosmo_bounds : np.ndarray, shape (n_free, 2)
    """
    if not vary_cosmo:
        return [], {name: _DEFAULT_PARAMS[name] for name in ALL_COSMO_NAMES}, np.empty((0, 2))

    fixed_cosmo = parse_fix_cosmo(fix_cosmo_tokens)
    free_cosmo_names = [name for name in ALL_COSMO_NAMES if name not in fixed_cosmo]

    # Fill in defaults for any param not mentioned at all
    for name in ALL_COSMO_NAMES:
        if name not in fixed_cosmo and name not in free_cosmo_names:
            fixed_cosmo[name] = _DEFAULT_PARAMS[name]

    cosmo_bounds = np.array([
        [DEFAULT_COSMO_RANGES[name][0], DEFAULT_COSMO_RANGES[name][1]]
        for name in free_cosmo_names
    ])
    return free_cosmo_names, fixed_cosmo, cosmo_bounds


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------
def _build_params(model_mode, free_cosmo_names=None, cosmo_bounds=None):
    """Return (param_names, bounds, is_linear) for the given model configuration."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"]
    bounds = np.vstack([cosmo_bounds, [[0.5, 4.0]]]) if len(cosmo_bounds) > 0 else np.array([[0.5, 4.0]])
    is_linear = [False] * len(param_names)  # cosmo params + b1 are nonlinear

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

    return param_names, bounds, is_linear


def _build_params_marginalized(model_mode, free_cosmo_names=None, cosmo_bounds=None):
    """Return (param_names, bounds) for nonlinear params only (marginalized mode)."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if cosmo_bounds is None:
        cosmo_bounds = np.empty((0, 2))

    param_names = list(free_cosmo_names) + ["b1"]
    bounds = np.vstack([cosmo_bounds, [[0.5, 4.0]]]) if len(cosmo_bounds) > 0 else np.array([[0.5, 4.0]])

    if model_mode != "tree":
        param_names.append("sigma_fog")
        bounds = np.vstack([bounds, [[0.0, 30.0]]])

    if model_mode == "one_loop":
        param_names += ["b2", "bs2"]
        bounds = np.vstack([bounds, [[-10.0, 10.0], [-10.0, 10.0]]])

    return param_names, bounds


def _unpack_theta_marginalized(theta, mode, free_cosmo_names=None, fixed_cosmo=None):
    """Unpack flat parameter vector into a dict (nonlinear params only)."""
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
    sigma_fog = 0.0
    if mode != "tree":
        sigma_fog = float(theta[idx]); idx += 1
    b2, bs2 = 0.0, 0.0
    if mode == "one_loop":
        b2 = float(theta[idx]); idx += 1
        bs2 = float(theta[idx]); idx += 1

    return dict(b1=b1, sigma_fog=sigma_fog, b2=b2, bs2=bs2, cosmo=cosmo_vals)



# ---------------------------------------------------------------------------
# Theory model helpers
# ---------------------------------------------------------------------------
def _unpack_theta(theta, mode, free_cosmo_names=None, fixed_cosmo=None):
    """Unpack flat parameter vector into a dict."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}

    idx = 0
    cosmo_vals = {}
    for name in free_cosmo_names:
        cosmo_vals[name] = float(theta[idx]); idx += 1
    # Merge with fixed
    cosmo_vals.update(fixed_cosmo)

    b1 = float(theta[idx]); idx += 1
    sigma_fog = 0.0
    if mode != "tree":
        sigma_fog = float(theta[idx]); idx += 1
    c0, c2, c4, s0, s2, b2, bs2, b3nl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if mode in ("eft_ct", "eft", "one_loop"):
        c0 = float(theta[idx]); idx += 1
    if mode in ("eft", "one_loop"):
        s0 = float(theta[idx]); idx += 1
    if mode in ("one_loop",):
        c2  = float(theta[idx]); idx += 1
        c4  = float(theta[idx]); idx += 1
        s2  = float(theta[idx]); idx += 1
    if mode == "one_loop":
        b2   = float(theta[idx]); idx += 1
        bs2  = float(theta[idx]); idx += 1
        b3nl = float(theta[idx]); idx += 1
    return dict(b1=b1, sigma_fog=sigma_fog, c0=c0, c2=c2, c4=c4, s0=s0, s2=s2,
                b2=b2, bs2=bs2, b3nl=b3nl,
                cosmo=cosmo_vals)


def make_eft_theory_model(cosmo, k, ells, mode, cosmo_grid=None,
                          free_cosmo_names=None, fixed_cosmo=None):
    """Return a callable theta -> flat_data_vector using GalaxyTemplateEmulator."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    vary_cosmo = cosmo_grid is not None
    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=Z, space=SPACE, mode=mode,
    )

    def theory(theta):
        p = _unpack_theta(theta, mode, free_cosmo_names=free_cosmo_names,
                          fixed_cosmo=fixed_cosmo)
        if vary_cosmo:
            cosmo_kw = {name: p["cosmo"][name] for name in free_cosmo_names}
            if mode in ("one_loop",):
                plin, f, loop_arrays = cosmo_grid.predict(**cosmo_kw)
                emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
            else:
                plin, f = cosmo_grid.predict(**cosmo_kw)
                emulator.update_cosmology(plin, f)
        params = {"b1": p["b1"], "sigma_fog": p.get("sigma_fog", 0.0),
                  "c0": p["c0"], "c2": p["c2"], "c4": p["c4"],
                  "s0": p["s0"], "s2": p["s2"],
                  "b2": p.get("b2", 0.0), "bs2": p.get("bs2", 0.0),
                  "b3nl": p.get("b3nl", 0.0)}
        return emulator.predict(params)

    return theory


def make_eft_theory_model_marginalized(cosmo, k, ells, mode, cosmo_grid=None,
                                        free_cosmo_names=None, fixed_cosmo=None):
    """Return a callable theta -> (m, T) for marginalized inference."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    vary_cosmo = len(free_cosmo_names or []) > 0
    use_grid = cosmo_grid is not None
    emulator = GalaxyTemplateEmulator(
        cosmo, k, ells=ells, z=Z, space=SPACE, mode=mode,
    )

    def decomposed_theory(theta):
        p = _unpack_theta_marginalized(theta, mode, free_cosmo_names=free_cosmo_names,
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
                # Direct computation (used by Taylor emulator builder)
                eval_cosmo = get_cosmology(p["cosmo"])
                plin = get_linear_power(eval_cosmo, k, Z)
                f = get_growth_rate(eval_cosmo, Z)
                if mode in ("one_loop",):
                    loop_arrays = _compute_loop_templates(
                        k, lambda kk: get_linear_power(eval_cosmo, np.asarray(kk), Z)
                    )
                    emulator.update_cosmology(plin, f, loop_arrays=loop_arrays)
                else:
                    emulator.update_cosmology(plin, f)
        params = {"b1": p["b1"], "sigma_fog": p.get("sigma_fog", 0.0),
                  "b2": p.get("b2", 0.0), "bs2": p.get("bs2", 0.0)}
        return emulator.predict_decomposed(params)

    return decomposed_theory


def make_direct_theory_model(cosmo, k, ells, mode,
                             free_cosmo_names=None, fixed_cosmo=None):
    """Return a callable theta -> flat_data_vector using direct GL quadrature."""
    if free_cosmo_names is None:
        free_cosmo_names = []
    if fixed_cosmo is None:
        fixed_cosmo = {}
    from drift.eft_bias import GalaxyEFTParams
    from drift.galaxy_models import pgg_eft_mu
    from drift.multipoles import compute_multipoles

    vary_cosmo = len(free_cosmo_names) > 0

    def theory(theta):
        p = _unpack_theta(theta, mode, free_cosmo_names=free_cosmo_names,
                          fixed_cosmo=fixed_cosmo)
        eval_cosmo = (
            get_cosmology(p["cosmo"])
            if vary_cosmo else cosmo
        )
        gal = GalaxyEFTParams(
            b1=p["b1"], sigma_fog=p.get("sigma_fog", 0.0),
            c0=p["c0"], c2=p["c2"], c4=p["c4"],
            s0=p["s0"], s2=p["s2"],
            b2=p.get("b2", 0.0), bs2=p.get("bs2", 0.0),
            b3nl=p.get("b3nl", 0.0),
        )

        def model(kk, mu):
            return pgg_eft_mu(
                kk, mu, z=Z, cosmo=eval_cosmo,
                gal_params=gal, space=SPACE, mode=mode,
            )

        poles = compute_multipoles(k, model, ells=ells)
        return np.concatenate([poles[ell] for ell in ells])

    return theory


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
        help=(
            "Fix cosmological parameters when --vary-cosmo is set. "
            "Accepts 'name=value' or just 'name' (uses Planck 2018 default). "
            "E.g. --fix-cosmo h=0.70 n_s omega_b=0.022"
        ),
    )
    parser.add_argument(
        "--kmax",
        nargs="+",
        default=None,
        metavar="[ELL:]VALUE",
        help=(
            "Maximum k to include in the fit. Either a single float (applied "
            "to all multipoles) or 'ell:value' pairs, e.g. '0:0.3 2:0.2'."
        ),
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=13,
        metavar="N",
        help="Keep every Nth k-bin when loading measurements (default: 13).",
    )
    parser.add_argument(
        "--kmin",
        type=float,
        default=0.01,
        metavar="VALUE",
        help="Minimum k to include in the fit (default: 0.01).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic (noiseless) data vector instead of measurements from disk.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["tree", "eft_ct", "eft", "one_loop"],
        default=None,
        metavar="MODE",
        help=(
            "Model mode used to generate synthetic data (default: same as MODEL_MODE). "
            "Allows cross-mode tests, e.g. generate with tree_only, fit with one_loop."
        ),
    )
    parser.add_argument(
        "--taylor",
        action="store_true",
        help="Wrap the theory model in a Taylor expansion emulator for fast evaluation.",
    )
    parser.add_argument(
        "--taylor-order",
        type=int,
        default=4,
        metavar="N",
        help="Maximum Taylor expansion order (default: 4). Only used with --taylor.",
    )
    parser.add_argument(
        "--taylor-step",
        type=float,
        default=0.01,
        metavar="FRAC",
        help="Relative step size for finite differences (default: 0.01). Only used with --taylor.",
    )
    parser.add_argument(
        "--diag-cov",
        action="store_true",
        help="Use diagonal covariance instead of mock covariance.",
    )
    parser.add_argument(
        "--analytic-cov",
        action="store_true",
        help="Use fixed fiducial analytic covariance for a cubic box.",
    )
    parser.add_argument(
        "--analytic-cov-terms",
        type=str,
        default="gaussian",
        metavar="TERMS",
        help=(
            "Analytic covariance terms to include. Supported values are "
            "'gaussian' and 'gaussian+effective_cng'."
        ),
    )
    parser.add_argument(
        "--cov-rescale",
        type=float,
        default=64.0,
        metavar="FACTOR",
        help="Divide the covariance matrix by this factor (default: 64).",
    )
    parser.add_argument(
        "--box-volume",
        type=float,
        default=None,
        metavar="V",
        help="Box volume in (Mpc/h)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--number-density",
        type=float,
        default=None,
        metavar="N",
        help="Galaxy number density in (h/Mpc)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--shot-noise",
        type=float,
        default=None,
        metavar="P0",
        help="Constant shot-noise power in (Mpc/h)^3 for analytic covariance.",
    )
    parser.add_argument(
        "--cng-amplitude",
        type=float,
        default=0.0,
        metavar="A",
        help="Amplitude of the effective connected non-Gaussian covariance term.",
    )
    parser.add_argument(
        "--cng-coherence",
        type=float,
        default=0.35,
        metavar="SIGMA",
        help="Log-k coherence length of the effective connected covariance term.",
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
        help=(
            "Override prior sigmas for analytic marginalization, e.g. 'c0=100 s0=5000'. "
            "Defaults are derived from uniform bounds: sigma = (hi - lo) / 2."
        ),
    )
    args = parser.parse_args()

    # Resolve which cosmo params are free vs fixed
    free_cosmo_names, fixed_cosmo, cosmo_bounds = resolve_cosmo_params(
        args.vary_cosmo, args.fix_cosmo,
    )

    use_marg = args.analytic_marg
    if use_marg:
        PARAM_NAMES, BOUNDS = _build_params_marginalized(
            MODEL_MODE, free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )
        # Resolve prior sigmas for marginalization
        marg_sigma_overrides = {}
        if args.marg_prior_sigma:
            for tok in args.marg_prior_sigma:
                name, val = tok.split("=", 1)
                marg_sigma_overrides[name.strip()] = float(val)
    else:
        PARAM_NAMES, BOUNDS, _ = _build_params(
            MODEL_MODE, free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
        )

    if args.vary_cosmo:
        print(f"Free cosmo params: {free_cosmo_names}")
        print(f"Fixed cosmo params: {fixed_cosmo}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements (or generate synthetic data)
    if args.synthetic:
        synthetic_mode = args.synthetic_mode or MODEL_MODE
        # True parameters depend on the synthetic mode:
        # simpler modes only need b1; extra params are zero by definition.
        if synthetic_mode == "tree":
            TRUE_PARAMS = {"b1": 2.0, "sigma_fog": 0.0}
        elif synthetic_mode == "eft_ct":
            TRUE_PARAMS = {"b1": 2.0, "sigma_fog": 5.0, "c0": 5.0}
        elif synthetic_mode == "eft":
            TRUE_PARAMS = {"b1": 2.0, "sigma_fog": 5.0, "c0": 5.0, "s0": 100.0}
        elif synthetic_mode in ("one_loop",):
            TRUE_PARAMS = {"b1": 2.0, "sigma_fog": 5.0, "c0": 5.0, "c2": 2.0, "c4": 0.0, "s0": 100.0, "s2": 0.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
        else:
            TRUE_PARAMS = {"b1": 2.0, "sigma_fog": 5.0, "c0": 5.0, "c2": 2.0, "c4": 0.0, "s0": 100.0, "s2": 0.0, "b2": 0.5, "bs2": -0.5, "b3nl": 0.1}
        k = np.linspace(0.01, 0.3, 30)
        cosmo = get_cosmology()
        print(f"Generating synthetic P_gg data vector (mode={synthetic_mode}) ...")
        print(f"  True parameters: {TRUE_PARAMS}")
        data_y, _ = make_synthetic_pgg(
            k, ells=ELLS, z=Z, space=SPACE, mode=synthetic_mode,
            true_params=TRUE_PARAMS, cosmo=cosmo,
        )
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")
        print(f"  Data vector length: {len(data_y)}")
    else:
        print(f"Loading measurements from {MEAS_PATH} ...")
        k, poles = load_pgg_measurements(MEAS_PATH, ells=ELLS, rebin=args.rebin, kmin=args.kmin)
        print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")

        data_y = np.concatenate([poles[ell] for ell in ELLS])
        print(f"  Data vector length: {len(data_y)}")

    # Apply per-ell kmax masking
    kmax_dict = {ell: 0.5 for ell in ELLS}
    if args.kmax is not None:
        if len(args.kmax) == 1 and ":" not in args.kmax[0]:
            val = float(args.kmax[0])
            kmax_dict = {ell: val for ell in ELLS}
        else:
            for item in args.kmax:
                ell_str, val_str = item.split(":")
                kmax_dict[int(ell_str)] = float(val_str)

    mask = np.concatenate([k <= kmax_dict[ell] for ell in ELLS])
    data_y_masked = data_y[mask]
    print(f"  Masked data vector length: {len(data_y_masked)}")

    # 2. Cosmology (may already be set in synthetic branch)
    if not args.synthetic:
        cosmo = get_cosmology()

    # 3. Theory model
    print(f"  EFT mode: {MODEL_MODE},  space: {SPACE}")
    if use_marg:
        print(f"  Analytic marginalization: ON (sampled params: {PARAM_NAMES})")

    # Determine linear param names for marginalized mode
    if use_marg:
        _tmp_emu = GalaxyTemplateEmulator(cosmo, k, ells=ELLS, z=Z, space=SPACE, mode=MODEL_MODE)
        linear_param_names = _tmp_emu.linear_param_names

        # Derive Gaussian prior sigmas from the standard run's uniform bounds:
        # sigma = (hi - lo) / 2 so ±1σ covers the full uniform interval.
        all_names, all_bounds, _ = _build_params(
            MODEL_MODE, free_cosmo_names=free_cosmo_names, cosmo_bounds=cosmo_bounds,
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
        # Taylor emulator over decomposed theory (nonlinear params only)
        from drift.taylor import TaylorEmulator
        fiducial = {name: 0.5 * (lo + hi) for name, (lo, hi) in zip(PARAM_NAMES, BOUNDS)}
        cache_hash = taylor_cache_key(
            model_mode=MODEL_MODE, space=SPACE, z=Z, ells=ELLS,
            kmax=str(kmax_dict), kmin=args.kmin, rebin=args.rebin,
            free_cosmo_names=str(free_cosmo_names),
            fixed_cosmo=str(sorted(fixed_cosmo.items())),
            taylor_order=args.taylor_order, taylor_step=args.taylor_step,
            fiducial=str(sorted(fiducial.items())),
            param_names=str(PARAM_NAMES),
            analytic_marg=True,
        )
        cache_path = OUTPUT_DIR / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            # Build the decomposed theory function
            decomposed_fn = make_eft_theory_model_marginalized(
                cosmo, k, ells=ELLS, mode=MODEL_MODE,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            n_data = mask.sum()
            n_linear = len(linear_param_names)

            def _dict_decomposed(params):
                theta = np.array([params[name] for name in PARAM_NAMES])
                m, T = decomposed_fn(theta)
                m_masked = m[mask]
                T_masked = T[mask]
                return np.concatenate([m_masked, T_masked.ravel()])

            print(f"Building Taylor emulator for decomposed theory "
                  f"(order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(
                _dict_decomposed, fiducial, order=args.taylor_order,
                step_sizes=args.taylor_step,
            )
            taylor_emu.save_coefficients(cache_path)
            print(f"  Cached Taylor coefficients to {cache_path}")

        # Build decomposed theory from Taylor output
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
            model_mode=MODEL_MODE, space=SPACE, z=Z, ells=ELLS,
            kmax=str(kmax_dict), kmin=args.kmin, rebin=args.rebin,
            free_cosmo_names=str(free_cosmo_names),
            fixed_cosmo=str(sorted(fixed_cosmo.items())),
            taylor_order=args.taylor_order, taylor_step=args.taylor_step,
            fiducial=str(sorted(fiducial.items())),
            param_names=str(PARAM_NAMES),
        )
        cache_path = OUTPUT_DIR / f".taylor_cache_{cache_hash}.npz"

        if cache_path.exists():
            print(f"Loading cached Taylor emulator from {cache_path} ...")
            taylor_emu = TaylorEmulator.from_coefficients(cache_path)
        else:
            # Build the base theory model (only needed on cache miss)
            print("Using direct GL quadrature ...")
            base_theory_fn = make_direct_theory_model(
                cosmo, k, ells=ELLS, mode=MODEL_MODE,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            base_theory_masked = lambda theta: base_theory_fn(theta)[mask]

            def _dict_theory(params):
                theta = np.array([params[name] for name in PARAM_NAMES])
                return base_theory_masked(theta)

            print(f"Building Taylor emulator (order={args.taylor_order}, step={args.taylor_step}) ...")
            taylor_emu = TaylorEmulator(
                _dict_theory, fiducial, order=args.taylor_order,
                step_sizes=args.taylor_step,
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
            if MODEL_MODE in ("one_loop",):
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
                cosmo, k, ells=ELLS, mode=MODEL_MODE, cosmo_grid=cosmo_grid,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
            def decomposed_fn_masked(theta):
                m, T = decomposed_fn_raw(theta)
                return m[mask], T[mask]
        elif args.no_emulator:
            print("Using direct GL quadrature ...")
            theory_fn = make_direct_theory_model(
                cosmo, k, ells=ELLS, mode=MODEL_MODE,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
        else:
            print("Building template emulator ...")
            theory_fn = make_eft_theory_model(
                cosmo, k, ells=ELLS, mode=MODEL_MODE, cosmo_grid=cosmo_grid,
                free_cosmo_names=free_cosmo_names, fixed_cosmo=fixed_cosmo,
            )
        if not use_marg:
            print("  done.")
            theory_fn_masked = lambda theta: theory_fn(theta)[mask]
        else:
            print("  done.")

    fiducial_poles = {ell: poles[ell] for ell in ELLS} if not args.synthetic else data_y
    cov, precision_matrix = _resolve_pgg_covariance(
        args, k, data_y_masked, mask, fiducial_poles=fiducial_poles,
    )

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
        output_dir=str(OUTPUT_DIR),
        random_state=42,
    )
    sampler.run(n_total=2000, n_evidence=0, progress=True)

    # 7. Save results
    samples, weights, logl, logp = sampler.posterior()
    chains_path = OUTPUT_DIR / "chains.npz"

    save_kwargs = dict(
        samples=samples,
        weights=weights,
        logl=logl,
        param_names=np.array(PARAM_NAMES),
        fixed_cosmo=np.array([f"{k}={v}" for k, v in fixed_cosmo.items()]),
    )

    if use_marg:
        # Recover best-fit linear params for each posterior sample
        n_samples = samples.shape[0]
        n_linear = len(linear_param_names)
        linear_samples = np.zeros((n_samples, n_linear))
        for i in range(n_samples):
            m, T = decomposed_fn_masked(samples[i])
            linear_samples[i] = marg_like.bestfit_linear_params(m, T)
        save_kwargs["linear_param_names"] = np.array(linear_param_names)
        save_kwargs["linear_samples"] = linear_samples
        save_kwargs["prior_sigmas"] = prior_sigmas

    np.savez(chains_path, **save_kwargs)
    print(f"\nChains saved to {chains_path}")

    # 8. Print summary
    if args.synthetic:
        print(f"\nTrue parameters: {TRUE_PARAMS}")
    print("\nParameter estimates (posterior mean +/- std):")
    header = f"  {'name':>12}  {'mean':>8}  {'std':>8}"
    if args.synthetic:
        header += f"  {'true':>8}"
    print(header)
    for i, name in enumerate(PARAM_NAMES):
        mean = np.average(samples[:, i], weights=weights)
        var  = np.average((samples[:, i] - mean) ** 2, weights=weights)
        std  = np.sqrt(var)
        if args.synthetic:
            true_val = TRUE_PARAMS.get(name, 0.0)
            print(f"  {name:>12}  {mean:>8.3f}  {std:>8.3f}  {true_val:>8.3f}")
        else:
            print(f"  {name:>12}  {mean:>8.3f}  {std:>8.3f}")

    if use_marg:
        print("\nRecovered linear parameters (posterior mean +/- std):")
        header = f"  {'name':>12}  {'mean':>8}  {'std':>8}"
        if args.synthetic:
            header += f"  {'true':>8}"
        print(header)
        for j, name in enumerate(linear_param_names):
            mean = np.average(linear_samples[:, j], weights=weights)
            var  = np.average((linear_samples[:, j] - mean) ** 2, weights=weights)
            std  = np.sqrt(var)
            if args.synthetic:
                true_val = TRUE_PARAMS.get(name, 0.0)
                print(f"  {name:>12}  {mean:>8.3f}  {std:>8.3f}  {true_val:>8.3f}")
            else:
                print(f"  {name:>12}  {mean:>8.3f}  {std:>8.3f}")


if __name__ == "__main__":
    main()
