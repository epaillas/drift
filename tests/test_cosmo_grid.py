"""Tests for LinearPowerGrid: accuracy and boundary behaviour."""

import numpy as np
import pytest

from drift.utils.cosmology import (
    get_cosmology,
    get_linear_power,
    get_growth_rate,
    LinearPowerGrid,
    _DEFAULT_PARAMS,
)

Z = 0.5

# Fiducial values for all 5 cosmo params
_FIDUCIALS = {
    "sigma8": _DEFAULT_PARAMS["sigma8"],
    "omega_cdm": _DEFAULT_PARAMS["omega_cdm"],
    "omega_b": _DEFAULT_PARAMS["omega_b"],
    "h": _DEFAULT_PARAMS["h"],
    "n_s": _DEFAULT_PARAMS["n_s"],
}


@pytest.fixture(scope="module")
def k():
    return np.logspace(-2, np.log10(0.3), 60)


@pytest.fixture(scope="module")
def small_grid_2d(k):
    """Small 2D grid (sigma8 x omega_cdm), 8 points per axis."""
    return LinearPowerGrid(
        k, z=Z,
        cosmo_ranges={
            "sigma8": (0.7, 0.9, 8),
            "omega_cdm": (0.10, 0.14, 8),
        },
    )


@pytest.fixture(scope="module")
def small_grid_1d(k):
    """1D grid varying only sigma8."""
    return LinearPowerGrid(
        k, z=Z,
        cosmo_ranges={"sigma8": (0.7, 0.9, 8)},
    )


@pytest.fixture(scope="module")
def small_grid_3d(k):
    """3D grid (sigma8 x omega_cdm x h), 5 points per axis."""
    return LinearPowerGrid(
        k, z=Z,
        cosmo_ranges={
            "sigma8": (0.7, 0.9, 5),
            "omega_cdm": (0.10, 0.14, 5),
            "h": (0.60, 0.75, 5),
        },
    )


# ---------------------------------------------------------------------------
# Accuracy tests — 2D grid
# ---------------------------------------------------------------------------

def test_grid_2d_recovers_fiducial_plin(k, small_grid_2d):
    """Interpolated P_lin at fiducial cosmology should match CLASS to < 0.5%."""
    cosmo = get_cosmology()
    plin_ref = get_linear_power(cosmo, k, Z)
    plin_interp, _ = small_grid_2d.predict(
        sigma8=_FIDUCIALS["sigma8"], omega_cdm=_FIDUCIALS["omega_cdm"],
    )
    max_err = np.max(np.abs(plin_interp / plin_ref - 1.0))
    assert max_err < 0.005, (
        f"Max relative P_lin interpolation error {max_err:.4f} exceeds 0.5%"
    )


def test_grid_2d_recovers_fiducial_f(k, small_grid_2d):
    """Interpolated f at fiducial cosmology should match CLASS to < 0.5%."""
    cosmo = get_cosmology()
    f_ref = get_growth_rate(cosmo, Z)
    _, f_interp = small_grid_2d.predict(
        sigma8=_FIDUCIALS["sigma8"], omega_cdm=_FIDUCIALS["omega_cdm"],
    )
    rel_err = abs(f_interp / f_ref - 1.0)
    assert rel_err < 0.005, (
        f"Relative f interpolation error {rel_err:.4f} exceeds 0.5%"
    )


# ---------------------------------------------------------------------------
# Accuracy tests — 1D grid
# ---------------------------------------------------------------------------

def test_grid_1d_recovers_fiducial(k, small_grid_1d):
    """1D grid (sigma8 only) at fiducial should match CLASS to < 0.5%."""
    cosmo = get_cosmology()
    plin_ref = get_linear_power(cosmo, k, Z)
    plin_interp, f_interp = small_grid_1d.predict(sigma8=_FIDUCIALS["sigma8"])
    max_err = np.max(np.abs(plin_interp / plin_ref - 1.0))
    assert max_err < 0.005, (
        f"Max relative P_lin interpolation error {max_err:.4f} exceeds 0.5% (1D grid)"
    )
    f_ref = get_growth_rate(cosmo, Z)
    assert abs(f_interp / f_ref - 1.0) < 0.005


# ---------------------------------------------------------------------------
# Accuracy tests — 3D grid
# ---------------------------------------------------------------------------

def test_grid_3d_recovers_fiducial(k, small_grid_3d):
    """3D grid (sigma8 x omega_cdm x h) at fiducial should match CLASS to < 1%."""
    cosmo = get_cosmology()
    plin_ref = get_linear_power(cosmo, k, Z)
    plin_interp, f_interp = small_grid_3d.predict(
        sigma8=_FIDUCIALS["sigma8"],
        omega_cdm=_FIDUCIALS["omega_cdm"],
        h=_FIDUCIALS["h"],
    )
    max_err = np.max(np.abs(plin_interp / plin_ref - 1.0))
    assert max_err < 0.01, (
        f"Max relative P_lin interpolation error {max_err:.4f} exceeds 1% (3D grid)"
    )
    f_ref = get_growth_rate(cosmo, Z)
    assert abs(f_interp / f_ref - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Fixed params test
# ---------------------------------------------------------------------------

def test_fixed_params_not_in_axes(k):
    """Fixed params should not appear in grid axes."""
    grid = LinearPowerGrid(
        k, z=Z,
        cosmo_ranges={"sigma8": (0.7, 0.9, 5)},
        fixed_params={"h": 0.70, "omega_cdm": 0.12},
    )
    assert grid._axis_names == ["sigma8"]
    assert "h" not in grid._axis_names
    assert grid._fixed_params["h"] == 0.70
    assert grid._fixed_params["omega_cdm"] == 0.12


# ---------------------------------------------------------------------------
# Boundary test
# ---------------------------------------------------------------------------

def test_grid_out_of_bounds_raises(k, small_grid_2d):
    """predict() with out-of-range parameters should raise ValueError."""
    with pytest.raises(ValueError):
        small_grid_2d.predict(sigma8=0.5, omega_cdm=_FIDUCIALS["omega_cdm"])

    with pytest.raises(ValueError):
        small_grid_2d.predict(sigma8=_FIDUCIALS["sigma8"], omega_cdm=0.01)
