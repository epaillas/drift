"""Tests for LinearPowerGrid: accuracy and boundary behaviour."""

import numpy as np
import pytest

from drift.cosmology import (
    get_cosmology,
    get_linear_power,
    get_growth_rate,
    LinearPowerGrid,
)

Z = 0.5
_FIDUCIAL_S8 = 0.8102
_FIDUCIAL_OM = 0.3111


@pytest.fixture(scope="module")
def k():
    return np.logspace(-2, np.log10(0.3), 60)


@pytest.fixture(scope="module")
def small_grid(k):
    """Small 8×8 grid centred on the fiducial cosmology."""
    return LinearPowerGrid(
        k, z=Z,
        sigma8_range=(0.7, 0.9, 8),
        omega_m_range=(0.25, 0.40, 8),
    )


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

def test_grid_recovers_fiducial_plin(k, small_grid):
    """Interpolated P_lin at fiducial cosmology should match CLASS to < 0.5%."""
    cosmo = get_cosmology()
    plin_ref = get_linear_power(cosmo, k, Z)
    plin_interp, _ = small_grid.predict(_FIDUCIAL_S8, _FIDUCIAL_OM)
    max_err = np.max(np.abs(plin_interp / plin_ref - 1.0))
    assert max_err < 0.005, (
        f"Max relative P_lin interpolation error {max_err:.4f} exceeds 0.5%"
    )


def test_grid_recovers_fiducial_f(k, small_grid):
    """Interpolated f at fiducial cosmology should match CLASS to < 0.5%."""
    cosmo = get_cosmology()
    f_ref = get_growth_rate(cosmo, Z)
    _, f_interp = small_grid.predict(_FIDUCIAL_S8, _FIDUCIAL_OM)
    rel_err = abs(f_interp / f_ref - 1.0)
    assert rel_err < 0.005, (
        f"Relative f interpolation error {rel_err:.4f} exceeds 0.5%"
    )


# ---------------------------------------------------------------------------
# Boundary test
# ---------------------------------------------------------------------------

def test_grid_out_of_bounds_raises(k, small_grid):
    """predict() with out-of-range parameters should raise ValueError."""
    with pytest.raises(ValueError):
        small_grid.predict(0.5, _FIDUCIAL_OM)   # sigma8 too low

    with pytest.raises(ValueError):
        small_grid.predict(_FIDUCIAL_S8, 0.1)   # Omega_m too low
