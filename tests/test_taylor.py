"""Tests for the Taylor expansion emulator."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from drift.taylor import TaylorEmulator


def test_polynomial_recovery_exact():
    """A polynomial of degree <= order should be recovered exactly."""
    # f(x, y) = x^2 + 3*x*y + y^3
    def poly(params):
        x, y = params['x'], params['y']
        return np.array([x**2 + 3*x*y + y**3])

    fiducial = {'x': 1.0, 'y': 1.0}
    emu = TaylorEmulator(poly, fiducial, order=3, step_sizes=0.01, verbose=False)

    # Test at several points
    test_points = [
        {'x': 1.5, 'y': 0.5},
        {'x': 0.7, 'y': 1.3},
        {'x': 2.0, 'y': 2.0},
        {'x': 0.0, 'y': 0.0},
    ]
    for p in test_points:
        pred = emu.predict(p)
        exact = poly(p)
        np.testing.assert_allclose(pred, exact, atol=1e-6,
                                   err_msg=f"Failed at {p}")


def test_fiducial_recovery():
    """predict(fiducial) should return theory_fn(fiducial) exactly."""
    def fn(params):
        x = params['a']
        return np.array([np.sin(x), np.cos(x), x**2])

    fiducial = {'a': 0.5}
    emu = TaylorEmulator(fn, fiducial, order=4, step_sizes=0.01, verbose=False)
    pred = emu.predict(fiducial)
    exact = fn(fiducial)
    np.testing.assert_allclose(pred, exact, atol=1e-12)


def test_1d_exponential():
    """exp(x) at x=0, order 4: should be accurate near the origin."""
    def fn(params):
        return np.array([np.exp(params['x'])])

    fiducial = {'x': 0.0}
    emu = TaylorEmulator(fn, fiducial, order=4, step_sizes=0.01, verbose=False)

    # At x=0.1, truncation error ~ x^5/5! = 1e-5/120 ~ 8e-8
    pred = emu.predict({'x': 0.1})
    exact = np.exp(0.1)
    np.testing.assert_allclose(pred[0], exact, atol=1e-5)

    # At x=0.5, less accurate but still reasonable
    pred = emu.predict({'x': 0.5})
    exact = np.exp(0.5)
    np.testing.assert_allclose(pred[0], exact, atol=1e-2)


def test_multi_parameter_mixed_derivatives():
    """Test with known mixed derivatives: f(x,y) = sin(x)*cos(y)."""
    def fn(params):
        x, y = params['x'], params['y']
        return np.array([np.sin(x) * np.cos(y)])

    fiducial = {'x': 0.5, 'y': 0.3}
    emu = TaylorEmulator(fn, fiducial, order=4, step_sizes=0.005, verbose=False)

    test_points = [
        {'x': 0.55, 'y': 0.35},
        {'x': 0.6, 'y': 0.2},
        {'x': 0.4, 'y': 0.4},
    ]
    for p in test_points:
        pred = emu.predict(p)
        exact = fn(p)
        np.testing.assert_allclose(pred, exact, atol=1e-5,
                                   err_msg=f"Failed at {p}")


def test_order_sensitivity():
    """Higher order should give better accuracy for smooth functions."""
    def fn(params):
        return np.array([np.exp(params['x'])])

    fiducial = {'x': 0.0}
    test_point = {'x': 0.3}
    exact = np.exp(0.3)

    errors = []
    for order in [1, 2, 3, 4]:
        emu = TaylorEmulator(fn, fiducial, order=order, step_sizes=0.01,
                             verbose=False)
        pred = emu.predict(test_point)
        errors.append(abs(pred[0] - exact))

    # Each higher order should reduce the error
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], \
            f"Order {i+2} error ({errors[i+1]:.2e}) not less than " \
            f"order {i+1} error ({errors[i]:.2e})"


def test_step_size_as_dict():
    """Test absolute step size specification via dict."""
    def fn(params):
        x, y = params['x'], params['y']
        return np.array([x**2 + y**2])

    fiducial = {'x': 1.0, 'y': 2.0}
    step_sizes = {'x': 0.01, 'y': 0.02}
    emu = TaylorEmulator(fn, fiducial, order=2, step_sizes=step_sizes,
                         verbose=False)

    pred = emu.predict({'x': 1.5, 'y': 2.5})
    exact = fn({'x': 1.5, 'y': 2.5})
    np.testing.assert_allclose(pred, exact, atol=1e-6)


def test_param_names_subset():
    """Expand over a subset of parameters, holding others fixed."""
    def fn(params):
        x, y, z = params['x'], params['y'], params['z']
        return np.array([x**2 + y * z + z**2])

    fiducial = {'x': 1.0, 'y': 2.0, 'z': 3.0}

    # Only expand over x and y; z stays at fiducial
    emu = TaylorEmulator(fn, fiducial, order=2, step_sizes=0.01,
                         param_names=['x', 'y'], verbose=False)

    # Predict with z at fiducial (should be accurate)
    pred = emu.predict({'x': 1.5, 'y': 2.5, 'z': 3.0})
    exact = fn({'x': 1.5, 'y': 2.5, 'z': 3.0})
    np.testing.assert_allclose(pred, exact, atol=1e-5)


def test_n_terms_property():
    """Check that n_terms matches C(N+M, M) = (N+M)! / (N! * M!)."""
    from math import comb
    fiducial = {'a': 1.0, 'b': 2.0, 'c': 3.0}

    def fn(params):
        return np.array([0.0])

    for order in [1, 2, 3, 4]:
        emu = TaylorEmulator(fn, fiducial, order=order, step_sizes=0.01,
                             verbose=False)
        expected = comb(3 + order, order)
        assert emu.n_terms == expected, \
            f"order={order}: got {emu.n_terms}, expected {expected}"


def test_n_evals_property():
    """n_evals should be positive and less than n_terms * stencil_size."""
    def fn(params):
        return np.array([params['x'] + params['y']])

    fiducial = {'x': 1.0, 'y': 1.0}
    emu = TaylorEmulator(fn, fiducial, order=3, step_sizes=0.01, verbose=False)
    assert emu.n_evals > 0
    # With caching, n_evals should be much less than worst case
    assert emu.n_evals < 1000


def test_vector_output():
    """Test with multi-dimensional output."""
    def fn(params):
        x = params['x']
        return np.array([x, x**2, x**3, np.sin(x)])

    fiducial = {'x': 1.0}
    emu = TaylorEmulator(fn, fiducial, order=4, step_sizes=0.005, verbose=False)

    pred = emu.predict({'x': 1.05})
    exact = fn({'x': 1.05})
    np.testing.assert_allclose(pred, exact, atol=1e-5)


def test_three_params():
    """Test with 3 parameters and mixed terms."""
    def fn(params):
        a, b, c = params['a'], params['b'], params['c']
        return np.array([a * b + b * c + a * c, a**2 * b + c])

    fiducial = {'a': 1.0, 'b': 1.0, 'c': 1.0}
    emu = TaylorEmulator(fn, fiducial, order=3, step_sizes=0.01, verbose=False)

    pred = emu.predict({'a': 1.2, 'b': 0.8, 'c': 1.1})
    exact = fn({'a': 1.2, 'b': 0.8, 'c': 1.1})
    np.testing.assert_allclose(pred, exact, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    """Build emulator, save, load via from_coefficients, verify identical predictions."""
    def poly(params):
        x, y = params['x'], params['y']
        return np.array([x**2 + 3*x*y + y**3, x - y**2])

    fiducial = {'x': 1.0, 'y': 1.0}
    emu = TaylorEmulator(poly, fiducial, order=3, step_sizes=0.01, verbose=False)

    save_path = tmp_path / "taylor_cache.npz"
    emu.save_coefficients(save_path)

    loaded = TaylorEmulator.from_coefficients(save_path, verbose=False)

    test_points = [
        {'x': 1.5, 'y': 0.5},
        {'x': 0.7, 'y': 1.3},
        {'x': 2.0, 'y': 2.0},
        {'x': 0.0, 'y': 0.0},
    ]
    for p in test_points:
        pred_orig = emu.predict(p)
        pred_loaded = loaded.predict(p)
        np.testing.assert_allclose(pred_loaded, pred_orig, atol=1e-12,
                                   err_msg=f"Mismatch at {p}")


def test_loaded_emulator_predict_only(tmp_path):
    """Loaded emulator has no theory_fn and can still predict."""
    def fn(params):
        return np.array([params['a'] ** 2])

    fiducial = {'a': 1.0}
    emu = TaylorEmulator(fn, fiducial, order=2, step_sizes=0.01, verbose=False)

    save_path = tmp_path / "taylor_cache.npz"
    emu.save_coefficients(save_path)

    loaded = TaylorEmulator.from_coefficients(save_path, verbose=False)
    assert loaded._theory_fn is None

    # Predictions should still work
    pred = loaded.predict({'a': 1.5})
    expected = emu.predict({'a': 1.5})
    np.testing.assert_allclose(pred, expected, atol=1e-12)
