"""Tests for drift.kernels."""

import numpy as np
import pytest
from drift.kernels import gaussian_kernel, tophat_kernel


def test_gaussian_at_zero():
    assert gaussian_kernel(np.array([0.0]), R=10.0)[0] == pytest.approx(1.0)


def test_gaussian_decay():
    k_large = np.array([1e3])
    assert gaussian_kernel(k_large, R=10.0)[0] == pytest.approx(0.0, abs=1e-10)


def test_tophat_at_zero():
    assert tophat_kernel(np.array([0.0]), R=10.0)[0] == pytest.approx(1.0)


def test_tophat_normalization():
    # Very small k should give ~1
    k_small = np.array([1e-10])
    assert tophat_kernel(k_small, R=10.0)[0] == pytest.approx(1.0, rel=1e-6)


def test_gaussian_shape():
    k = np.linspace(0, 1, 50)
    W = gaussian_kernel(k, R=5.0)
    assert W.shape == (50,)
    assert np.all(W >= 0)
    assert np.all(W <= 1.0)


def test_tophat_shape():
    k = np.linspace(0.01, 1, 50)
    W = tophat_kernel(k, R=5.0)
    assert W.shape == (50,)
