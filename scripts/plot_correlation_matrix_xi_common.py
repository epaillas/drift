"""Shared helpers for configuration-space correlation-matrix scripts."""

from __future__ import annotations

import numpy as np


DEFAULT_S_STEP = 5.0


def default_s_limits(k):
    """Return the default reliable s-range implied by the k grid."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("k must be a non-empty one-dimensional array.")
    if np.any(k <= 0.0):
        raise ValueError("k must be strictly positive.")
    return 3.0 / float(np.max(k)), 0.7 / float(np.min(k))


def implied_k_limits(*, smin=None, smax=None):
    """Return the rough reciprocal k-range implied by an s-range."""
    kmin = None if smax is None else 0.7 / float(smax)
    kmax = None if smin is None else 3.0 / float(smin)
    return kmin, kmax


def apply_reciprocal_analytic_k_limits(args):
    """Fill missing analytic k cuts from the provided s-range heuristic."""
    analytic_kmin = getattr(args, "analytic_kmin", None)
    analytic_kmax = getattr(args, "analytic_kmax", None)
    implied_kmin, implied_kmax = implied_k_limits(
        smin=getattr(args, "smin", None),
        smax=getattr(args, "smax", None),
    )
    if analytic_kmin is None:
        analytic_kmin = implied_kmin
    if analytic_kmax is None:
        analytic_kmax = implied_kmax
    return analytic_kmin, analytic_kmax


def build_s_grid(k, *, smin=None, smax=None, ds=None, ns=None):
    """Build a linear separation grid for xi covariance plots."""
    default_smin, default_smax = default_s_limits(k)
    if smin is None:
        smin = default_smin
    if smax is None:
        smax = default_smax
    smin = float(smin)
    smax = float(smax)
    if smax <= smin:
        raise ValueError("Require smax > smin.")

    if ds is not None and ns is not None:
        raise ValueError("Provide at most one of ds and ns.")
    if ds is None and ns is None:
        ds = DEFAULT_S_STEP

    if ns is not None:
        ns = int(ns)
        if ns < 2:
            raise ValueError("ns must be at least 2.")
        return np.linspace(smin, smax, ns)

    ds = float(ds)
    if ds <= 0.0:
        raise ValueError("ds must be positive.")
    n = int(np.floor((smax - smin) / ds + 1e-12)) + 1
    s = smin + np.arange(n) * ds
    if s[-1] < smax - 1e-9:
        s = np.append(s, smax)
    return s
