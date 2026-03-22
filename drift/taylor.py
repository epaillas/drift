"""Taylor expansion emulator for arbitrary theory callables.

Wraps any function mapping parameter dicts to 1D arrays,
computes numerical derivatives via finite differences at a fiducial point,
and predicts via multivariate Taylor expansion.
"""

import numpy as np
from itertools import combinations_with_replacement
from math import factorial


def _fornberg_weights(center, stencil_points, max_deriv):
    """Compute finite difference weights using Fornberg's algorithm.

    Parameters
    ----------
    center : float
        Point at which to approximate the derivative.
    stencil_points : array-like
        Grid points used in the stencil.
    max_deriv : int
        Maximum derivative order to compute.

    Returns
    -------
    weights : np.ndarray
        Shape (max_deriv + 1, n_points). weights[d] gives the stencil
        weights for the d-th derivative.
    """
    n = len(stencil_points)
    x = np.asarray(stencil_points, dtype=float)
    C = np.zeros((max_deriv + 1, n))
    C[0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            for k in range(min(i, max_deriv), 0, -1):
                C[k, i] = c1 * (k * C[k - 1, i - 1] - (x[i - 1] - center) * C[k, i - 1]) / c2
            C[0, i] = -c1 * (x[i - 1] - center) * C[0, i - 1] / c2
            for k in range(min(i, max_deriv), 0, -1):
                C[k, j] = ((x[i] - center) * C[k, j] - k * C[k - 1, j]) / c3
            C[0, j] = (x[i] - center) * C[0, j] / c3
        c1 = c2
    return C


def _central_stencil(deriv_order, accuracy=2):
    """Return (offsets, weights) for a 1D central finite difference stencil.

    Parameters
    ----------
    deriv_order : int
        Order of derivative (1, 2, 3, ...).
    accuracy : int
        Order of accuracy (default 2).

    Returns
    -------
    offsets : np.ndarray of int
        Integer offsets from center.
    weights : np.ndarray
        Corresponding weights.
    """
    # Number of points on each side of center
    half = (deriv_order + accuracy - 1) // 2 + (deriv_order + accuracy) % 2
    # Ensure enough points for the derivative order
    half = max(half, (deriv_order + 1) // 2 + 1)
    offsets = np.arange(-half, half + 1)
    W = _fornberg_weights(0.0, offsets.astype(float), deriv_order)
    weights = W[deriv_order]
    return offsets, weights


def _enumerate_multi_indices(n_params, max_order):
    """Generate all multi-index tuples alpha with |alpha| <= max_order.

    Parameters
    ----------
    n_params : int
        Number of parameters.
    max_order : int
        Maximum total order.

    Returns
    -------
    list of tuples
        Each tuple has length n_params with non-negative integer entries.
    """
    if n_params == 0:
        return [()]
    result = []
    # Use recursive approach
    def _recurse(depth, remaining, current):
        if depth == n_params - 1:
            result.append(tuple(current) + (remaining,))
            for r in range(remaining - 1, -1, -1):
                result.append(tuple(current) + (r,))
            return
        for k in range(remaining, -1, -1):
            _recurse(depth + 1, remaining - k, current + [k])

    # Simpler iterative approach
    result = []
    if n_params == 1:
        for k in range(max_order + 1):
            result.append((k,))
        return result

    def _gen(depth, remaining, current):
        if depth == n_params - 1:
            for k in range(remaining + 1):
                result.append(tuple(current) + (k,))
            return
        for k in range(remaining + 1):
            _gen(depth + 1, remaining - k, current + [k])

    _gen(0, max_order, [])
    return result


class TaylorEmulator:
    """Taylor expansion emulator for arbitrary theory functions.

    Wraps a callable that maps a parameter dict to a 1D numpy array,
    computes numerical derivatives via finite differences at a fiducial
    parameter point, and evaluates predictions via multivariate Taylor
    expansion.

    Parameters
    ----------
    theory_fn : callable
        Function mapping dict -> np.ndarray (1D output).
    fiducial : dict
        Fiducial parameter values {param_name: value}.
    order : int
        Maximum Taylor expansion order (default 4).
    step_sizes : float or dict
        If float, treated as relative fraction: h_i = step_sizes * |fiducial_i|
        (falls back to step_sizes as absolute if fiducial_i == 0).
        If dict, absolute step sizes per parameter.
    param_names : list of str or None
        Subset of fiducial keys to expand over. None means all keys.
    verbose : bool
        Print progress information.
    """

    def __init__(self, theory_fn, fiducial, order=4, step_sizes=0.01,
                 param_names=None, verbose=True):
        self._theory_fn = theory_fn
        self._fiducial = dict(fiducial)
        self._order = order
        self._verbose = verbose

        if param_names is None:
            self._param_names = list(fiducial.keys())
        else:
            self._param_names = list(param_names)

        self._n_params = len(self._param_names)

        # Compute step sizes
        self._step_sizes = {}
        if isinstance(step_sizes, dict):
            for name in self._param_names:
                self._step_sizes[name] = step_sizes[name]
        else:
            for name in self._param_names:
                fid_val = self._fiducial[name]
                if fid_val == 0:
                    self._step_sizes[name] = step_sizes
                else:
                    self._step_sizes[name] = step_sizes * abs(fid_val)

        # Precompute stencils for each derivative order
        self._stencils = {}
        for d in range(1, order + 1):
            self._stencils[d] = _central_stencil(d)

        # Build multi-indices and compute coefficients
        self._multi_indices = _enumerate_multi_indices(self._n_params, order)
        self._eval_cache = {}
        self._build_coefficients()

    def _eval_at_offset(self, offset_tuple):
        """Evaluate theory_fn at fiducial + offset, with caching.

        Parameters
        ----------
        offset_tuple : tuple of int
            Integer offsets for each parameter (multiplied by step size).

        Returns
        -------
        np.ndarray
        """
        if offset_tuple in self._eval_cache:
            return self._eval_cache[offset_tuple]

        params = dict(self._fiducial)
        for i, name in enumerate(self._param_names):
            params[name] = self._fiducial[name] + offset_tuple[i] * self._step_sizes[name]

        result = np.asarray(self._theory_fn(params))
        self._eval_cache[offset_tuple] = result
        return result

    def _compute_derivative(self, multi_index):
        """Compute the mixed partial derivative for a given multi-index.

        Uses successive application of 1D finite difference stencils
        along each active axis via tensor product.

        Parameters
        ----------
        multi_index : tuple of int
            Derivative orders for each parameter.

        Returns
        -------
        np.ndarray
            The derivative D^alpha f evaluated at the fiducial point.
        """
        # Find active dimensions (non-zero derivative orders)
        active_dims = [(i, multi_index[i]) for i in range(self._n_params)
                       if multi_index[i] > 0]

        if not active_dims:
            # Zeroth derivative = function value
            offset = tuple(0 for _ in range(self._n_params))
            return self._eval_at_offset(offset)

        # Build tensor product of stencils
        # Start with identity (single point at origin)
        # Each active dimension expands the set of evaluation points
        # via outer product of stencils

        # Represent current state as list of (offset_tuple, weight) pairs
        base_offset = [0] * self._n_params
        points = [(tuple(base_offset), 1.0)]

        for dim_idx, deriv_ord in active_dims:
            offsets_1d, weights_1d = self._stencils[deriv_ord]
            h = self._step_sizes[self._param_names[dim_idx]]
            # Scale weights by 1/h^deriv_ord
            scale = 1.0 / (h ** deriv_ord)

            new_points = []
            for base_off, base_w in points:
                for s, w in zip(offsets_1d, weights_1d):
                    if w == 0.0:
                        continue
                    new_off = list(base_off)
                    new_off[dim_idx] += int(s)
                    new_points.append((tuple(new_off), base_w * w * scale))
            points = new_points

        # Consolidate duplicate offsets
        consolidated = {}
        for off, w in points:
            if off in consolidated:
                consolidated[off] += w
            else:
                consolidated[off] = w

        # Evaluate and sum
        result = None
        for off, w in consolidated.items():
            val = self._eval_at_offset(off)
            if result is None:
                result = w * val
            else:
                result = result + w * val

        return result

    def _build_coefficients(self):
        """Compute Taylor coefficients for all multi-indices."""
        if self._verbose:
            print(f"TaylorEmulator: building {len(self._multi_indices)} "
                  f"coefficients for {self._n_params} parameters at order "
                  f"{self._order}")

        # Get output dimension from fiducial evaluation
        fid_val = self._eval_at_offset(tuple(0 for _ in range(self._n_params)))
        n_output = len(fid_val)

        self._coefficients = np.zeros((len(self._multi_indices), n_output))

        for i, alpha in enumerate(self._multi_indices):
            deriv = self._compute_derivative(alpha)
            # Divide by alpha! = product of factorials
            alpha_factorial = 1
            for a in alpha:
                alpha_factorial *= factorial(a)
            self._coefficients[i] = deriv / alpha_factorial

        if self._verbose:
            print(f"TaylorEmulator: done. {self.n_evals} theory evaluations, "
                  f"{self.n_terms} terms.")

    def predict(self, params):
        """Evaluate the Taylor expansion at the given parameters.

        Parameters
        ----------
        params : dict
            Parameter values {param_name: value}.

        Returns
        -------
        np.ndarray
            Predicted output (1D array).
        """
        dp = np.array([params[name] - self._fiducial[name]
                        for name in self._param_names])

        # Compute monomial vector: product of dp_i^alpha_i for each multi-index
        monomials = np.empty(len(self._multi_indices))
        for i, alpha in enumerate(self._multi_indices):
            m = 1.0
            for j, a in enumerate(alpha):
                if a > 0:
                    m *= dp[j] ** a
            monomials[i] = m

        return monomials @ self._coefficients

    def save_coefficients(self, path):
        """Save emulator state to a .npz file for later reuse.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path (will add .npz extension if absent).
        """
        param_names = np.array(self._param_names)
        fiducial_values = np.array([self._fiducial[n] for n in self._param_names])
        step_sizes_values = np.array([self._step_sizes[n] for n in self._param_names])
        np.savez(
            path,
            coefficients=self._coefficients,
            param_names=param_names,
            fiducial_values=fiducial_values,
            order=np.array(self._order),
            step_sizes_values=step_sizes_values,
        )

    @classmethod
    def from_coefficients(cls, path, verbose=True):
        """Load a cached emulator in predict-only mode.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the .npz file saved by ``save_coefficients``.
        verbose : bool
            Print a message on load.

        Returns
        -------
        TaylorEmulator
            An emulator that can ``predict`` but has no ``theory_fn``.
        """
        data = np.load(path, allow_pickle=False)
        param_names = list(data["param_names"])
        fiducial_values = data["fiducial_values"]
        order = int(data["order"])
        step_sizes_values = data["step_sizes_values"]
        coefficients = data["coefficients"]

        obj = cls.__new__(cls)
        obj._theory_fn = None
        obj._param_names = param_names
        obj._n_params = len(param_names)
        obj._order = order
        obj._fiducial = {name: float(val) for name, val in zip(param_names, fiducial_values)}
        obj._step_sizes = {name: float(val) for name, val in zip(param_names, step_sizes_values)}
        obj._multi_indices = _enumerate_multi_indices(obj._n_params, order)
        obj._coefficients = coefficients
        obj._eval_cache = {}
        obj._verbose = verbose

        if verbose:
            print(f"TaylorEmulator: loaded {len(obj._multi_indices)} terms "
                  f"for {obj._n_params} parameters at order {order} from {path}")
        return obj

    @property
    def n_terms(self):
        """Number of Taylor expansion terms."""
        return len(self._multi_indices)

    @property
    def n_evals(self):
        """Number of unique theory_fn evaluations performed."""
        return len(self._eval_cache)
