"""Analytic marginalization of linear nuisance parameters.

For a Gaussian likelihood with model d = m(theta_nl) + T(theta_nl) @ alpha,
where alpha are linear parameters with Gaussian prior alpha ~ N(0, S),
the marginalized log-likelihood integrates out alpha analytically.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class MarginalizedLikelihood:
    """Gaussian likelihood with linear parameters analytically marginalized.

    Parameters
    ----------
    data : np.ndarray, shape (n_data,)
        Observed data vector.
    precision_matrix : np.ndarray, shape (n_data, n_data)
        Inverse covariance matrix C^{-1}.
    prior_sigmas : np.ndarray, shape (n_linear,)
        Standard deviations of the Gaussian prior on each linear parameter.
        Large values approximate flat priors.
    """

    def __init__(self, data, precision_matrix, prior_sigmas):
        self.data = np.asarray(data, dtype=float)
        self.Cinv = np.asarray(precision_matrix, dtype=float)
        self.prior_sigmas = np.asarray(prior_sigmas, dtype=float)
        self.Sinv = np.diag(1.0 / self.prior_sigmas ** 2)
        self.log_det_Sinv = np.sum(np.log(1.0 / self.prior_sigmas ** 2))

        # Precompute Cinv @ data
        self.Cinv_d = self.Cinv @ self.data

    def __call__(self, m, T):
        """Evaluate the marginalized log-likelihood.

        Parameters
        ----------
        m : np.ndarray, shape (n_data,)
            Nonlinear model piece (linear params set to 0).
        T : np.ndarray, shape (n_data, n_linear)
            Template matrix for linear parameters.

        Returns
        -------
        float
            Marginalized log-likelihood value.
        """
        r = self.data - m
        Cinv_r = self.Cinv @ r

        # A = T^T C^{-1} T + S^{-1}
        Cinv_T = self.Cinv @ T
        A = T.T @ Cinv_T + self.Sinv

        # v = T^T C^{-1} r
        v = T.T @ Cinv_r

        # Cholesky decomposition of A (small n_linear x n_linear matrix)
        cho_A = cho_factor(A)
        log_det_A = 2.0 * np.sum(np.log(np.diag(cho_A[0])))

        # Marginalized log-likelihood:
        # -2 log L = r^T C^{-1} r - v^T A^{-1} v + log|A| - log|S^{-1}| + const
        # We drop constant terms (n_linear * log(2*pi)) as they don't affect sampling
        chi2_full = r @ Cinv_r
        Ainv_v = cho_solve(cho_A, v)
        chi2_marg = chi2_full - v @ Ainv_v

        return -0.5 * (chi2_marg + log_det_A - self.log_det_Sinv)

    def bestfit_linear_params(self, m, T):
        """Recover MAP estimates of linear parameters.

        Parameters
        ----------
        m : np.ndarray, shape (n_data,)
        T : np.ndarray, shape (n_data, n_linear)

        Returns
        -------
        alpha_MAP : np.ndarray, shape (n_linear,)
        """
        r = self.data - m
        Cinv_T = self.Cinv @ T
        A = T.T @ Cinv_T + self.Sinv
        v = T.T @ (self.Cinv @ r)
        return cho_solve(cho_factor(A), v)
