"""Analytic covariance utilities for power-spectrum multipoles."""

from __future__ import annotations

import numpy as np

from .multipoles import legendre


def _bin_widths_from_centers(k: np.ndarray) -> np.ndarray:
    """Infer bin widths from a 1-D array of bin centers."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("k must be a non-empty 1-D array.")
    if k.size == 1:
        raise ValueError("At least two k bins are required to infer bin widths.")

    edges = np.empty(k.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (k[1:] + k[:-1])
    edges[0] = k[0] - 0.5 * (k[1] - k[0])
    edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("Inferred k-bin widths must be positive.")
    return widths


def _nmodes_cubic_box(k: np.ndarray, volume: float) -> np.ndarray:
    """Return the number of Fourier modes in each shell of a cubic box."""
    if volume <= 0.0:
        raise ValueError("volume must be positive.")
    delta_k = _bin_widths_from_centers(k)
    return volume * k ** 2 * delta_k / (2.0 * np.pi ** 2)


def _coerce_pole_dict(poles, k: np.ndarray, ells) -> dict[int, np.ndarray]:
    """Normalize multipoles into a {ell: array(nk)} dictionary."""
    nk = len(k)
    if isinstance(poles, dict):
        out = {}
        for ell in ells:
            if ell not in poles:
                raise ValueError(f"Missing fiducial multipole ell={ell}.")
            arr = np.asarray(poles[ell], dtype=float)
            if arr.shape != (nk,):
                raise ValueError(f"Multipole ell={ell} must have shape ({nk},).")
            out[int(ell)] = arr
        return out

    vec = np.asarray(poles, dtype=float)
    if vec.shape != (len(ells) * nk,):
        raise ValueError(
            f"Flat multipole vector must have shape ({len(ells) * nk},), got {vec.shape}."
        )
    out = {}
    start = 0
    for ell in ells:
        out[int(ell)] = vec[start:start + nk]
        start += nk
    return out


def _normalize_terms(terms: str) -> tuple[str, ...]:
    """Return a normalized tuple of covariance term labels."""
    aliases = {
        "gaussian_only": "gaussian",
        "connected": "effective_cng",
        "cng": "effective_cng",
        "non_gaussian": "effective_cng",
    }
    normalized = []
    for token in str(terms).lower().replace(" ", "").split("+"):
        if not token:
            continue
        normalized.append(aliases.get(token, token))
    if not normalized:
        raise ValueError("At least one covariance term must be provided.")

    allowed = {"gaussian", "effective_cng"}
    invalid = [token for token in normalized if token not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported covariance terms={terms!r}. "
            "Supported terms are 'gaussian' and 'effective_cng'."
        )

    deduped = []
    for token in normalized:
        if token not in deduped:
            deduped.append(token)
    if "gaussian" not in deduped:
        deduped.insert(0, "gaussian")
    return tuple(deduped)


def _effective_cng_ell_weights(ells) -> np.ndarray:
    """Down-weight higher multipoles in the connected covariance correction."""
    ells = tuple(int(ell) for ell in ells)
    return np.array([1.0 / np.sqrt(1.0 + 0.5 * ell) for ell in ells], dtype=float)


def _rbf_kernel_logk(k: np.ndarray, coherence: float) -> np.ndarray:
    """Smooth positive-semidefinite kernel in log-k."""
    if coherence <= 0.0:
        raise ValueError("cng_coherence must be positive.")
    logk = np.log(np.asarray(k, dtype=float))
    dist2 = (logk[:, None] - logk[None, :]) ** 2
    return np.exp(-0.5 * dist2 / coherence ** 2)


def _gaussian_covariance(
    k: np.ndarray,
    pole_dict: dict[int, np.ndarray],
    ells,
    *,
    shot: float,
    volume: float,
    rescale: float,
    mu_points: int,
) -> np.ndarray:
    """Gaussian disconnected cubic-box covariance."""
    mu, weights = np.polynomial.legendre.leggauss(mu_points)
    L = {ell: legendre(ell, mu) for ell in ells}
    pkmu = np.zeros((len(k), len(mu)), dtype=float)
    for ell in ells:
        pkmu += pole_dict[ell][:, None] * L[ell][None, :]
    total_power = pkmu + shot

    nmodes = _nmodes_cubic_box(k, volume)
    nk = len(k)
    nells = len(ells)
    cov = np.zeros((nells * nk, nells * nk), dtype=float)

    for i, ell_a in enumerate(ells):
        for j, ell_b in enumerate(ells):
            prefactor = (2 * ell_a + 1) * (2 * ell_b + 1) / nmodes
            integrand = L[ell_a][None, :] * L[ell_b][None, :] * total_power ** 2
            block_diag = prefactor * (integrand @ weights)
            block = np.diag(block_diag / rescale)
            row = slice(i * nk, (i + 1) * nk)
            col = slice(j * nk, (j + 1) * nk)
            cov[row, col] = block

    return cov


def _effective_cng_covariance(
    k: np.ndarray,
    gaussian_cov: np.ndarray,
    ells,
    *,
    amplitude: float,
    coherence: float,
) -> np.ndarray:
    """Phenomenological connected non-Gaussian box correction.

    This preserves the Gaussian diagonal scaling but introduces broad
    off-diagonal mode coupling in k and suppresses higher multipoles.
    """
    if amplitude < 0.0:
        raise ValueError("cng_amplitude must be non-negative.")
    if amplitude == 0.0:
        return np.zeros_like(gaussian_cov)

    k = np.asarray(k, dtype=float)
    nk = len(k)
    kernel_k = _rbf_kernel_logk(k, coherence)
    ell_weights = _effective_cng_ell_weights(ells)

    sigma = np.sqrt(np.clip(np.diag(gaussian_cov), 0.0, None))
    nells = len(ells)
    response = np.empty_like(sigma)
    for i, weight in enumerate(ell_weights):
        sl = slice(i * nk, (i + 1) * nk)
        response[sl] = weight * sigma[sl]

    kernel = np.zeros_like(gaussian_cov)
    for i in range(nells):
        row = slice(i * nk, (i + 1) * nk)
        for j in range(nells):
            col = slice(j * nk, (j + 1) * nk)
            kernel[row, col] = kernel_k

    return amplitude * np.outer(response, response) * kernel


def analytic_pgg_covariance(
    k: np.ndarray,
    poles,
    ells=(0, 2, 4),
    *,
    volume: float,
    number_density: float | None = None,
    shot_noise: float | None = None,
    mask: np.ndarray | None = None,
    rescale: float = 1.0,
    terms: str = "gaussian",
    mu_points: int = 256,
    cng_amplitude: float = 0.0,
    cng_coherence: float = 0.35,
):
    """Gaussian cubic-box covariance for galaxy power-spectrum multipoles.

    Parameters
    ----------
    k : np.ndarray, shape (nk,)
        Bin centers in h/Mpc.
    poles : dict or np.ndarray
        Fiducial multipoles either as ``{ell: P_ell(k)}`` or a flat vector with
        the same ordering used in inference, i.e. ``[P_0, P_2, ...]`` blocks.
    ells : tuple of int
        Multipoles included in the data vector.
    volume : float
        Survey/box volume in (Mpc/h)^3.
    number_density : float, optional
        Number density in (h/Mpc)^3. Mutually exclusive with ``shot_noise``.
    shot_noise : float, optional
        Constant shot-noise power in (Mpc/h)^3. Mutually exclusive with
        ``number_density``.
    mask : np.ndarray of bool, optional
        Boolean mask applied to the flattened covariance.
    rescale : float, default 1.0
        Divide covariance by this factor.
    terms : str, default "gaussian"
        Covariance terms to include. Supported values are ``"gaussian"`` and
        ``"gaussian+effective_cng"``.
    mu_points : int, default 256
        Number of Gauss-Legendre nodes for the mu integral.
    cng_amplitude : float, default 0.0
        Amplitude of the phenomenological connected non-Gaussian correction.
        Ignored unless ``terms`` includes ``effective_cng``.
    cng_coherence : float, default 0.35
        Correlation length of the effective connected correction in ``ln k``.
    """
    term_labels = _normalize_terms(terms)

    if rescale <= 0.0:
        raise ValueError("rescale must be positive.")
    if (number_density is None) == (shot_noise is None):
        raise ValueError("Provide exactly one of number_density or shot_noise.")
    if number_density is not None and number_density <= 0.0:
        raise ValueError("number_density must be positive.")
    if shot_noise is not None and shot_noise < 0.0:
        raise ValueError("shot_noise must be non-negative.")

    k = np.asarray(k, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    pole_dict = _coerce_pole_dict(poles, k, ells)
    shot = (1.0 / float(number_density)) if number_density is not None else float(shot_noise)
    cov = _gaussian_covariance(
        k,
        pole_dict,
        ells,
        shot=shot,
        volume=volume,
        rescale=rescale,
        mu_points=mu_points,
    )
    if "effective_cng" in term_labels:
        cov = cov + _effective_cng_covariance(
            k,
            cov,
            ells,
            amplitude=cng_amplitude,
            coherence=cng_coherence,
        )

    if mask is not None:
        cov = cov[np.ix_(mask, mask)]

    precision = np.linalg.inv(cov)
    return cov, precision


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    sigma = np.sqrt(np.diag(cov))
    denom = np.outer(sigma, sigma)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def multipole_block_edges(k: np.ndarray | None = None, ells=None, block_sizes=None) -> list[int]:
    """Return flattened block boundaries for per-ell matrix annotations."""
    if block_sizes is not None:
        edges = [0]
        total = 0
        for size in block_sizes:
            total += int(size)
            edges.append(total)
        return edges

    if k is None or ells is None:
        raise ValueError("Provide either block_sizes or both k and ells.")
    nk = len(np.asarray(k))
    return [i * nk for i in range(len(tuple(ells)) + 1)]


def plot_correlation_matrix(
    cov_or_corr: np.ndarray,
    *,
    k: np.ndarray | None = None,
    ells=None,
    block_sizes=None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    colorbar: bool = True,
):
    """Plot a covariance or correlation matrix with optional ell boundaries."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(cov_or_corr, dtype=float)
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, atol=1e-8, rtol=1e-8):
        matrix = correlation_matrix(matrix)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xlabel("data bin")
    ax.set_ylabel("data bin")
    if title is not None:
        ax.set_title(title)

    if ells is not None and (k is not None or block_sizes is not None):
        edges = multipole_block_edges(k=k, ells=ells, block_sizes=block_sizes)
        centers = [0.5 * (start + stop - 1) for start, stop in zip(edges[:-1], edges[1:])]
        for edge in edges[1:-1]:
            ax.axvline(edge - 0.5, color="k", lw=0.6, alpha=0.6)
            ax.axhline(edge - 0.5, color="k", lw=0.6, alpha=0.6)
        ax.set_xticks(centers, [rf"$\ell={ell}$" for ell in ells], rotation=45, ha="right")
        ax.set_yticks(centers, [rf"$\ell={ell}$" for ell in ells])

    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    return fig, ax
