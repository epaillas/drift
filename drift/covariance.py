"""Analytic covariance utilities for power-spectrum multipoles."""

from __future__ import annotations

import itertools
import numpy as np

from .utils.multipoles import legendre


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


def _canonicalize_pair_label(pair) -> tuple[str, str]:
    """Return a canonical symmetric DS-pair label."""
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(
            f"Pair labels must be length-2 tuples/lists, got {pair!r}."
        )
    a, b = str(pair[0]), str(pair[1])
    return tuple(sorted((a, b)))


def _normalize_pair_order(pair_order) -> tuple[tuple[str, str], ...]:
    """Validate and canonicalize the observed DS-pair ordering."""
    if pair_order is None:
        raise ValueError("pair_order must be provided.")
    canonical = tuple(_canonicalize_pair_label(pair) for pair in pair_order)
    if not canonical:
        raise ValueError("pair_order must contain at least one pair.")
    if len(set(canonical)) != len(canonical):
        raise ValueError("pair_order contains duplicate pairs after canonicalization.")
    return canonical


def _normalize_label_order(labels) -> tuple[str, ...]:
    """Validate and canonicalize the observed DS-label ordering."""
    if labels is None:
        raise ValueError("labels must be provided.")
    canonical = tuple(str(label) for label in labels)
    if not canonical:
        raise ValueError("labels must contain at least one entry.")
    if len(set(canonical)) != len(canonical):
        raise ValueError("labels contains duplicate entries.")
    return canonical


def _required_pair_labels(pair_order) -> tuple[tuple[str, str], ...]:
    """Return all DS pairs required by Wick contractions over the observed bins."""
    bins = sorted({label for pair in pair_order for label in pair})
    return tuple((a, b) for a, b in itertools.combinations_with_replacement(bins, 2))


def _coerce_pair_pole_dict(
    poles,
    k: np.ndarray,
    ells,
    pair_order,
) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    """Normalize DS-pair fiducial multipoles into a nested canonical dict."""
    nk = len(k)
    pair_order = _normalize_pair_order(pair_order)
    required_pairs = _required_pair_labels(pair_order)

    if isinstance(poles, dict):
        normalized = {}
        for key, value in poles.items():
            pair = _canonicalize_pair_label(key)
            normalized[pair] = _coerce_pole_dict(value, k, ells)
        missing = [pair for pair in required_pairs if pair not in normalized]
        if missing:
            raise ValueError(
                "Missing fiducial DS-pair multipoles for "
                f"{missing}. Provide all contraction pairs over the bins in pair_order."
            )
        return normalized

    vec = np.asarray(poles, dtype=float)
    expected = len(pair_order) * len(ells) * nk
    if vec.shape != (expected,):
        raise ValueError(
            f"Flat DS-pair multipole vector must have shape ({expected},), got {vec.shape}."
        )
    if required_pairs != pair_order:
        raise ValueError(
            "Flat DS-pair multipole vectors are only supported when pair_order "
            "contains the full canonical pair list over its DS bins."
        )

    normalized = {}
    start = 0
    block_size = len(ells) * nk
    for pair in pair_order:
        normalized[pair] = _coerce_pole_dict(vec[start:start + block_size], k, ells)
        start += block_size
    return normalized


def _coerce_pair_shot_noise(shot_noise, pair_order) -> dict[tuple[str, str], float]:
    """Normalize DS-pair constant noise values into a canonical dict."""
    if not isinstance(shot_noise, dict):
        raise ValueError("shot_noise must be a dict keyed by DS-pair labels.")

    normalized = {}
    for key, value in shot_noise.items():
        pair = _canonicalize_pair_label(key)
        val = float(value)
        if val < 0.0:
            raise ValueError(f"shot_noise for pair {pair} must be non-negative.")
        normalized[pair] = val

    required_pairs = _required_pair_labels(_normalize_pair_order(pair_order))
    missing = [pair for pair in required_pairs if pair not in normalized]
    if missing:
        raise ValueError(
            "Missing DS-pair shot-noise entries for "
            f"{missing}. Provide all contraction pairs over the bins in pair_order."
        )
    return normalized


def _coerce_label_pole_dict(
    poles,
    k: np.ndarray,
    ells,
    label_order,
) -> dict[str, dict[int, np.ndarray]]:
    """Normalize DS-labeled fiducial multipoles into a dict keyed by label."""
    nk = len(k)
    label_order = _normalize_label_order(label_order)

    if isinstance(poles, dict):
        normalized = {}
        for key, value in poles.items():
            normalized[str(key)] = _coerce_pole_dict(value, k, ells)
        missing = [label for label in label_order if label not in normalized]
        if missing:
            raise ValueError(
                f"Missing fiducial multipoles for labels {missing}."
            )
        return normalized

    vec = np.asarray(poles, dtype=float)
    expected = len(label_order) * len(ells) * nk
    if vec.shape != (expected,):
        raise ValueError(
            f"Flat DS-labeled multipole vector must have shape ({expected},), got {vec.shape}."
        )

    normalized = {}
    start = 0
    block_size = len(ells) * nk
    for label in label_order:
        normalized[label] = _coerce_pole_dict(vec[start:start + block_size], k, ells)
        start += block_size
    return normalized


def _coerce_label_shot_noise(shot_noise, label_order) -> dict[str, float]:
    """Normalize DS-labeled constant noise values into a canonical dict."""
    if shot_noise is None:
        return {label: 0.0 for label in _normalize_label_order(label_order)}
    if not isinstance(shot_noise, dict):
        raise ValueError("shot_noise must be a dict keyed by DS labels.")

    normalized = {}
    for key, value in shot_noise.items():
        label = str(key)
        val = float(value)
        if val < 0.0:
            raise ValueError(f"shot_noise for label {label!r} must be non-negative.")
        normalized[label] = val

    required = _normalize_label_order(label_order)
    missing = [label for label in required if label not in normalized]
    if missing:
        raise ValueError(
            f"Missing DS-label shot-noise entries for {missing}."
        )
    return normalized


def _reconstruct_pkmu(
    k: np.ndarray,
    pole_dict: dict[int, np.ndarray],
    ells,
    mu: np.ndarray,
    L: dict[int, np.ndarray],
) -> np.ndarray:
    """Reconstruct P(k,mu) from fiducial multipoles."""
    pkmu = np.zeros((len(k), len(mu)), dtype=float)
    for ell in ells:
        pkmu += pole_dict[ell][:, None] * L[ell][None, :]
    return pkmu


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
    pkmu = _reconstruct_pkmu(k, pole_dict, ells, mu, L)
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


def _gaussian_dspair_covariance(
    k: np.ndarray,
    pair_poles: dict[tuple[str, str], dict[int, np.ndarray]],
    ells,
    pair_order,
    *,
    shot_noise: dict[tuple[str, str], float],
    volume: float,
    rescale: float,
    mu_points: int,
) -> np.ndarray:
    """Gaussian disconnected cubic-box covariance for DS-pair multipoles."""
    mu, weights = np.polynomial.legendre.leggauss(mu_points)
    L = {ell: legendre(ell, mu) for ell in ells}
    pair_order = _normalize_pair_order(pair_order)
    nmodes = _nmodes_cubic_box(k, volume)
    nk = len(k)
    nells = len(ells)
    npairs = len(pair_order)
    cov = np.zeros((npairs * nells * nk, npairs * nells * nk), dtype=float)

    pkmu = {
        pair: _reconstruct_pkmu(k, pair_poles[pair], ells, mu, L)
        for pair in pair_poles
    }
    total_power = {
        pair: pkmu[pair] + shot_noise[pair]
        for pair in pair_poles
    }

    for p_idx, pair_ab in enumerate(pair_order):
        a, b = pair_ab
        for q_idx, pair_cd in enumerate(pair_order):
            c, d = pair_cd
            pair_ac = _canonicalize_pair_label((a, c))
            pair_bd = _canonicalize_pair_label((b, d))
            pair_ad = _canonicalize_pair_label((a, d))
            pair_bc = _canonicalize_pair_label((b, c))
            wick_sum = (
                total_power[pair_ac] * total_power[pair_bd]
                + total_power[pair_ad] * total_power[pair_bc]
            )

            for i, ell_a in enumerate(ells):
                row = slice(
                    (p_idx * nells + i) * nk,
                    (p_idx * nells + i + 1) * nk,
                )
                for j, ell_b in enumerate(ells):
                    col = slice(
                        (q_idx * nells + j) * nk,
                        (q_idx * nells + j + 1) * nk,
                    )
                    prefactor = (2 * ell_a + 1) * (2 * ell_b + 1) / nmodes
                    integrand = L[ell_a][None, :] * L[ell_b][None, :] * wick_sum
                    block_diag = prefactor * (integrand @ weights)
                    cov[row, col] = np.diag(block_diag / rescale)

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


def _gaussian_dsg_covariance(
    k: np.ndarray,
    pqg_poles: dict[str, dict[int, np.ndarray]],
    pqq_poles: dict[tuple[str, str], dict[int, np.ndarray]],
    pgg_poles: dict[int, np.ndarray],
    ells,
    label_order,
    *,
    ds_cross_shot_noise: dict[str, float],
    ds_pair_shot_noise: dict[tuple[str, str], float],
    galaxy_shot_noise: float,
    volume: float,
    rescale: float,
    mu_points: int,
) -> np.ndarray:
    """Gaussian disconnected cubic-box covariance for DS-galaxy multipoles."""
    mu, weights = np.polynomial.legendre.leggauss(mu_points)
    L = {ell: legendre(ell, mu) for ell in ells}
    label_order = _normalize_label_order(label_order)
    nmodes = _nmodes_cubic_box(k, volume)
    nk = len(k)
    nells = len(ells)
    nlabels = len(label_order)
    cov = np.zeros((nlabels * nells * nk, nlabels * nells * nk), dtype=float)

    pqg_pkmu = {
        label: _reconstruct_pkmu(k, pqg_poles[label], ells, mu, L)
        for label in label_order
    }
    pqq_pkmu = {
        pair: _reconstruct_pkmu(k, pqq_poles[pair], ells, mu, L)
        for pair in pqq_poles
    }
    pgg_pkmu = _reconstruct_pkmu(k, pgg_poles, ells, mu, L)

    pqg_total = {
        label: pqg_pkmu[label] + ds_cross_shot_noise[label]
        for label in label_order
    }
    pqq_total = {
        pair: pqq_pkmu[pair] + ds_pair_shot_noise[pair]
        for pair in pqq_poles
    }
    pgg_total = pgg_pkmu + galaxy_shot_noise

    for q_idx, label_a in enumerate(label_order):
        for r_idx, label_b in enumerate(label_order):
            pair_ab = _canonicalize_pair_label((label_a, label_b))
            wick_sum = (
                pqq_total[pair_ab] * pgg_total
                + pqg_total[label_a] * pqg_total[label_b]
            )

            for i, ell_a in enumerate(ells):
                row = slice(
                    (q_idx * nells + i) * nk,
                    (q_idx * nells + i + 1) * nk,
                )
                for j, ell_b in enumerate(ells):
                    col = slice(
                        (r_idx * nells + j) * nk,
                        (r_idx * nells + j + 1) * nk,
                    )
                    prefactor = (2 * ell_a + 1) * (2 * ell_b + 1) / nmodes
                    integrand = L[ell_a][None, :] * L[ell_b][None, :] * wick_sum
                    block_diag = prefactor * (integrand @ weights)
                    cov[row, col] = np.diag(block_diag / rescale)

    return cov


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


def analytic_pqq_covariance(
    k: np.ndarray,
    poles,
    ells=(0, 2, 4),
    *,
    volume: float,
    pair_order,
    shot_noise,
    mask: np.ndarray | None = None,
    rescale: float = 1.0,
    terms: str = "gaussian",
    mu_points: int = 256,
):
    """Gaussian cubic-box covariance for density-split pair multipoles."""
    term_labels = _normalize_terms(terms)
    if any(term != "gaussian" for term in term_labels):
        raise NotImplementedError(
            "Beyond-Gaussian density-split pair covariance terms are not yet implemented."
        )
    if rescale <= 0.0:
        raise ValueError("rescale must be positive.")

    k = np.asarray(k, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    pair_order = _normalize_pair_order(pair_order)
    pair_poles = _coerce_pair_pole_dict(poles, k, ells, pair_order)
    pair_shot = _coerce_pair_shot_noise(shot_noise, pair_order)

    cov = _gaussian_dspair_covariance(
        k,
        pair_poles,
        ells,
        pair_order,
        shot_noise=pair_shot,
        volume=volume,
        rescale=rescale,
        mu_points=mu_points,
    )

    if mask is not None:
        cov = cov[np.ix_(mask, mask)]

    precision = np.linalg.inv(cov)
    return cov, precision


def analytic_pqg_covariance(
    k: np.ndarray,
    pqg_poles,
    pqq_poles,
    pgg_poles,
    ells=(0, 2, 4),
    *,
    volume: float,
    ds_labels,
    galaxy_shot_noise: float,
    ds_pair_shot_noise,
    ds_cross_shot_noise=None,
    mask: np.ndarray | None = None,
    rescale: float = 1.0,
    terms: str = "gaussian",
    mu_points: int = 256,
):
    """Gaussian cubic-box covariance for density-split-galaxy multipoles."""
    term_labels = _normalize_terms(terms)
    if any(term != "gaussian" for term in term_labels):
        raise NotImplementedError(
            "Beyond-Gaussian density-split-galaxy covariance terms are not yet implemented."
        )
    if rescale <= 0.0:
        raise ValueError("rescale must be positive.")
    if galaxy_shot_noise < 0.0:
        raise ValueError("galaxy_shot_noise must be non-negative.")

    k = np.asarray(k, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    ds_labels = _normalize_label_order(ds_labels)
    pair_order = tuple(
        itertools.combinations_with_replacement(ds_labels, 2)
    )
    pqg_pole_dict = _coerce_label_pole_dict(pqg_poles, k, ells, ds_labels)
    pqq_pole_dict = _coerce_pair_pole_dict(pqq_poles, k, ells, pair_order)
    pgg_pole_dict = _coerce_pole_dict(pgg_poles, k, ells)
    ds_pair_shot = _coerce_pair_shot_noise(ds_pair_shot_noise, pair_order)
    ds_cross_shot = _coerce_label_shot_noise(ds_cross_shot_noise, ds_labels)

    cov = _gaussian_dsg_covariance(
        k,
        pqg_pole_dict,
        pqq_pole_dict,
        pgg_pole_dict,
        ells,
        ds_labels,
        ds_cross_shot_noise=ds_cross_shot,
        ds_pair_shot_noise=ds_pair_shot,
        galaxy_shot_noise=float(galaxy_shot_noise),
        volume=volume,
        rescale=rescale,
        mu_points=mu_points,
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
