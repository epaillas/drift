"""Save and load DRIFT predictions."""

import hashlib
from pathlib import Path

import numpy as np


def save_predictions(path, k: np.ndarray, multipoles_per_bin: dict) -> None:
    """Save multipole predictions to a compressed .npz file.

    Parameters
    ----------
    path : str or Path
        Output file path (will add .npz if absent).
    k : np.ndarray
        Wavenumbers.
    multipoles_per_bin : dict
        Nested dict: {bin_label: {ell: P_ell(k)}}.
        Example: {'DS1': {0: array, 2: array, 4: array}, ...}
    """
    arrays = {"k": k}
    for label, poles in multipoles_per_bin.items():
        for ell, pk in poles.items():
            arrays[f"{label}_P{ell}"] = pk
    np.savez(path, **arrays)


def load_predictions(path) -> tuple:
    """Load predictions from a .npz file.

    Returns
    -------
    k : np.ndarray
    multipoles_per_bin : dict
        {bin_label: {ell: array}}
    """
    data = np.load(path)
    k = data["k"]

    multipoles_per_bin: dict = {}
    for key in data.files:
        if key == "k":
            continue
        # Key format: <label>_P<ell>
        label, pole = key.rsplit("_P", 1)
        ell = int(pole)
        multipoles_per_bin.setdefault(label, {})[ell] = data[key]

    return k, multipoles_per_bin


def load_measurements(path, nquantiles=5, ells=(0, 2, 4), rebin=5) -> tuple:
    """Load measured multipoles from an lsstypes HDF5 file.

    Reads an ObservableTree produced by ACM's DensitySplit.quantile_data_power()
    and returns data in the same format as load_predictions().

    Parameters
    ----------
    path : str or Path
    nquantiles : int, default 5
    ells : tuple of int, default (0, 2, 4)

    Returns
    -------
    k : np.ndarray
    multipoles_per_bin : dict  —  {bin_label: {ell: array}}
    """
    from lsstypes import read as lss_read

    tree = lss_read(str(path)).select(k=slice(0, None, rebin))

    k = None
    multipoles_per_bin = {}
    for qid in range(nquantiles):
        label = f"DS{qid + 1}"
        quantile = tree.get(quantiles=qid)
        poles = {}
        for ell in ells:
            leaf = quantile.get(ells=ell)
            if k is None:
                k = np.array(leaf.coords('k'))
            poles[ell] = np.array(leaf.value())
        multipoles_per_bin[label] = poles
    return k, multipoles_per_bin


def load_pgg_measurements(path, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf) -> tuple:
    """Load galaxy auto-power spectrum multipoles from a jaxpower HDF5 file.

    Parameters
    ----------
    path : str or Path
    ells : tuple of int, default (0, 2)
    rebin : int, default 13
    kmin : float, default 0.0
    kmax : float, default np.inf

    Returns
    -------
    k : np.ndarray
    poles : dict  —  {ell: array}
    """
    from jaxpower import read

    data = read(str(path)).select(k=slice(0, None, rebin))
    if kmin > 0.0 or kmax < np.inf:
        data = data.select(k=(kmin, kmax))

    k = None
    poles = {}
    for ell in ells:
        leaf = data.get(ell)
        if k is None:
            k = np.array(leaf.coords('k'))
        poles[ell] = np.array(leaf.value())
    return k, poles


def mock_covariance(mock_dir, statistic, ells, k_data=None, mask=None,
                    rescale=1.0, rebin=13, **stat_kwargs):
    """Load mocks from directory, compute sample covariance + Hartlap-corrected precision.

    Parameters
    ----------
    mock_dir : path  — directory containing mock measurement files
    statistic : str  — "pgg" or "ds"
    ells : tuple of int
    k_data : np.ndarray, optional  — measurement k-grid; if the mock k-grid
        differs, each mock is interpolated onto k_data before computing the
        covariance (fixes k-grid mismatches).
    mask : np.ndarray of bool, optional  — subset the data vector
    rescale : float  — divide covariance by this factor (default 1.0)
    rebin : int  — rebinning factor for mock loading
    **stat_kwargs : forwarded to statistic-specific loader
        For "ds": nquantiles (int), quantiles (tuple of int)

    Returns
    -------
    cov : np.ndarray  shape (n_data, n_data)
    precision : np.ndarray  shape (n_data, n_data)
    """
    cache_hash = _mock_cache_key(statistic, ells, rebin, **stat_kwargs)
    cache_path = Path(mock_dir) / f".mock_cache_{cache_hash}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        k_mock, mock_matrix = cached["k"], cached["mock_matrix"]
    elif statistic == "pgg":
        k_mock, mock_matrix = _load_pgg_mocks(mock_dir, ells=ells, rebin=rebin,
                                               kmin=stat_kwargs.get("kmin", 0.0),
                                               kmax=stat_kwargs.get("kmax", np.inf))
        np.savez(cache_path, k=k_mock, mock_matrix=mock_matrix)
    elif statistic == "ds":
        k_mock, mock_matrix = _load_ds_mocks(mock_dir, ells=ells, rebin=rebin,
                                              nquantiles=stat_kwargs.get("nquantiles", 5),
                                              quantiles=stat_kwargs.get("quantiles"))
        np.savez(cache_path, k=k_mock, mock_matrix=mock_matrix)
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}. Expected 'pgg' or 'ds'.")

    # Interpolate mock data vectors onto measurement k-grid if they differ
    if k_data is not None and not np.array_equal(k_mock, k_data):
        n_mocks = mock_matrix.shape[0]
        nk_mock, nk_data = len(k_mock), len(k_data)
        n_blocks = mock_matrix.shape[1] // nk_mock
        interp = np.empty((n_mocks, n_blocks * nk_data))
        for b in range(n_blocks):
            src = mock_matrix[:, b * nk_mock:(b + 1) * nk_mock]
            for i in range(n_mocks):
                interp[i, b * nk_data:(b + 1) * nk_data] = np.interp(k_data, k_mock, src[i])
        mock_matrix = interp

    return _compute_covariance(mock_matrix, mask=mask, rescale=rescale)


def diagonal_covariance(data_y, noise_frac=0.05, floor=50.0, rescale=1.0):
    """Diagonal covariance with fractional noise + absolute floor.

    Returns
    -------
    cov : np.ndarray  shape (n, n)
    precision : np.ndarray  shape (n, n)
    """
    var = (noise_frac * np.abs(data_y)) ** 2 + floor ** 2
    cov = np.diag(var) / rescale
    precision = np.diag(1.0 / (var / rescale))
    return cov, precision


def _mock_cache_key(statistic, ells, rebin, **stat_kwargs):
    """Return a short hex hash identifying a unique mock-loading configuration."""
    parts = [statistic, str(tuple(sorted(ells))), str(rebin)]
    if statistic == "pgg":
        parts.append(f"kmin={stat_kwargs.get('kmin', 0.0)}")
        parts.append(f"kmax={stat_kwargs.get('kmax', np.inf)}")
    elif statistic == "ds":
        parts.append(f"nq={stat_kwargs.get('nquantiles', 5)}")
        q = stat_kwargs.get('quantiles')
        parts.append(f"q={tuple(q) if q is not None else None}")
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


def _load_pgg_mocks(directory, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf):
    """Load P_gg mock files -> (k, mock_matrix)."""
    paths = sorted(Path(directory).glob("mesh2_spectrum_poles_ph*.h5"))
    rows = []
    k = None
    for p in paths:
        k_i, poles = load_pgg_measurements(p, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax)
        if k is None:
            k = k_i
        row = np.concatenate([poles[ell] for ell in ells])
        rows.append(row)
    return k, np.array(rows)


def _load_ds_mocks(directory, nquantiles=5, quantiles=None, ells=(0, 2), rebin=5):
    """Load DS mock files -> (k, mock_matrix)."""
    if quantiles is None:
        quantiles = range(1, nquantiles + 1)
    paths = sorted(Path(directory).glob("*.h5"))
    rows = []
    k = None
    for p in paths:
        k_i, meas = load_measurements(p, nquantiles=nquantiles, ells=ells, rebin=rebin)
        if k is None:
            k = k_i
        row = np.concatenate([
            meas[f"DS{q}"][ell]
            for q in quantiles
            for ell in ells
        ])
        rows.append(row)
    return k, np.array(rows)


def _compute_covariance(mock_matrix, mask=None, rescale=1.0):
    """np.cov + Hartlap correction -> (cov, precision)."""
    vecs = mock_matrix[:, mask] if mask is not None else mock_matrix
    n_mocks, n_data = vecs.shape
    if n_mocks <= n_data:
        raise ValueError(
            f"Covariance matrix is singular: n_mocks={n_mocks} <= n_data={n_data}. "
            "Apply a kmax cut to reduce n_data below n_mocks."
        )
    cov = np.cov(vecs.T) / rescale                  # (n_data, n_data)
    precision = np.linalg.inv(cov)
    hartlap = (n_mocks - n_data - 2) / (n_mocks - 1)
    precision *= hartlap
    return cov, precision


def save_text(path, k: np.ndarray, multipoles_per_bin: dict) -> None:
    """Save predictions to a human-readable text file.

    Columns: k, then for each bin and each ell: <label>_P<ell>.

    Parameters
    ----------
    path : str or Path
        Output file path.
    k : np.ndarray
    multipoles_per_bin : dict
        {bin_label: {ell: array}}
    """
    path = Path(path)
    cols = [k]
    header_parts = ["k"]

    for label, poles in multipoles_per_bin.items():
        for ell in sorted(poles):
            cols.append(poles[ell])
            header_parts.append(f"{label}_P{ell}")

    data = np.column_stack(cols)
    header = "  ".join(header_parts)
    np.savetxt(path, data, header=header)
