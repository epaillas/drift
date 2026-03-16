"""Save and load DRIFT predictions."""

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


def load_pgg_covariance_mocks(directory, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf):
    """Load all P_gg mock realizations from a directory for covariance estimation.

    Parameters
    ----------
    directory : str or Path
    ells : tuple of int
    rebin : int
    kmin : float
    kmax : float

    Returns
    -------
    k : np.ndarray  shape (nk,)
    mock_matrix : np.ndarray  shape (n_mocks, n_ells * nk)
        Row order: for ell in ells, k-values.
    """
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


def load_covariance_mocks(directory, nquantiles=5, quantiles=None, ells=(0, 2), rebin=5):
    """Load all mock realizations from a directory for covariance estimation.

    Parameters
    ----------
    directory : str or Path
    nquantiles : int
        Total number of quantiles in each file (used for loading).
    quantiles : sequence of int, optional
        Which quantile indices (1-based) to include in the output data vector.
        Defaults to range(1, nquantiles + 1) — all quantiles.
    ells : tuple of int
    rebin : int

    Returns
    -------
    k : np.ndarray  shape (nk,)
    mock_matrix : np.ndarray  shape (n_mocks, len(quantiles) * n_ells * nk)
        Row order matches the flat data vector built by inference scripts:
        for q in quantiles, for ell in ells, k-values.
    """
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


def make_mock_covariance(mock_matrix, mask=None, rescale=1.0):
    """Compute the sample covariance matrix from mock realizations.

    Applies the Hartlap (2007) correction to the precision matrix when
    n_mocks > n_data + 2.  Raises ValueError if n_mocks <= n_data (singular).

    Parameters
    ----------
    mock_matrix : np.ndarray  shape (n_mocks, n_data_full)
    mask : np.ndarray of bool, optional
        If provided, selects a subset of the data vector before computing
        the covariance (must be applied consistently with inference).
    rescale : float, optional
        Divide the covariance by this factor before inverting (default: 1.0).
        Equivalent to multiplying the precision matrix by rescale.

    Returns
    -------
    cov : np.ndarray  shape (n_data, n_data)
    precision : np.ndarray  shape (n_data, n_data)
        Hartlap-corrected inverse covariance.
    """
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
