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
