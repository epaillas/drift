"""I/O and covariance-estimation helpers for DRIFT observables."""

import hashlib
from pathlib import Path

import numpy as np

from .covariance import (
    analytic_pgg_covariance as _analytic_pgg_covariance,
    analytic_pqg_covariance as _analytic_pqg_covariance,
    analytic_pqq_covariance as _analytic_pqq_covariance,
)


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


def _load_measurements_pqg(path, nquantiles=5, ells=(0, 2, 4), rebin=5, kmin=0.0, kmax=np.inf) -> tuple:
    """Load DS-galaxy power-spectrum multipoles from an lsstypes HDF5 file."""
    from lsstypes import read as lss_read

    tree = lss_read(str(path)).select(k=slice(0, None, rebin))
    if kmin > 0.0 or kmax < np.inf:
        tree = tree.select(k=(kmin, kmax))

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


def _load_measurements_xiqg(
    path,
    nquantiles=5,
    quantiles=None,
    ells=(0, 2, 4),
    rebin=4,
    smin=0.0,
    smax=np.inf,
) -> tuple:
    """Load DS-galaxy correlation-function multipoles from an lsstypes HDF5 file.

    Parameters
    ----------
    path : str or Path
    nquantiles : int, default 5
    quantiles : tuple of int, optional
        1-based quantile indices to return. Defaults to all quantiles.
    ells : tuple of int, default (0, 2, 4)
    rebin : int, default 1
    smin : float, default 0.0
    smax : float, default np.inf

    Returns
    -------
    s : np.ndarray
    multipoles_per_bin : dict
        {bin_label: {ell: array}}
    """
    from lsstypes import read as lss_read

    if quantiles is None:
        quantiles = tuple(range(1, nquantiles + 1))
    if rebin < 1:
        raise ValueError("rebin must be a positive integer.")

    tree = lss_read(str(path))

    s = None
    multipoles_per_bin = {}
    for quantile in quantiles:
        label = f"DS{quantile}"
        corr = tree.get(quantiles=quantile - 1)
        if smin > 0.0 or smax < np.inf:
            corr = corr.select(s=(smin, smax))
        if rebin > 1:
            corr = corr.select(s=slice(0, None, rebin))
        poles = corr.project(ells=list(ells))
        multipoles = {}
        for ell in ells:
            leaf = poles.get(ell)
            if s is None:
                s = np.array(leaf.s)
            multipoles[ell] = np.array(leaf.value())
        multipoles_per_bin[label] = multipoles

    return s, multipoles_per_bin


def _load_measurements_pgg(path, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf) -> tuple:
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


def load_observable_measurements(path, observable, **kwargs) -> tuple:
    """Load measured multipoles for a supported observable.

    Parameters
    ----------
    path : str or Path
    observable : str
        Observable key such as ``"pgg"``, ``"pqg"``, ``"ds"``, or ``"xiqg"``.
    **kwargs
        Forwarded to the observable-specific loader.
    """
    loaders = {
        "pgg": _load_measurements_pgg,
        "pqg": _load_measurements_pqg,
        "ds": _load_measurements_pqg,
        "xiqg": _load_measurements_xiqg,
    }
    try:
        loader = loaders[observable]
    except KeyError as exc:
        supported = ", ".join(sorted(loaders))
        raise ValueError(
            f"Unknown observable={observable!r}. Supported observables: {supported}."
        ) from exc
    return loader(path, **kwargs)


def estimate_mock_covariance(mock_dir, observable, ells, k_data=None, s_data=None, mask=None,
                             rescale=1.0, rebin=13, return_precision=False, **stat_kwargs):
    """Estimate a mock covariance matrix and optionally its Hartlap-corrected precision.

    Parameters
    ----------
    mock_dir : path  — directory containing mock measurement files
    observable : str  — "pgg", "pqg", "pqq_auto", or "xiqg"
    ells : tuple of int
    k_data : np.ndarray, optional  — measurement k-grid; if the mock k-grid
        differs, each mock is interpolated onto k_data before computing the
        covariance (fixes k-grid mismatches).
    s_data : np.ndarray, optional  — measurement s-grid for configuration-space
        statistics; if the mock s-grid differs, each mock is interpolated onto
        s_data before computing the covariance.
    mask : np.ndarray of bool, optional  — subset the data vector
    rescale : float  — divide covariance by this factor (default 1.0)
    rebin : int  — rebinning factor for mock loading
    return_precision : bool, default False
        When True, also return the Hartlap-corrected precision matrix.
    **stat_kwargs : forwarded to statistic-specific loader
        For "ds" / "pqq_auto": nquantiles (int), quantiles (tuple of int)

    Returns
    -------
    cov : np.ndarray  shape (n_data, n_data)
        Returned when ``return_precision=False``.
    (cov, precision) : tuple[np.ndarray, np.ndarray]
        Returned when ``return_precision=True``.
    """
    mock_matrix = _load_mock_matrix(
        mock_dir, observable, ells, k_data=k_data, s_data=s_data, rebin=rebin, **stat_kwargs
    )
    if return_precision:
        return _compute_covariance(mock_matrix, mask=mask, rescale=rescale)
    return _sample_covariance(mock_matrix, mask=mask, rescale=rescale)


def build_diagonal_covariance(data_vector, noise_frac=0.05, floor=50.0, rescale=1.0):
    """Diagonal covariance with fractional noise + absolute floor.

    Returns
    -------
    cov : np.ndarray  shape (n, n)
    precision : np.ndarray  shape (n, n)
    """
    var = (noise_frac * np.abs(data_vector)) ** 2 + floor ** 2
    cov = np.diag(var) / rescale
    precision = np.diag(1.0 / (var / rescale))
    return cov, precision


def analytic_pgg_covariance(k_data, poles, ells, volume,
                            number_density=None, shot_noise=None,
                            mask=None, rescale=1.0, terms="gaussian",
                            cng_amplitude=0.0, cng_coherence=0.35,
                            ssc_sigma_b2=None):
    """Fixed fiducial analytic covariance for galaxy power-spectrum multipoles."""
    return _analytic_pgg_covariance(
        k_data,
        poles,
        ells=ells,
        volume=volume,
        number_density=number_density,
        shot_noise=shot_noise,
        mask=mask,
        rescale=rescale,
        terms=terms,
        cng_amplitude=cng_amplitude,
        cng_coherence=cng_coherence,
        ssc_sigma_b2=ssc_sigma_b2,
    )


def analytic_pqq_covariance(k_data, poles, ells, volume, pair_order,
                            shot_noise, mask=None, rescale=1.0,
                            terms="gaussian", mu_points=256,
                            cng_amplitude=0.0, cng_coherence=0.35,
                            ssc_sigma_b2=None):
    """Fixed fiducial analytic covariance for DS-pair power-spectrum multipoles."""
    return _analytic_pqq_covariance(
        k_data,
        poles,
        ells=ells,
        volume=volume,
        pair_order=pair_order,
        shot_noise=shot_noise,
        mask=mask,
        rescale=rescale,
        terms=terms,
        mu_points=mu_points,
        cng_amplitude=cng_amplitude,
        cng_coherence=cng_coherence,
        ssc_sigma_b2=ssc_sigma_b2,
    )


def analytic_pqg_covariance(k_data, pqg_poles, pqq_poles, pgg_poles, ells,
                            volume, ds_labels, galaxy_shot_noise,
                            ds_pair_shot_noise, ds_cross_shot_noise=None,
                            mask=None, rescale=1.0, terms="gaussian",
                            mu_points=256, cng_amplitude=0.0,
                            cng_coherence=0.35, ssc_sigma_b2=None):
    """Fixed fiducial analytic covariance for DS-galaxy power-spectrum multipoles."""
    return _analytic_pqg_covariance(
        k_data,
        pqg_poles,
        pqq_poles,
        pgg_poles,
        ells=ells,
        volume=volume,
        ds_labels=ds_labels,
        galaxy_shot_noise=galaxy_shot_noise,
        ds_pair_shot_noise=ds_pair_shot_noise,
        ds_cross_shot_noise=ds_cross_shot_noise,
        mask=mask,
        rescale=rescale,
        terms=terms,
        mu_points=mu_points,
        cng_amplitude=cng_amplitude,
        cng_coherence=cng_coherence,
        ssc_sigma_b2=ssc_sigma_b2,
    )


def make_taylor_cache_key(**kwargs):
    """Return a short hex hash identifying a Taylor emulator configuration.

    Takes arbitrary key-value pairs, sorts by key, and returns a 12-char
    SHA-256 hex digest.  Scripts pass whatever config determines their
    theory function output.

    Returns
    -------
    str
        12-character hexadecimal digest.
    """
    parts = [f"{k}={v}" for k, v in sorted(kwargs.items())]
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


def _mock_cache_key(statistic, ells, rebin, **stat_kwargs):
    """Return a short hex hash identifying a unique mock-loading configuration."""
    parts = [statistic, str(tuple(sorted(ells))), str(rebin)]
    if statistic == "pgg":
        parts.append(f"kmin={stat_kwargs.get('kmin', 0.0)}")
        parts.append(f"kmax={stat_kwargs.get('kmax', np.inf)}")
    elif statistic in {"ds", "pqq_auto"}:
        parts.append(f"nq={stat_kwargs.get('nquantiles', 5)}")
        q = stat_kwargs.get('quantiles')
        parts.append(f"q={tuple(q) if q is not None else None}")
        parts.append(f"kmin={stat_kwargs.get('kmin', 0.0)}")
        parts.append(f"kmax={stat_kwargs.get('kmax', np.inf)}")
    elif statistic == "xiqg":
        parts.append(f"nq={stat_kwargs.get('nquantiles', 5)}")
        q = stat_kwargs.get('quantiles')
        parts.append(f"q={tuple(q) if q is not None else None}")
        parts.append(f"smin={stat_kwargs.get('smin', 0.0)}")
        parts.append(f"smax={stat_kwargs.get('smax', np.inf)}")
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


def _load_pgg_mocks(directory, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf):
    """Load P_gg mock files -> (k, mock_matrix)."""
    paths = sorted(Path(directory).glob("mesh2_spectrum_poles_ph*.h5"))
    rows = []
    k = None
    for p in paths:
        k_i, poles = load_observable_measurements(
            p, "pgg", ells=ells, rebin=rebin, kmin=kmin, kmax=kmax
        )
        if k is None:
            k = k_i
        row = np.concatenate([poles[ell] for ell in ells])
        rows.append(row)
    return k, np.array(rows)


def _load_ds_mocks(directory, nquantiles=5, quantiles=None, ells=(0, 2), rebin=5,
                   kmin=0.0, kmax=np.inf):
    """Load DS-galaxy mock files -> (k, mock_matrix)."""
    if quantiles is None:
        quantiles = range(1, nquantiles + 1)
    paths = sorted(Path(directory).glob("dsc_pkqg_poles_ph*.h5"))
    rows = []
    k = None
    for p in paths:
        k_i, meas = load_observable_measurements(
            p, "pqg", nquantiles=nquantiles, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax
        )
        if k is None:
            k = k_i
        row = np.concatenate([
            meas[f"DS{q}"][ell]
            for q in quantiles
            for ell in ells
        ])
        rows.append(row)
    return k, np.array(rows)


def _load_pqq_auto_mocks(directory, nquantiles=5, quantiles=None, ells=(0, 2), rebin=5,
                         kmin=0.0, kmax=np.inf):
    """Load DS auto-pair mock files -> (k, mock_matrix)."""
    if quantiles is None:
        quantiles = range(1, nquantiles + 1)
    paths = sorted(Path(directory).glob("dsc_pkqq_poles_ph*.h5"))
    rows = []
    k = None
    for p in paths:
        k_i, meas = load_observable_measurements(
            p, "pqg", nquantiles=nquantiles, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax
        )
        if k is None:
            k = k_i
        row = np.concatenate([
            meas[f"DS{q}"][ell]
            for q in quantiles
            for ell in ells
        ])
        rows.append(row)
    return k, np.array(rows)


def _load_xiqg_mocks(directory, nquantiles=5, quantiles=None, ells=(0, 2), rebin=1, smin=0.0, smax=np.inf):
    """Load xi_qg mock files -> (s, mock_matrix)."""
    if quantiles is None:
        quantiles = range(1, nquantiles + 1)
    paths = sorted(Path(directory).glob("dsc_xiqg/dsc_xiqg_poles_ph*.h5"))
    rows = []
    s = None
    for p in paths:
        s_i, meas = load_observable_measurements(
            p,
            "xiqg",
            nquantiles=nquantiles,
            quantiles=quantiles,
            ells=ells,
            rebin=rebin,
            smin=smin,
            smax=smax,
        )
        if s is None:
            s = s_i
        row = np.concatenate([
            meas[f"DS{q}"][ell]
            for q in quantiles
            for ell in ells
        ])
        rows.append(row)
    return s, np.array(rows)


def _compute_covariance(mock_matrix, mask=None, rescale=1.0):
    """np.cov + Hartlap correction -> (cov, precision)."""
    cov = _sample_covariance(mock_matrix, mask=mask, rescale=rescale)
    vecs = mock_matrix[:, mask] if mask is not None else mock_matrix
    n_mocks, n_data = vecs.shape
    if n_mocks <= n_data:
        raise ValueError(
            f"Covariance matrix is singular: n_mocks={n_mocks} <= n_data={n_data}. "
            "Apply a kmax cut to reduce n_data below n_mocks."
        )
    precision = np.linalg.inv(cov)
    hartlap = (n_mocks - n_data - 2) / (n_mocks - 1)
    precision *= hartlap
    return cov, precision


def _sample_covariance(mock_matrix, mask=None, rescale=1.0):
    """Return the sample covariance matrix without inverting it."""
    vecs = mock_matrix[:, mask] if mask is not None else mock_matrix
    return np.cov(vecs.T) / rescale


def _load_mock_matrix(mock_dir, statistic, ells, k_data=None, s_data=None, rebin=13, **stat_kwargs):
    """Load and optionally interpolate the mock data matrix."""
    cache_hash = _mock_cache_key(statistic, ells, rebin, **stat_kwargs)
    cache_path = Path(mock_dir) / f".mock_cache_{cache_hash}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        k_mock, mock_matrix = cached["k"], cached["mock_matrix"]
    elif statistic == "pgg":
        k_mock, mock_matrix = _load_pgg_mocks(
            mock_dir,
            ells=ells,
            rebin=rebin,
            kmin=stat_kwargs.get("kmin", 0.0),
            kmax=stat_kwargs.get("kmax", np.inf),
        )
        np.savez(cache_path, k=k_mock, mock_matrix=mock_matrix)
    elif statistic == "ds":
        k_mock, mock_matrix = _load_ds_mocks(
            mock_dir,
            ells=ells,
            rebin=rebin,
            nquantiles=stat_kwargs.get("nquantiles", 5),
            quantiles=stat_kwargs.get("quantiles"),
            kmin=stat_kwargs.get("kmin", 0.0),
            kmax=stat_kwargs.get("kmax", np.inf),
        )
        np.savez(cache_path, k=k_mock, mock_matrix=mock_matrix)
    elif statistic == "pqq_auto":
        k_mock, mock_matrix = _load_pqq_auto_mocks(
            mock_dir,
            ells=ells,
            rebin=rebin,
            nquantiles=stat_kwargs.get("nquantiles", 5),
            quantiles=stat_kwargs.get("quantiles"),
            kmin=stat_kwargs.get("kmin", 0.0),
            kmax=stat_kwargs.get("kmax", np.inf),
        )
        np.savez(cache_path, k=k_mock, mock_matrix=mock_matrix)
    elif statistic == "xiqg":
        s_mock, mock_matrix = _load_xiqg_mocks(
            mock_dir,
            ells=ells,
            nquantiles=stat_kwargs.get("nquantiles", 5),
            quantiles=stat_kwargs.get("quantiles"),
            rebin=rebin,
            smin=stat_kwargs.get("smin", 0.0),
            smax=stat_kwargs.get("smax", np.inf),
        )
        np.savez(cache_path, k=s_mock, mock_matrix=mock_matrix)
    else:
        raise ValueError(
            f"Unknown statistic: {statistic!r}. Expected 'pgg', 'ds', 'pqq_auto', or 'xiqg'."
        )

    grid_mock = locals().get("k_mock", locals().get("s_mock"))
    grid_data = k_data if k_data is not None else s_data

    if grid_data is not None and not np.array_equal(grid_mock, grid_data):
        n_mocks = mock_matrix.shape[0]
        n_grid_mock, n_grid_data = len(grid_mock), len(grid_data)
        n_blocks = mock_matrix.shape[1] // n_grid_mock
        interp = np.empty((n_mocks, n_blocks * n_grid_data))
        for b in range(n_blocks):
            src = mock_matrix[:, b * n_grid_mock:(b + 1) * n_grid_mock]
            for i in range(n_mocks):
                interp[i, b * n_grid_data:(b + 1) * n_grid_data] = np.interp(grid_data, grid_mock, src[i])
        mock_matrix = interp

    return mock_matrix


def save_predictions_text(path, k: np.ndarray, multipoles_per_bin: dict) -> None:
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


def load_measurements(path, nquantiles=5, ells=(0, 2, 4), rebin=5, kmin=0.0, kmax=np.inf) -> tuple:
    """Compatibility wrapper for DS-galaxy power-spectrum measurements."""
    return load_observable_measurements(
        path,
        "pqg",
        nquantiles=nquantiles,
        ells=ells,
        rebin=rebin,
        kmin=kmin,
        kmax=kmax,
    )


def load_pgg_measurements(path, ells=(0, 2), rebin=13, kmin=0.0, kmax=np.inf) -> tuple:
    """Compatibility wrapper for galaxy power-spectrum measurements."""
    return load_observable_measurements(
        path,
        "pgg",
        ells=ells,
        rebin=rebin,
        kmin=kmin,
        kmax=kmax,
    )


def load_correlation_measurements(path, nquantiles=5, quantiles=None, ells=(0, 2, 4), rebin=1, smin=0.0, smax=np.inf) -> tuple:
    """Compatibility wrapper for DS-galaxy correlation-function measurements."""
    return load_observable_measurements(
        path,
        "xiqg",
        nquantiles=nquantiles,
        quantiles=quantiles,
        ells=ells,
        rebin=rebin,
        smin=smin,
        smax=smax,
    )


def mock_covariance(mock_dir, statistic, ells, k_data=None, s_data=None, mask=None,
                    rescale=1.0, rebin=13, **stat_kwargs):
    """Compatibility wrapper for mock covariance estimation."""
    return estimate_mock_covariance(
        mock_dir,
        statistic,
        ells,
        k_data=k_data,
        s_data=s_data,
        mask=mask,
        rescale=rescale,
        rebin=rebin,
        return_precision=True,
        **stat_kwargs,
    )


def mock_covariance_matrix(mock_dir, statistic, ells, k_data=None, s_data=None, mask=None,
                           rescale=1.0, rebin=13, **stat_kwargs):
    """Compatibility wrapper for mock covariance-matrix estimation."""
    return estimate_mock_covariance(
        mock_dir,
        statistic,
        ells,
        k_data=k_data,
        s_data=s_data,
        mask=mask,
        rescale=rescale,
        rebin=rebin,
        return_precision=False,
        **stat_kwargs,
    )


def diagonal_covariance(data_y, noise_frac=0.05, floor=50.0, rescale=1.0):
    """Compatibility wrapper for diagonal covariance construction."""
    return build_diagonal_covariance(data_y, noise_frac=noise_frac, floor=floor, rescale=rescale)


def taylor_cache_key(**kwargs):
    """Compatibility wrapper for Taylor-emulator cache keys."""
    return make_taylor_cache_key(**kwargs)


def save_text(path, k: np.ndarray, multipoles_per_bin: dict) -> None:
    """Compatibility wrapper for text prediction output."""
    save_predictions_text(path, k, multipoles_per_bin)
