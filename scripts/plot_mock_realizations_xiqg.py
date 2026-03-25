"""Plot xi_qg mock realizations and their mean for artifact inspection."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_observable_measurements

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
DEFAULT_MOCK_DIR = Path(__file__).parents[1] / "outputs" / "hods" / "for_covariance" / "dsc_xiqg"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "mock_realizations_xiqg.png"
DEFAULT_QUANTILES = (1, 2, 3, 4, 5)
DEFAULT_ELLS = (0, 2)


def _parse_quantiles(values):
    if values is None:
        return DEFAULT_QUANTILES
    quantiles = tuple(int(value) for value in values)
    if not quantiles:
        raise ValueError("--quantiles must contain at least one quantile.")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("--quantiles must not contain duplicates.")
    return quantiles


def _parse_ells(values):
    if values is None:
        return DEFAULT_ELLS
    ells = tuple(int(value) for value in values)
    if not ells:
        raise ValueError("--ells must contain at least one multipole.")
    if len(set(ells)) != len(ells):
        raise ValueError("--ells must not contain duplicates.")
    return ells


def discover_mock_paths(mock_dir, max_realizations=None):
    """Return sorted xi_qg mock files, optionally capped to the first N."""
    paths = sorted(Path(mock_dir).glob("dsc_xiqg_poles_ph*.h5"))
    if not paths:
        raise FileNotFoundError(f"No dsc_xiqg_poles_ph*.h5 files found in {mock_dir}.")
    if max_realizations is not None:
        if max_realizations < 1:
            raise ValueError("--max-realizations must be a positive integer.")
        paths = paths[:max_realizations]
    return tuple(paths)


def load_mock_realizations(
    mock_dir,
    *,
    quantiles=DEFAULT_QUANTILES,
    ells=DEFAULT_ELLS,
    rebin=1,
    smin=0.0,
    smax=200.0,
    max_realizations=None,
):
    """Load xi_qg multipoles for each selected mock realization."""
    paths = discover_mock_paths(mock_dir, max_realizations=max_realizations)

    s_ref = None
    realizations = {f"DS{quantile}": {ell: [] for ell in ells} for quantile in quantiles}
    phases = []

    for path in paths:
        s, multipoles = load_observable_measurements(
            path,
            "xiqg",
            nquantiles=max(quantiles),
            quantiles=quantiles,
            ells=ells,
            rebin=rebin,
            smin=smin,
            smax=smax,
        )
        s = np.asarray(s, dtype=float)
        if s_ref is None:
            s_ref = s
        elif not np.allclose(s, s_ref):
            raise ValueError(f"Inconsistent separation grid in {path}.")

        for label in realizations:
            for ell in ells:
                realizations[label][ell].append(np.asarray(multipoles[label][ell], dtype=float))
        phases.append(path.stem.split("_")[-1])

    stacked = {
        label: {ell: np.stack(values, axis=0) for ell, values in ell_map.items()}
        for label, ell_map in realizations.items()
    }
    means = {
        label: {ell: np.mean(values, axis=0) for ell, values in ell_map.items()}
        for label, ell_map in stacked.items()
    }
    return np.asarray(s_ref, dtype=float), stacked, means, tuple(phases)


def make_figure(s, realizations, means, *, quantiles=DEFAULT_QUANTILES, ells=DEFAULT_ELLS, phases=(), show_mean_only=False):
    """Build a one-row-per-quantile xi_qg realization figure."""
    labels = tuple(f"DS{quantile}" for quantile in quantiles)
    fig, axes = plt.subplots(
        len(labels),
        len(ells),
        figsize=(4.0 * len(ells), 2.6 * len(labels)),
        squeeze=False,
        sharex=True,
    )

    y = np.asarray(s, dtype=float) ** 2
    for row, label in enumerate(labels):
        for col, ell in enumerate(ells):
            ax = axes[row, col]
            if not show_mean_only:
                for values in realizations[label][ell]:
                    ax.plot(
                        s,
                        y * values,
                        color="0.7",
                        lw=0.7,
                        alpha=0.35,
                        zorder=1,
                    )
            ax.plot(
                s,
                y * means[label][ell],
                color="k",
                lw=1.8,
                zorder=3,
            )
            ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.7)
            if row == 0:
                ax.set_title(rf"$\ell = {ell}$")
            if col == 0:
                ax.set_ylabel(rf"{label}" + "\n" + rf"$s^2\,\xi_\ell(s)$ [$(\mathrm{{Mpc}}/h)^2$]")
            if row == len(labels) - 1:
                ax.set_xlabel(r"$s$ [Mpc$/h$]")

    mean_label = "mean only" if show_mean_only else "mean + realizations"
    fig.suptitle(
        rf"DRIFT: xi_{{qg}} mocks ({len(phases)} realizations; {mean_label})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-dir", type=Path, default=DEFAULT_MOCK_DIR, metavar="PATH",
                        help="Directory containing dsc_xiqg_poles_ph*.h5 covariance mocks.")
    parser.add_argument("--quantiles", nargs="+", default=None, metavar="Q",
                        help="Density-split quantiles to plot. Defaults to 1 2 3 4 5.")
    parser.add_argument("--ells", nargs="+", type=int, default=None, metavar="ELL",
                        help="Multipoles to plot. Defaults to 0 2.")
    parser.add_argument("--smin", type=float, default=0.0, metavar="VALUE",
                        help="Minimum separation retained from the xi_qg mocks, in Mpc/h.")
    parser.add_argument("--smax", type=float, default=200.0, metavar="VALUE",
                        help="Maximum separation retained from the xi_qg mocks, in Mpc/h.")
    parser.add_argument("--rebin", type=int, default=1, metavar="N",
                        help="Keep every Nth separation bin when loading the xi_qg mocks.")
    parser.add_argument("--max-realizations", type=int, default=None, metavar="N",
                        help="Only load the first N sorted realizations.")
    parser.add_argument("--show-mean-only", action="store_true",
                        help="Only draw the black realization mean, without the grey individual mocks.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, metavar="PATH",
                        help="Output figure path.")
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    ells = _parse_ells(args.ells)
    s, realizations, means, phases = load_mock_realizations(
        args.mock_dir,
        quantiles=quantiles,
        ells=ells,
        rebin=args.rebin,
        smin=args.smin,
        smax=args.smax,
        max_realizations=args.max_realizations,
    )

    print("Preparing xi_qg mock-realization plot")
    print(f"  mock directory: {args.mock_dir}")
    print(f"  realizations: {len(phases)}")
    print(f"  quantiles: {tuple(f'DS{quantile}' for quantile in quantiles)}")
    print(f"  ells: {ells}")
    print(f"  rebin: {args.rebin}")
    print(f"  smin: {float(s.min())}")
    print(f"  smax: {float(s.max())}")

    fig, _ = make_figure(
        s,
        realizations,
        means,
        quantiles=quantiles,
        ells=ells,
        phases=phases,
        show_mean_only=args.show_mean_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
