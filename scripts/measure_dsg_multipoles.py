"""Measure density-split × galaxy cross power spectrum multipoles from a lognormal mock.

Generates a lognormal mock galaxy catalog, applies RSD along the z-axis, then
measures the density-split × galaxy cross power spectrum multipoles (ell=0,2,4)
using ACM's DensitySplit estimator.

Results are saved to outputs/dsg_measured.hdf5 in lsstypes ObservableTree format.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.io import load_observable_measurements

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BOXSIZE = 2000.0        # Mpc/h
NBAR = 5e-4            # (Mpc/h)^-3, ~37k galaxies in 500^3
NMESH = 512
BIAS = 2.0             # Eulerian galaxy bias
Z = 0.5
R = 10.0               # Mpc/h, smoothing radius
SEED = 42
ELLS = (0, 2, 4)
NQUANTILES = 5
SPACE = "redshift"     # "redshift" | "real"

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


def generate_lognormal_mock(boxsize, nbar, nmesh, bias, z, seed, space="redshift"):
    """Generate a lognormal mock galaxy catalog with Zeldovich displacements."""
    from mockfactory import LagrangianLinearMock
    from cosmoprimo.fiducial import DESI

    cosmo = DESI()
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    # Linear growth rate f = d ln D / d ln a
    f = cosmo.sigma8_z(z=z, of="theta_cb") / cosmo.sigma8_z(z=z, of="delta_cb")

    mock = LagrangianLinearMock(
        power,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=[0.0, 0.0, 0.0],
        seed=seed,
        unitary_amplitude=False,
    )

    # Set the density field with Lagrangian bias (Eulerian bias - 1)
    mock.set_real_delta_field(bias=bias - 1)

    # Set the selection function
    mock.set_analytic_selection_function(nbar=nbar)

    # Sample galaxies from the density field
    mock.poisson_sample(seed=seed + 1)

    # add redshift-space distortions
    if space == "redshift":
        mock.set_rsd(f=f, los='z')

    # Convert to catalog and extract positions
    data = mock.to_catalog()
    positions = np.array(data['Position'])

    return positions


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Generate lognormal mock
    print("Generating lognormal mock ...")
    positions = generate_lognormal_mock(
        BOXSIZE, NBAR, NMESH, BIAS, Z, SEED, space=SPACE
    )
    n_gal = len(positions)
    nbar_eff = n_gal / BOXSIZE ** 3
    print(f"  N_gal = {n_gal:,}  (nbar_eff = {nbar_eff:.3e} (Mpc/h)^-3)")

    # 2. Run DensitySplit measurement
    from acm.estimators.galaxy_clustering.density_split import DensitySplit
    from acm import setup_logging

    setup_logging()

    print("Setting up DensitySplit estimator ...")
    ds = DensitySplit(
        data_positions=positions,
        boxsize=BOXSIZE,
        boxcenter=0.0,
        meshsize=NMESH,
    )
    ds.set_density_contrast(smoothing_radius=R)
    ds.set_quantiles(nquantiles=NQUANTILES, query_method="randoms")

    suffix = "_real" if SPACE == "real" else ""
    output_fn = OUTPUT_DIR / f"dsg_measured{suffix}.hdf5"
    print(f"Computing quantile x galaxy cross power (saving to {output_fn}) ...")
    ds.quantile_data_power(
        positions,
        edges={"step": 0.005},
        ells=ELLS,
        los="z",
        save_fn=str(output_fn),
    )

    # 3. Summary
    k, multipoles_per_bin = load_observable_measurements(output_fn, "pqg")
    print(f"\nSummary:")
    print(f"  k range: [{k.min():.4f}, {k.max():.4f}] h/Mpc  ({len(k)} bins)")
    for label, poles in multipoles_per_bin.items():
        p0_mean = np.mean(poles[0])
        print(f"  {label} mean P0 = {p0_mean:.1f} (Mpc/h)^3")


if __name__ == "__main__":
    main()
