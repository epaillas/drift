"""Generate synthetic (noiseless) data vectors for pipeline validation."""

import numpy as np

from .cosmology import get_cosmology
from .galaxy_emulator import GalaxyTemplateEmulator


def make_synthetic_pgg(
    k,
    ells,
    z,
    space,
    mode,
    true_params,
    cosmo=None,
):
    """Generate synthetic P_gg multipoles from the template emulator.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Wavenumbers in h/Mpc.
    ells : tuple of int
        Multipole orders.
    z : float
        Redshift.
    space : str
        'redshift' or 'real'.
    mode : str
        EFT mode.
    true_params : dict
        Galaxy bias / EFT parameters (b1, c0, c2, s0, b2, bs2).
    cosmo : cosmoprimo.Cosmology, optional
        Defaults to ``get_cosmology()``.

    Returns
    -------
    data_vector : np.ndarray
        Flat array [P_ell0(k), P_ell1(k), ...].
    true_params : dict
        The parameter dict used (for reference).
    """
    if cosmo is None:
        cosmo = get_cosmology()

    emulator = GalaxyTemplateEmulator(cosmo, k, ells=ells, z=z, space=space, mode=mode)
    data_vector = emulator.predict(true_params)

    return data_vector, true_params
