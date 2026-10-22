"""Basic tests for the cwaves public API (skeleton).

The full test suite (>= 80 % coverage) is added in roadmap task 8.
"""

import cwaves
import numpy as np
import xarray as xr
from cwaves import (
    compute_cwave_parameters,
    compute_kernel,
    gegenbauer_polynoms,
    harmonic_functions,
    symmetrize_xspectrum,
)


def test_public_api():
    assert callable(compute_cwave_parameters)
    assert callable(compute_kernel)
    assert callable(gegenbauer_polynoms)
    assert callable(harmonic_functions)
    assert callable(symmetrize_xspectrum)
    assert cwaves.__version__


def test_gegenbauer_polynoms_values():
    x = np.linspace(-1, 1, 11)
    # C_0 = 1, C_1 = 2*lambda*x
    assert np.allclose(gegenbauer_polynoms(x, 0), 1.0)
    assert np.allclose(gegenbauer_polynoms(x, 1), 3.0 * x)


def test_harmonic_functions_values():
    x = np.linspace(-np.pi / 2, np.pi / 2, 11)
    # f_1 = 1/sqrt(pi)
    assert np.allclose(harmonic_functions(x, 1), np.sqrt(1.0 / np.pi))
    # f_2 = sqrt(2/pi) sin(2x)
    assert np.allclose(harmonic_functions(x, 2), np.sqrt(2.0 / np.pi) * np.sin(2.0 * x))


def test_symmetrize_xspectrum_shape(mock_xspectra_da):
    xs = symmetrize_xspectrum(mock_xspectra_da)
    n = mock_xspectra_da.sizes["freq_sample"]
    # symmetrization mirrors the range (freq_sample) dimension:
    # the zero wavenumber is kept once -> 2n - 2 points
    assert xs.sizes["freq_sample"] == 2 * n - 2


def test_compute_cwave_parameters(mock_xspectra_da):
    cp = compute_cwave_parameters(
        mock_xspectra_da, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5
    )
    assert cp.name == "cwave_params"
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).any()


def test_compute_kernel_orthonormality():
    """The corrected kernels must be (mutually) orthogonal on the domain A.

    This is the key property fixed by the missing-sqrt / eta correction
    (see repository_specifications.md section 5).
    """
    g = 2.0
    kmax = 2 * np.pi / 60
    kmin = 2 * np.pi / 624
    a1 = (g**2 - g**4) / (g**2 * kmin**2 - kmax**2)
    a2 = (kmax**2 - g**4 * kmin**2) / (kmax**2 - g**2 * kmin**2)

    npts = 200
    kx = np.linspace(-0.2, 0.2, npts)
    ky = np.linspace(-0.2, 0.2, npts)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    krg = xr.DataArray(ky_grid, dims=("j", "i"))
    kaz = xr.DataArray(kx_grid, dims=("j", "i"))

    kernel = compute_kernel(krg, kaz, kmin, kmax, cwave_Nk=3, cwave_Nphi=3)
    # cwave_kernel dims: (k_gp, <grid dims...>, phi_hf)
    da = kernel["cwave_kernel"]
    grid_dims = [d for d in da.dims if d not in ("k_gp", "phi_hf")]
    h = da.transpose("k_gp", *grid_dims, "phi_hf").values  # (3, j, i, 3)
    h = h.reshape(3, -1, 3)  # (k_gp, grid, phi_hf)

    # mask to the domain A (reshape to (1, grid, 1) to broadcast over h)
    u = a1 * kx_grid**4 + a2 * kx_grid**2 + ky_grid**2
    mask = (u >= kmin**2) & (u <= kmax**2)
    ngrid = mask.size
    h = np.where(mask.reshape(1, ngrid, 1), h, 0.0)

    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    flat = h.transpose(0, 2, 1).reshape(9, -1)  # (k_gp*phi_hf, grid)
    gram = flat @ flat.T * (dkx * dky)
    diag = np.diag(gram)
    offdiag = np.abs(gram - np.diag(diag)).max()
    # orthogonality : off-diagonal must be small compared to the norm
    assert offdiag < 0.05 * np.abs(diag).mean()
