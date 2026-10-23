"""Unit tests for the cwaves public API.

Adapted from ``xsar_slc/xsarslc/tests/test_methods_related_to_xspectra.py`` and
extended to cover the corrected kernel implementation (missing ``sqrt()`` in the
Gegenbauer coefficient + exact ``eta`` Jacobian, see
``repository_specifications.md`` section 5).
"""

import cwaves
import numpy as np
import pytest
import xarray as xr
from cwaves import (
    compute_cwave_parameters,
    compute_kernel,
    gegenbauer_polynoms,
    get_default_config,
    harmonic_functions,
    load_config,
    symmetrize_xspectrum,
)

# =============================================================================
# Public API / basic values
# =============================================================================


def test_public_api():
    assert callable(compute_cwave_parameters)
    assert callable(compute_kernel)
    assert callable(gegenbauer_polynoms)
    assert callable(harmonic_functions)
    assert callable(symmetrize_xspectrum)
    assert callable(get_default_config)
    assert callable(load_config)
    assert cwaves.__version__


def test_gegenbauer_polynoms_values():
    # gegenbauer_polynoms expects a DataArray (as used in compute_kernel)
    x = xr.DataArray(np.linspace(-1, 1, 11), dims=("n",))
    # C_0 = 1, C_1 = 2*lambda*x, C_2 = 2*lambda*(lambda+1)*x^2 - lambda
    # with lambda = 1.5 : C_1 = 3x, C_2 = 7.5 x^2 - 1.5
    assert np.allclose(gegenbauer_polynoms(x, 0), 1.0)
    assert np.allclose(gegenbauer_polynoms(x, 1), 3.0 * x)
    assert np.allclose(gegenbauer_polynoms(x, 2), 7.5 * x**2 - 1.5)
    # higher order (recursion) : returns a DataArray of the right shape
    c5 = gegenbauer_polynoms(x, 5)
    assert c5.dims == x.dims
    assert np.all(np.isfinite(c5.values))


def test_harmonic_functions_values():
    x = np.linspace(-np.pi / 2, np.pi / 2, 11)
    # f_1 = 1/sqrt(pi), f_2 = sqrt(2/pi) sin(2x), f_3 = sqrt(2/pi) cos(2x)
    assert np.allclose(harmonic_functions(x, 1), np.sqrt(1.0 / np.pi))
    assert np.allclose(harmonic_functions(x, 2), np.sqrt(2.0 / np.pi) * np.sin(2.0 * x))
    assert np.allclose(harmonic_functions(x, 3), np.sqrt(2.0 / np.pi) * np.cos(2.0 * x))


# =============================================================================
# Configuration
# =============================================================================


def test_default_config():
    cfg = get_default_config()
    assert "cwave" in cfg
    cwave = cfg["cwave"]
    assert cwave["cwave_wlmax"] == 600
    assert cwave["cwave_wlmin"] == 25
    assert cwave["cwave_Nk"] == 4
    assert cwave["cwave_Nphi"] == 5
    assert load_config() == cfg


def test_load_config_not_mapping(tmp_path):
    # a YAML file that does not contain a mapping must be rejected
    bad = tmp_path / "bad.yml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_config_as_kwargs(mock_xspectra_da_small):
    # the cwave block can be passed directly as kwargs to the computation
    cwave = get_default_config()["cwave"]
    cp = compute_cwave_parameters(mock_xspectra_da_small, **cwave)
    assert cp.sizes["k_gp"] == cwave["cwave_Nk"]
    assert cp.sizes["phi_hf"] == cwave["cwave_Nphi"]


# =============================================================================
# symmetrize_xspectrum
# =============================================================================


def test_symmetrize_xspectrum(mock_xspectra_da):
    xs = symmetrize_xspectrum(mock_xspectra_da)
    assert isinstance(xs, xr.DataArray)
    # symmetrization mirrors the range (freq_sample) dimension:
    # 2n - 2 points (the zero wavenumber is kept once, as in xsar_slc: 506 -> 1010)
    assert xs.sizes["freq_sample"] == 2 * mock_xspectra_da.sizes["freq_sample"] - 2
    assert xs.dims == mock_xspectra_da.dims


def test_symmetrize_xspectrum_real_warning(mock_xspectra_da):
    # a real (non complex) spectrum should only raise a warning, not fail
    real_da = mock_xspectra_da.real
    with pytest.warns(UserWarning):
        symmetrize_xspectrum(real_da)


def test_symmetrize_xspectrum_bad_dims(mock_xspectra_da):
    # dim_range / dim_azimuth must be dimensions (not coordinates)
    with pytest.raises(ValueError):
        symmetrize_xspectrum(mock_xspectra_da, dim_range="k_rg")


# =============================================================================
# compute_cwave_parameters
# =============================================================================


def test_compute_cwave_parameters(mock_xspectra_da):
    cp = compute_cwave_parameters(mock_xspectra_da)
    assert isinstance(cp, xr.DataArray)
    assert cp.name == "cwave_params"
    for key in ["k_gp", "tile_line", "tile_sample", "phi_hf", "burst", "pol"]:
        assert key in cp.dims
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.all(np.isfinite(cp.values))
    if "k_gp" in cp.coords:
        assert np.all(cp["k_gp"].values >= 0)


def test_compute_cwave_parameters_rejects_full_spectrum(mock_xspectra_da):
    # a full / symmetrized spectrum (two-sided range, k_rg < 0) must be rejected :
    # the CWAVE computation is performed on the one-sided (half) range domain only
    # (see repository_specifications.md section 6).
    full = symmetrize_xspectrum(mock_xspectra_da)
    assert float(full.k_rg.min()) < 0  # it is indeed two-sided in range
    with pytest.raises(ValueError, match="one-sided"):
        compute_cwave_parameters(full)


def test_compute_cwave_parameters_with_kwargs(mock_xspectra_da_small):
    # custom cwave_* parameters (and unrelated kwargs) are honoured
    kwargs = {
        "cwave_wlmax": 701,
        "cwave_wlmin": 23,
        "cwave_Nk": 5,
        "cwave_Nphi": 7,
        "lambda_range_max": [26.0, 51.0, 76.0],
        "excluded_variables": ["var_xspectra_0tau"],
        "default_az_cutoff_definition": "ipf",
        "bright_target_nstddev": 11,
    }
    cp = compute_cwave_parameters(mock_xspectra_da_small, **kwargs)
    assert cp.sizes["k_gp"] == kwargs["cwave_Nk"]
    assert cp.sizes["phi_hf"] == kwargs["cwave_Nphi"]
    assert np.all(np.isfinite(cp.values))


# =============================================================================
# compute_kernel / orthonormality of the corrected kernels
# =============================================================================


def _kernel_gram_matrix(npts=200):
    """Return the orthonormality (Gram) matrix of the kernels on the domain A.

    Built with the paper parameters (g=2, kmin=2pi/624m, kmax=2pi/60m), so it
    is independent of the xsar_slc default wavelengths.
    """
    g = 2.0
    kmax = 2 * np.pi / 60
    kmin = 2 * np.pi / 624
    a1 = (g**2 - g**4) / (g**2 * kmin**2 - kmax**2)
    a2 = (kmax**2 - g**4 * kmin**2) / (kmax**2 - g**2 * kmin**2)

    kx = np.linspace(-0.2, 0.2, npts)
    ky = np.linspace(-0.2, 0.2, npts)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    krg = xr.DataArray(ky_grid, dims=("j", "i"))
    kaz = xr.DataArray(kx_grid, dims=("j", "i"))

    n_k, n_phi = 3, 3
    kernel = compute_kernel(krg, kaz, kmin, kmax, cwave_Nk=n_k, cwave_Nphi=n_phi)
    da = kernel["cwave_kernel"]
    grid_dims = [d for d in da.dims if d not in ("k_gp", "phi_hf")]
    h = da.transpose("k_gp", *grid_dims, "phi_hf").values
    h = h.reshape(n_k, -1, n_phi)

    u = a1 * kx_grid**4 + a2 * kx_grid**2 + ky_grid**2
    mask = (u >= kmin**2) & (u <= kmax**2)
    h = np.where(mask.reshape(1, mask.size, 1), h, 0.0)

    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    flat = h.transpose(0, 2, 1).reshape(n_k * n_phi, -1)
    return flat @ flat.T * (dkx * dky)


def test_compute_kernel_orthonormality():
    """The corrected kernels must be mutually orthogonal on the domain A.

    This is the key property restored by the missing-sqrt / eta correction
    (see repository_specifications.md section 5): the original xsar_slc
    ``eta`` gave a ~24-39 % off-diagonal contamination.
    """
    gram = _kernel_gram_matrix()
    diag = np.diag(gram)
    offdiag = np.abs(gram - np.diag(diag)).max()
    assert offdiag < 0.05 * np.abs(diag).mean()


def test_compute_kernel_returns_components():
    kmax = 2 * np.pi / 60
    kmin = 2 * np.pi / 624
    kx = np.linspace(-0.2, 0.2, 40)
    ky = np.linspace(-0.2, 0.2, 40)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    kernel = compute_kernel(
        xr.DataArray(ky_grid, dims=("j", "i")),
        xr.DataArray(kx_grid, dims=("j", "i")),
        kmin,
        kmax,
        cwave_Nk=2,
        cwave_Nphi=2,
    )
    for var in ["cwave_kernel", "Gegenbauer_polynoms", "Harmonic_functions", "eta"]:
        assert var in kernel


def test_compute_kernel_drops_pol():
    """If the wavenumber grids carry a ``pol`` coordinate, the kernel output
    must drop it (defensive branch of compute_kernel)."""
    kx = np.linspace(-0.2, 0.2, 20)
    ky = np.linspace(-0.2, 0.2, 20)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    krg = xr.DataArray(ky_grid, dims=("j", "i"), coords={"pol": ("i", ["VV"] * 20)})
    kaz = xr.DataArray(kx_grid, dims=("j", "i"), coords={"pol": ("i", ["VV"] * 20)})
    kernel = compute_kernel(
        krg, kaz, 2 * np.pi / 624, 2 * np.pi / 60, cwave_Nk=2, cwave_Nphi=2
    )
    # "pol" must not remain as a variable of the merged kernel dataset
    assert "pol" not in kernel
    assert "cwave_kernel" in kernel


def test_compute_kernel_save(tmp_path, monkeypatch):
    # save_cwave_kernel=True writes cwaves_kernel.nc in the working directory
    monkeypatch.chdir(tmp_path)
    kx = np.linspace(-0.2, 0.2, 20)
    ky = np.linspace(-0.2, 0.2, 20)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    compute_kernel(
        xr.DataArray(ky_grid, dims=("j", "i")),
        xr.DataArray(kx_grid, dims=("j", "i")),
        2 * np.pi / 624,
        2 * np.pi / 60,
        cwave_Nk=2,
        cwave_Nphi=2,
        save_cwave_kernel=True,
    )
    assert (tmp_path / "cwaves_kernel.nc").exists()
