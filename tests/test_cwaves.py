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
from cwaves.client import _l1b_cross_spectrum

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


# =============================================================================
# Round-trip (back-and-forth) reconstruction — issue #6
# =============================================================================
#
# Decompose a real WV cross-spectrum (the SARWave L1B XSP fixture) into CWAVE
# parameters, reconstruct the spectrum from the parameters and the kernels,
# and compare the reconstruction with the original. A perfect reconstruction
# is not expected (finite kernel truncation): the tolerances below are the
# decided pixel-to-pixel thresholds. The fixture grid (65 x 128 = 8320
# pixels) is far larger than the number of CWAVE coefficients (default
# 4 x 5 = 20), so the projection is well sampled.


def _normalized_spectrum(xs, kmin, kmax):
    """Return the normalized |xs| density projected by ``compute_cwave_parameters``.

    Mirrors the documented input contract (``repository_specifications.md``
    section 6): mask to the annulus ``kmin < kk < kmax``, drop fully masked
    rows/columns, take the real modulus, and normalize to unit mass.

    Args:
        xs: complex one-sided cross-spectrum (``k_rg >= 0``).
        kmin: minimum wavenumber (``2 pi / cwave_wlmax``).
        kmax: maximum wavenumber (``2 pi / cwave_wlmin``).

    Returns:
        (xr.DataArray, float): the normalized density and the cell area
        ``dkrg * dkaz`` of the uniform wavenumber grid.
    """
    kk = np.sqrt(xs.k_rg**2 + xs.k_az**2)
    xxs = xs.where((kk > kmin) & (kk < kmax))
    xxs = xxs.dropna(dim="freq_line", how="all").dropna(dim="freq_sample", how="all")
    xxsm = np.abs(xxs)
    dkrg = float(xxs.k_rg.diff(dim="freq_sample").mean(dim="freq_sample"))
    dkaz = float(xxs.k_az.diff(dim="freq_line").mean(dim="freq_line"))
    xxsmn = xxsm / (xxsm.sum(dim=["freq_line", "freq_sample"]) * dkrg * dkaz)
    return xxsmn, dkrg * dkaz


def _reconstruct(xs, nk, nphi, wlmax=600.0, wlmin=25.0):
    """Decompose ``xs`` and reconstruct the normalized spectrum from the CWAVES.

    Args:
        xs: complex one-sided cross-spectrum (``k_rg >= 0``).
        nk: number of Gegenbauer basis functions (``cwave_Nk``).
        nphi: number of harmonic basis functions (``cwave_Nphi``).
        wlmax: maximum wavelength in meters.
        wlmin: minimum wavelength in meters.

    Returns:
        (xr.DataArray, xr.DataArray): the reconstructed density
        ``sum(cwave_params * cwave_kernel)`` and the CWAVE parameters.
    """
    kmin, kmax = 2.0 * np.pi / wlmax, 2.0 * np.pi / wlmin
    cp = compute_cwave_parameters(
        xs, cwave_wlmax=wlmax, cwave_wlmin=wlmin, cwave_Nk=nk, cwave_Nphi=nphi
    )
    kernel = compute_kernel(xs.k_rg, xs.k_az, kmin, kmax, cwave_Nk=nk, cwave_Nphi=nphi)
    fhat = (kernel["cwave_kernel"] * cp).sum(dim=["k_gp", "phi_hf"])
    return fhat, cp


def _load_l1b_xs(l1b_xsp_nc):
    """Load the one-sided 2-tau cross-spectrum from the L1B XSP fixture."""
    with xr.open_dataset(l1b_xsp_nc) as ds:
        return _l1b_cross_spectrum(ds, lag=2, time_idx=0, pol_idx=0)


def _capture(f, fhat):
    """Fraction of the domain-A energy captured by the reconstruction."""
    err = f - fhat  # xarray aligns the two by coordinates
    fv = np.asarray(f, dtype=float)
    ev = np.asarray(err, dtype=float)
    return 1.0 - float(np.nansum(ev**2) / np.nansum(fv**2))


def test_cwave_roundtrip_reconstruction(l1b_xsp_nc):
    """Back-and-forth: decompose a real WV cross-spectrum and reconstruct it.

    The reconstruction error is dominated by the finite kernel truncation
    (default 4 x 5 = 20 coefficients): the median pixel error is well below
    1 % of the spectral peak, while the few pixels at the peak carry the
    truncation error. The thresholds below are the decided pixel-to-pixel
    tolerances (issue #6).
    """
    xs = _load_l1b_xs(l1b_xsp_nc)
    kmin, kmax = 2.0 * np.pi / 600.0, 2.0 * np.pi / 25.0
    f, _ = _normalized_spectrum(xs, kmin, kmax)
    fhat, _ = _reconstruct(xs, 4, 5)

    err = f - fhat
    fv = np.asarray(f, dtype=float)
    ev = np.asarray(err, dtype=float)
    fmax = float(np.nanmax(fv))
    rel_l2 = float(np.sqrt(np.nansum(ev**2) / np.nansum(fv**2)))
    max_pix = float(np.nanmax(np.abs(ev)) / fmax)
    valid = ~np.isnan(ev)
    within_half_peak = float(np.mean(np.abs(ev[valid]) <= 0.5 * fmax))

    # global (relative L2) reconstruction error
    assert rel_l2 < 0.65
    # pixel-to-pixel: the worst pixel stays below 90 % of the spectral peak
    assert max_pix < 0.9
    # and the overwhelming majority of pixels are reconstructed within
    # 50 % of the peak
    assert within_half_peak > 0.95
    # a substantial fraction of the domain-A energy is captured
    assert _capture(f, fhat) > 0.5


def test_cwave_roundtrip_energy_balance(l1b_xsp_nc):
    """The round trip satisfies the discrete Parseval energy balance.

    For an orthonormal kernel basis the CWAVE coefficients ``c_i`` and the
    reconstruction residual obey ``||f - fhat||^2 = ||f||^2 - sum_i c_i^2``
    (discrete Parseval identity). This validates that the decomposition and
    the reconstruction use the same orthonormal kernels and the same
    quadrature, independently of the spectrum smoothness.
    """
    xs = _load_l1b_xs(l1b_xsp_nc)
    kmin, kmax = 2.0 * np.pi / 600.0, 2.0 * np.pi / 25.0
    f, cell_area = _normalized_spectrum(xs, kmin, kmax)
    fhat, cp = _reconstruct(xs, 4, 5)

    err = f - fhat
    fv = np.asarray(f, dtype=float)
    ev = np.asarray(err, dtype=float)
    norm2 = float(np.nansum(fv**2) * cell_area)
    sum_c2 = float(np.sum(np.asarray(cp, dtype=float) ** 2))
    resid2 = float(np.nansum(ev**2) * cell_area)

    assert abs(resid2 - (norm2 - sum_c2)) < 0.05 * norm2


def test_cwave_roundtrip_convergence(l1b_xsp_nc):
    """Captured energy is non-decreasing when the kernel set is enlarged.

    The kernel sets (4, 5) < (4, 8) < (8, 12) are nested, so the truncated
    expansion captures at least as much of the spectrum energy as the
    coarser one: the decomposition converges to the original spectrum as
    the number of CWAVE coefficients grows.
    """
    xs = _load_l1b_xs(l1b_xsp_nc)
    kmin, kmax = 2.0 * np.pi / 600.0, 2.0 * np.pi / 25.0
    f, _ = _normalized_spectrum(xs, kmin, kmax)
    captures = []
    for nk, nphi in ((4, 5), (4, 8), (8, 12)):
        fhat, _ = _reconstruct(xs, nk, nphi)
        captures.append(_capture(f, fhat))
    assert captures[0] <= captures[1] + 0.01
    assert captures[1] <= captures[2] + 0.01
