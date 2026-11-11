"""Tests for the polar-grid CWAVE parameter computation.

Covers ``cwaves.compute_cwave_parameters_polar`` (direct polar-grid quadrature
of the CWAVE projection, no cartesian resampling), its cell-weight helpers,
and the OCN WV client paths ``compute_cwaves_from_osw_polar`` / the CLI flag
``--polar-direct``.

The cartesian-vs-polar agreement tolerances are calibrated on smooth analytic
spectra (both paths converge to the same integral) and on the real S1A OCN
WV fixture shipped in ``assets/`` (see docs/certification.rst, "Direct
polar-grid computation (no resampling)").
"""

import numpy as np
import pytest
import xarray as xr
from cwaves import compute_cwave_parameters_polar
from cwaves.client import (
    compute_cwaves_from_osw,
    compute_cwaves_from_osw_polar,
    entrypoint,
)
from cwaves.compute_cwaves import (
    _polar_k_weights,
    _polar_phi_weights,
    compute_cwave_parameters,
    compute_kernel,
)

# =============================================================================
# helpers: smooth analytic spectrum sampled on polar / cartesian grids
# =============================================================================

#: paper parameters, as in tests/test_cwaves.py::_kernel_gram_matrix
KMIN_PAPER = 2.0 * np.pi / 624.0
KMAX_PAPER = 2.0 * np.pi / 60.0

PHI0 = np.deg2rad(40.0)
K0 = 0.08
SIGMA_K = 0.03
SIGMA_P = np.deg2rad(30.0)


def _radial(k):
    """Radial profile: gaussian around K0."""
    return np.exp(-0.5 * ((k - K0) / SIGMA_K) ** 2)


def _angular(phi):
    """Angular profile: gaussian (wrapped) around PHI0."""
    dphi = np.minimum(np.abs(phi - PHI0), 2.0 * np.pi - np.abs(phi - PHI0))
    return np.exp(-0.5 * (dphi / SIGMA_P) ** 2)


def _synthetic_polar(nk=300, nphi=360):
    """The analytic spectrum sampled on a log-polar grid (OCN-like)."""
    k = np.exp(np.linspace(np.log(0.004), np.log(0.24), nk))
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    spec = _radial(k)[None, :] * _angular(phi)[:, None]
    return spec, k, phi


def _synthetic_cartesian():
    """The same analytic spectrum sampled on a fine cartesian grid.

    One-sided in range (``k_rg >= 0``), two-sided in azimuth, as expected by
    :func:`cwaves.compute_cwave_parameters`.
    """
    k_rg = np.linspace(0.0, 0.24, 400)
    k_az = np.linspace(-0.24, 0.24, 800)
    kx, ky = np.meshgrid(k_rg, k_az, indexing="ij")
    kk = np.hypot(kx, ky)
    ph = (np.arctan2(ky, kx) + 2.0 * np.pi) % (2.0 * np.pi)
    spec = _radial(kk) * _angular(ph)
    return xr.DataArray(
        spec.astype(np.complex128),
        dims=("freq_sample", "freq_line"),
        coords={"k_rg": ("freq_sample", k_rg), "k_az": ("freq_line", k_az)},
        name="xspectra_test",
    )


def _polar_gram_matrix(nk=128, nphi=360):
    """Orthonormality (Gram) matrix of the kernels on the domain A, polar grid.

    Same paper parameters and (symmetric) domain A mask as
    ``tests/test_cwaves.py::_kernel_gram_matrix``, integrated over the full
    circle with the polar cell areas :math:`k\\,\\mathrm{d}k\\,\\mathrm{d}\\varphi`.
    Cells outside the elliptic domain A (where the kernel is NaN) are excluded,
    exactly as in the cartesian quadrature.
    """
    k = np.exp(np.linspace(np.log(0.004), np.log(0.13), nk))
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    kx = k[None, :] * np.cos(phi)[:, None]
    ky = k[None, :] * np.sin(phi)[:, None]
    krg = xr.DataArray(kx, dims=("phi", "k"), coords={"phi": phi, "k": k})
    kaz = xr.DataArray(ky, dims=("phi", "k"), coords={"phi": phi, "k": k})
    n_k, n_phi = 3, 3
    with np.errstate(invalid="ignore"):
        kernel = compute_kernel(
            krg, kaz, KMIN_PAPER, KMAX_PAPER, cwave_Nk=3, cwave_Nphi=3
        )
    da = kernel["cwave_kernel"]
    grid_dims = [d for d in da.dims if d not in ("k_gp", "phi_hf")]
    h = da.transpose("k_gp", *grid_dims, "phi_hf").values
    h = h.reshape(n_k, -1, n_phi)
    # domain A mask: the kernel is NaN outside A (same cells for all kernels)
    in_a = ~np.isnan(h).any(axis=(0, 2))  # (cells,)
    h = np.where(in_a[None, :, None], h, 0.0)  # 0 * NaN would poison the Gram
    w = (_polar_phi_weights(phi)[:, None] * _polar_k_weights(k)[None, :]).ravel()
    flat = h.transpose(0, 2, 1).reshape(n_k * n_phi, -1)
    return (flat * w[None, :]) @ flat.T


# =============================================================================
# public API / weight helpers
# =============================================================================


def test_public_api_polar():
    import cwaves

    assert callable(cwaves.compute_cwave_parameters_polar)
    assert callable(cwaves.compute_cwaves_from_osw_polar)
    assert "compute_cwave_parameters_polar" in cwaves.__all__
    assert "compute_cwaves_from_osw_polar" in cwaves.__all__


def test_polar_k_weights_log_grid():
    """The radial weights telescope to (b_hi^2 - b_lo^2)/2 (a k-dk rule)."""
    k = np.exp(np.linspace(np.log(0.005), np.log(0.2), 60))
    w = _polar_k_weights(k)
    assert np.all(w > 0.0)
    r0 = k[1] / k[0]
    rn = k[-1] / k[-2]
    b_lo2 = k[0] ** 2 / r0
    b_hi2 = k[-1] ** 2 * rn
    assert np.isclose(w.sum(), 0.5 * (b_hi2 - b_lo2))


def test_polar_phi_weights_uniform():
    phi = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    w = _polar_phi_weights(phi)
    assert np.allclose(w, 2.0 * np.pi / 72)
    assert np.isclose(w.sum(), 2.0 * np.pi)


# =============================================================================
# polar vs cartesian agreement (synthetic + real asset)
# =============================================================================


def test_polar_matches_cartesian_synthetic():
    """Both paths integrate the same smooth spectrum: they must agree."""
    spec, k, phi = _synthetic_polar(300, 360)
    cp_pol = compute_cwave_parameters_polar(spec, k, phi)
    cp_cart = compute_cwave_parameters(_synthetic_cartesian())
    assert cp_pol.name == "cwave_params"
    assert cp_pol.sizes == {"k_gp": 4, "phi_hf": 5}
    assert cp_pol.attrs["long_name"] == "CWAVE parameters (polar grid)"
    # calibrated on the fine-grid convergence (see docs/certification.rst)
    np.testing.assert_allclose(cp_pol.values, cp_cart.values, rtol=0.10, atol=0.01)


def test_polar_uses_modulus_only():
    """The complex phase of the input must be discarded (|spec| only)."""
    spec, k, phi = _synthetic_polar(96, 144)
    rng = np.random.default_rng(42)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, spec.shape))
    cp_real = compute_cwave_parameters_polar(spec, k, phi)
    cp_cplx = compute_cwave_parameters_polar(spec * phase, k, phi)
    np.testing.assert_allclose(cp_real.values, cp_cplx.values)


def test_polar_kernel_orthonormality():
    """The kernels must be mutually orthonormal on the domain A (polar grid)."""
    gram = _polar_gram_matrix()
    diag = np.diag(gram)
    offdiag = np.abs(gram - np.diag(diag)).max()
    assert offdiag < 0.05 * np.abs(diag).mean()


def test_polar_matches_cartesian_real_asset(osw_measurement_nc):
    """On the real S1A OCN WV fixture: polar-direct vs cartesian (128x128).

    The residual difference is the quadrature error of the product's own
    coarse 72x60 polar grid (shared by both paths), not a resampling error.
    """
    cp_pol = compute_cwaves_from_osw_polar(str(osw_measurement_nc))
    cp_cart = compute_cwaves_from_osw(
        str(osw_measurement_nc), n_range=128, n_azimuth=128
    )
    np.testing.assert_allclose(cp_pol.values, cp_cart.values, rtol=0.20, atol=0.5)


# =============================================================================
# client: compute_cwaves_from_osw_polar
# =============================================================================


def test_compute_cwaves_from_osw_polar(osw_measurement_nc):
    cp = compute_cwaves_from_osw_polar(str(osw_measurement_nc))
    assert cp.name == "cwave_params"
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).all()
    assert "polar grid" in cp.attrs["long_name"]


def test_compute_cwaves_from_osw_polar_custom_cwave(osw_measurement_nc):
    cp = compute_cwaves_from_osw_polar(
        str(osw_measurement_nc),
        cwave_wlmax=800,
        cwave_wlmin=30,
        cwave_Nk=3,
        cwave_Nphi=3,
    )
    assert cp.sizes["k_gp"] == 3
    assert cp.sizes["phi_hf"] == 3


def test_compute_cwaves_from_osw_polar_missing_file():
    with pytest.raises(FileNotFoundError):
        compute_cwaves_from_osw_polar("/nonexistent/ocn.nc")


def test_entrypoint_osw_polar_direct(osw_measurement_nc, capsys):
    rc = entrypoint(["--input-nc-osw", str(osw_measurement_nc), "--polar-direct"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "polar-direct" in out
    assert "CWAVE parameters computed" in out


def test_entrypoint_osw_polar_direct_output_nc(osw_measurement_nc, tmp_path, capsys):
    out_nc = tmp_path / "cwave_polar.nc"
    rc = entrypoint(
        [
            "--input-nc-osw",
            str(osw_measurement_nc),
            "--polar-direct",
            "--output-nc",
            str(out_nc),
        ]
    )
    assert rc == 0
    assert out_nc.exists()
    with xr.open_dataset(out_nc) as out:
        assert out.sizes["k_gp"] == 4
        assert "cwave_params" in out


# =============================================================================
# input validation
# =============================================================================


def test_polar_k_not_positive():
    spec, k, phi = _synthetic_polar(64, 72)
    bad_k = k.copy()
    bad_k[0] = -k[0]
    with pytest.raises(ValueError, match="k must be strictly positive"):
        compute_cwave_parameters_polar(spec, bad_k, phi)


def test_polar_k_not_increasing():
    spec, k, phi = _synthetic_polar(64, 72)
    bad_k = k.copy()
    bad_k[-1] = bad_k[-2]
    with pytest.raises(ValueError, match="k must be strictly increasing"):
        compute_cwave_parameters_polar(spec, bad_k, phi)


def test_polar_k_too_short():
    spec, k, phi = _synthetic_polar(64, 72)
    with pytest.raises(ValueError, match="at least 2 samples"):
        compute_cwave_parameters_polar(spec, k[:1], phi)


def test_polar_phi_not_increasing():
    spec, k, phi = _synthetic_polar(64, 72)
    bad_phi = phi.copy()
    bad_phi[-1] = bad_phi[-2]
    with pytest.raises(ValueError, match="phi must be strictly increasing"):
        compute_cwave_parameters_polar(spec, k, bad_phi)


def test_polar_phi_full_circle_rejected():
    """A grid that repeats 0 and 2*pi (span == 2*pi) is not a valid circle."""
    spec, k, phi = _synthetic_polar(64, 72)
    bad_phi = np.linspace(0.0, 2.0 * np.pi, len(phi))  # span == 2*pi
    with pytest.raises(ValueError, match="full 2\\*pi circle"):
        compute_cwave_parameters_polar(spec, k, bad_phi)


def test_polar_shape_mismatch():
    spec, k, phi = _synthetic_polar(64, 72)
    with pytest.raises(ValueError, match="does not match"):
        compute_cwave_parameters_polar(spec, k[:-1], phi)


def test_polar_spec_not_2d():
    spec, k, phi = _synthetic_polar(64, 72)
    with pytest.raises(ValueError, match="must be 2-D"):
        compute_cwave_parameters_polar(spec.ravel(), k, phi)


def test_polar_zero_mass():
    spec, k, phi = _synthetic_polar(64, 72)
    with pytest.raises(ValueError, match="zero.*mass"):
        compute_cwave_parameters_polar(np.zeros_like(spec), k, phi)


def test_polar_zero_mass_outside_annulus():
    """Spectrum energy entirely below kmin -> nothing to normalise."""
    spec, k, phi = _synthetic_polar(64, 72)
    zero = np.zeros_like(spec)
    zero[:, 0] = 1.0  # energy only in the lowest (k < kmin) bin
    with pytest.raises(ValueError, match="zero.*mass"):
        compute_cwave_parameters_polar(zero, k, phi)
