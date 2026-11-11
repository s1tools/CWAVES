"""Unit tests for the OCN WV sinh-cartesian cross-spectrum decode (``ocn_sinh``)."""

import numpy as np
import pytest
import xarray as xr
from cwaves.compute_cwaves import compute_cwave_parameters
from cwaves.config import get_default_config
from cwaves.ocn_sinh import (
    _physical_scale,
    _sinh2linear_mapper,
    compute_cwaves_from_ocn_sinh,
    compute_ocn_sinh_cartesian,
    reconstruct_full_cart_spec,
)

KMAX = 2.0 * np.pi / 25.0


def test_reconstruct_full_cart_spec_symmetry():
    """The full spectrum must be Hermitian (Re even, Im odd) along the range axis."""
    rng = np.random.default_rng(0)
    n_az, n_ra = 7, 5
    re = rng.standard_normal((n_az, n_ra))
    im = rng.standard_normal((n_az, n_ra))
    full_re, full_im = reconstruct_full_cart_spec(re, im)
    assert full_re.shape == (n_az, 2 * n_ra - 1)
    assert full_im.shape == (n_az, 2 * n_ra - 1)
    # the stored (right) half is preserved as-is (central column = range k = 0)
    mid = n_ra - 1
    assert np.allclose(full_re[:, mid:], re)
    assert np.allclose(full_im[:, mid:], im)
    # the left half mirrors the stored half with BOTH range and azimuth reversed
    assert np.allclose(full_re[:, :mid], re[::-1, ::-1][:, :-1])
    assert np.allclose(full_im[:, :mid], -im[::-1, ::-1][:, :-1])


def test_reconstruct_full_cart_spec_hermitian_real(osw_measurement_nc):
    """On a real OCN tile the full spectrum is 2D-Hermitian (Re even, Im odd)."""
    with xr.open_dataset(osw_measurement_nc) as ds:
        re = (
            ds["oswCartSpecRe"]
            .isel(oswAzSize=0, oswRaSize=0, oswLag=2)
            .transpose("oswKyBinSize", "oswKxBinSize")
            .values
        )
        im = (
            ds["oswCartSpecIm"]
            .isel(oswAzSize=0, oswRaSize=0, oswLag=2)
            .transpose("oswKyBinSize", "oswKxBinSize")
            .values
        )
    n_az, n_ra = re.shape
    full_re, full_im = reconstruct_full_cart_spec(re, im)
    mid = n_ra - 1
    max_re = max_im = 0.0
    for d in range(1, n_ra):
        # range mirror kx -> -kx, azimuth mirror ky -> -ky (2D Hermitian)
        max_re = max(
            max_re, float(np.max(np.abs(full_re[:, mid + d] - full_re[::-1, mid - d])))
        )
        max_im = max(
            max_im, float(np.max(np.abs(full_im[:, mid + d] + full_im[::-1, mid - d])))
        )
    assert max_re < 1e-9
    assert max_im < 1e-9


def test_sinh2linear_mapper_shape_and_center():
    q, k_inv = _sinh2linear_mapper(13, 20, 1.0 / 30.1)
    assert q.shape == (20, 13)
    # the linear grid is centered : the middle index maps to k_inv ~ 0
    assert abs(float(k_inv[len(k_inv) // 2])) < 1e-9


def test_compute_ocn_sinh_cartesian_grid(osw_measurement_nc):
    """The decoded spectrum must be one-sided, uniform, and in a sane rad/m range."""
    with xr.open_dataset(osw_measurement_nc) as ds:
        xs = compute_ocn_sinh_cartesian(ds, lag=2, m_range=256, m_azimuth=256)
    assert xs.dims == ("freq_sample", "freq_line")
    assert np.iscomplexobj(xs.values)
    # one-sided range (k_rg >= 0)
    assert float(xs.k_rg.min()) >= 0.0
    # uniform grid
    assert np.allclose(np.diff(xs.k_rg), np.diff(xs.k_rg)[0])
    assert np.allclose(np.diff(xs.k_az), np.diff(xs.k_az)[0])
    # two-sided azimuth
    assert float(xs.k_az.min()) < 0 < float(xs.k_az.max())
    # physical extents : within the CWAVE domain, azimuth extent ~ O(0.05) rad/m
    assert float(xs.k_rg.max()) < KMAX
    assert 0.02 < abs(float(xs.k_az.max())) < 0.2
    assert np.isfinite(xs.values).all()


def test_physical_scale_from_product(osw_measurement_nc):
    with xr.open_dataset(osw_measurement_nc) as ds:
        kx_idx = np.arange(256) - 128
        ky_idx = np.arange(256) - 128
        kx, ky = _physical_scale(ds, kx_idx, ky_idx)
    # uniform grid (even M => k spans [-M/2, M/2 - 1], centered within one cell)
    assert np.allclose(np.diff(kx), np.diff(kx)[0])
    assert np.allclose(np.diff(ky), np.diff(ky)[0])
    # mean is offset by half a cell (grid spans [-M/2, M/2 - 1])
    assert abs(float(np.mean(kx)) + float(np.diff(kx)[0]) / 2.0) < 1e-12
    assert abs(float(np.mean(ky)) + float(np.diff(ky)[0]) / 2.0) < 1e-12
    # the range extent must exceed the azimuth extent (range-elongated domain)
    assert abs(float(kx.max())) > abs(float(ky.max()))


def test_compute_ocn_sinh_cwave(osw_measurement_nc):
    """CWAVE computed from the decoded spectrum must be finite and well-sized."""
    cfg = get_default_config()["cwave"]
    with xr.open_dataset(osw_measurement_nc) as ds:
        xs = compute_ocn_sinh_cartesian(ds, lag=2)
        cp = compute_cwave_parameters(xs, **cfg)
    assert cp.sizes["k_gp"] == cfg["cwave_Nk"]
    assert cp.sizes["phi_hf"] == cfg["cwave_Nphi"]
    assert np.isfinite(cp.values).all()


def test_compute_cwaves_from_ocn_sinh_cli(osw_measurement_nc):
    cp = compute_cwaves_from_ocn_sinh(str(osw_measurement_nc), lag=2)
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).all()


def test_missing_vars_raise():
    with pytest.raises(ValueError, match="oswCartSpecRe"):
        compute_ocn_sinh_cartesian(xr.Dataset({"foo": ("x", [1.0])}), lag=2)
