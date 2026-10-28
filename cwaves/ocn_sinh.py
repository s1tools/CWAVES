"""Decompress the OCN WV cartesian cross-spectrum (sinh grid) to a regular grid.

The ESA OCN WV Sentinel-1 measurement stores the cartesian cross-spectrum
(``oswCartSpecRe`` / ``oswCartSpecIm``) on a **sinh (non-uniform) wavenumber
grid** in normalized (``m/m``) units. :func:`compute_cwave_parameters` needs a
**uniform** grid in **rad/m**, so this module performs the two decoding steps
mirroring the IFREMER ``l2_osw_decompressor`` :

1. **Full-spectrum reconstruction** : the product stores only the
   range-``>= 0`` half; the full two-sided range spectrum is recovered from the
   Hermitian symmetry of the SAR cross-spectrum (Re even, Im odd).
2. **Sinh -> linear decompression** : the (Re, Im) are mapped from the sinh grid
   to a regular linear grid with a sinc mapping matrix.

The normalized (``m/m``) output grid is then **calibrated to physical wavenumbers
(rad/m)** from the product's own arrays (no external ``int_pp2`` needed) : the
range (``dkx``) and azimuth (``dky``) cell spacings are computed independently
from ``oswKy`` / ``oswAziSize`` (azimuth) and ``raSamplingFreq`` /
``owiIncidenceAngle`` (range), matching the IFREMER ``get_k_matrix``.

This complements :func:`cwaves.client.polar_to_cartesian`, which goes through the
polar spectrum ``oswPolSpec`` : both are valid ways to obtain a regular cartesian
cross-spectrum from an OCN WV product.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import xarray as xr

#: default sinh growth rate (``resGrowthRate``), the IFREMER processing default.
DEFAULT_GROWTH_RATE = 1.0 / 30.1

#: default range sub-image size (``xsubsize``), the IFREMER int_pp2 default.
DEFAULT_XSUBSIZE = 1024

#: speed of light (m/s)
_SPEED_OF_LIGHT = 299792458.0


def _sinc(arg: np.ndarray) -> np.ndarray:
    """Normalized sinc (``sin(x) / x``, 1 at ``x = 0``)."""
    out = np.ones_like(arg, dtype=np.float64)
    ind = arg != 0
    out[ind] = np.sin(arg[ind]) / arg[ind]
    return out


def _sinh2linear_mapper(
    n_sinh: int, n_linear: int, growth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sinc mapping matrix from the ``n_sinh``-point sinh grid to a linear grid.

    Args:
        n_sinh: number of points of the sinh (compressed) grid.
        n_linear: number of points of the regular linear output grid.
        growth: the sinh growth rate (``resGrowthRate``).

    Returns:
        A tuple ``(Q, k_inv)`` where ``Q`` is the ``(n_linear, n_sinh)`` mapping
        matrix and ``k_inv`` the linear-grid coordinates (normalized units).
    """
    kappa = np.arange(n_sinh) - np.floor(n_sinh / 2)
    k = np.arange(n_linear) - np.floor(n_linear / 2)
    k_inv = np.arcsinh(growth * k) / growth
    q = np.zeros((n_linear, n_sinh), dtype=np.float64)
    for i in range(n_sinh):
        q[:, i] = (
            _sinc(np.pi * (k_inv - kappa[i]))
            * np.cosh(growth * kappa[i])
            / np.cosh(growth * k_inv)
        )
    return q, k_inv


def reconstruct_full_cart_spec(
    re: np.ndarray, im: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the full two-sided range spectrum from the stored half.

    The OCN product stores only the range-``>= 0`` half of the cartesian
    cross-spectrum. The full range spectrum is recovered from the Hermitian
    symmetry of the SAR cross-spectrum (Re even, Im odd) along the range axis :
    ``full = [mirror, stored]`` (the shared central column is not duplicated).

    Args:
        re: the stored real part, dims ``(azimuth, range_half)``.
        im: the stored imaginary part, dims ``(azimuth, range_half)``.

    Returns:
        A tuple ``(full_re, full_im)`` of the reconstructed full spectra, dims
        ``(azimuth, 2 * range_half - 1)``.
    """
    full_re = np.hstack((re[::-1, ::-1][:, :-1], re))
    full_im = np.hstack((-im[::-1, ::-1][:, :-1], im))
    return full_re, full_im


def _physical_scale(
    ds: xr.Dataset,
    kx_idx: np.ndarray,
    ky_idx: np.ndarray,
    growth: float = DEFAULT_GROWTH_RATE,
    xsubsize: int = DEFAULT_XSUBSIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the decompressed linear-index grid to physical wavenumbers (rad/m).

    The decompressed spectrum is uniform in its linear index. The physical
    wavenumber is ``index * dk`` (a uniform rad/m grid, as in the IFREMER
    ``get_k_matrix``). The range (``dkx``) and azimuth (``dky``) cell spacings are
    computed **independently** from the product geometry : range and azimuth have
    independent physical scales, so tying them by a single ratio (the old
    ``oswKx.max()/oswKy.max()`` approach) under-estimates the azimuth scale and
    compresses the azimuth energy.

    * **azimuth** ``dky`` : ``mean(oswKy / (oswAziSize * KKy))`` with
      ``KKy = sinh(growth * kappa) / growth`` (the azimuth sinh mapper), matching
      the IFREMER ``get_k_matrix``.
    * **range** ``dkx`` : ``(xresfac * raSamplingFreq / (c * M_range)) *
      4 pi sin(inca)`` with ``xresfac = M_range / xsubsize``.

    Args:
        ds: the OCN WV dataset (must contain ``oswKy``, ``oswAziSize``,
            ``owiIncidenceAngle`` and the ``raSamplingFreq`` attribute).
        kx_idx: the range linear-index grid (``arange(M) - M // 2``).
        ky_idx: the azimuth linear-index grid (``arange(M) - M // 2``).
        growth: the sinh growth rate (``resGrowthRate``); default
            :data:`DEFAULT_GROWTH_RATE`.
        xsubsize: the range sub-image size (``xsubsize`` from the IFREMER
            ``int_pp2``); default :data:`DEFAULT_XSUBSIZE`.

    Returns:
        A tuple ``(kx, ky)`` of the physical wavenumber grids in rad/m.
    """
    osw_ky = ds["oswKy"].squeeze().values
    n_az = osw_ky.size
    kappa = np.arange(n_az) - n_az // 2
    kky = np.sinh(growth * kappa) / growth
    azi_size = float(np.ravel(ds["oswAziSize"].values)[0])
    dky = float(np.nanmean(osw_ky / (azi_size * kky)))

    ra_sample_freq = float(ds.attrs["raSamplingFreq"])
    inca = np.deg2rad(float(np.ravel(ds["owiIncidenceAngle"].values)[0]))
    m_range = kx_idx.size
    xresfac = m_range / xsubsize
    dkx = (xresfac * ra_sample_freq / (_SPEED_OF_LIGHT * m_range)) * (
        4.0 * np.pi * np.sin(inca)
    )
    return kx_idx * dkx, ky_idx * dky


def compute_ocn_sinh_cartesian(
    ds: xr.Dataset,
    lag: int = 2,
    m_range: int = 256,
    m_azimuth: int = 256,
    growth: float = DEFAULT_GROWTH_RATE,
    xsubsize: int = DEFAULT_XSUBSIZE,
    osw_az_idx: int = 0,
    osw_ra_idx: int = 0,
) -> xr.DataArray:
    """Compute the one-sided cartesian cross-spectrum of an OCN WV tile.

    Decodes the ``oswCartSpec`` of one ``iw`` tile from the sinh grid to a
    **regular** grid in rad/m (see module docstring), and returns the
    **one-sided (range ``k_rg >= 0``, two-sided azimuth)** complex
    cross-spectrum expected by :func:`cwaves.compute_cwave_parameters`.

    Args:
        ds: an opened OCN WV measurement dataset (must contain
            ``oswCartSpecRe`` / ``oswCartSpecIm`` / ``oswKy`` / ``oswAziSize`` /
            ``owiIncidenceAngle`` and the ``raSamplingFreq`` attribute).
        lag: the ``oswLag`` index (0/1/2; default 2 = the 2-tau cross-spectrum).
        m_range: number of range wavenumbers of the regular output grid.
        m_azimuth: number of azimuth wavenumbers of the regular output grid.
        growth: the sinh growth rate (``resGrowthRate``); default
            :data:`DEFAULT_GROWTH_RATE`.
        xsubsize: the range sub-image size (``xsubsize`` from the IFREMER
            ``int_pp2``); default :data:`DEFAULT_XSUBSIZE`.
        osw_az_idx: the ``oswAzSize`` tile index.
        osw_ra_idx: the ``oswRaSize`` tile index.

    Returns:
        The one-sided (``k_rg >= 0``) complex cross-spectrum, dims
        ``(freq_sample, freq_line)`` with ``k_rg`` / ``k_az`` coordinates in
        rad/m.
    """
    if "oswCartSpecRe" not in ds or "oswCartSpecIm" not in ds:
        raise ValueError("oswCartSpecRe / oswCartSpecIm not found in the dataset")

    if "oswAzSize" in ds.dims and "oswRaSize" in ds.dims:
        tile = ds.isel(oswAzSize=osw_az_idx, oswRaSize=osw_ra_idx)
    else:  # single-tile product (WV)
        tile = ds

    spec_re = tile["oswCartSpecRe"]
    spec_im = tile["oswCartSpecIm"]
    if "oswLag" in spec_re.dims:
        spec_re = spec_re.isel(oswLag=lag)
        spec_im = spec_im.isel(oswLag=lag)
    re = spec_re.transpose("oswKyBinSize", "oswKxBinSize").values
    im = spec_im.transpose("oswKyBinSize", "oswKxBinSize").values
    if not (np.isfinite(re).all() and np.isfinite(im).all()):
        raise ValueError("oswCartSpec contains NaN/inf (check the tile index)")

    # 1. full-spectrum reconstruction (range->=0 half -> full two-sided range)
    full_re, full_im = reconstruct_full_cart_spec(re, im)

    # 2. sinh -> regular decompression (separately per look axis)
    qra, _ = _sinh2linear_mapper(full_re.shape[1], m_range, growth)
    qaz, _ = _sinh2linear_mapper(full_re.shape[0], m_azimuth, growth)
    reg_re = (qra @ full_re.T @ qaz.T).T  # (azimuth, range)
    reg_im = (qra @ full_im.T @ qaz.T).T  # (azimuth, range)

    # 3. calibrate the uniform linear-index grid to physical rad/m
    kx_idx = np.arange(m_range) - np.floor(m_range / 2)
    ky_idx = np.arange(m_azimuth) - np.floor(m_azimuth / 2)
    kx, ky = _physical_scale(ds, kx_idx, ky_idx, growth=growth, xsubsize=xsubsize)

    # 4. keep the one-sided (range >= 0) half, expected by compute_cwave_parameters
    sel = kx >= 0
    reg_re = reg_re[:, sel]
    reg_im = reg_im[:, sel]
    kx1 = kx[sel]

    values = (reg_re + 1j * reg_im).T  # (range, azimuth) = (freq_sample, freq_line)
    return xr.DataArray(
        values,
        dims=("freq_sample", "freq_line"),
        coords={"k_rg": ("freq_sample", kx1), "k_az": ("freq_line", ky)},
        name="oswCartSpec_sinh",
    )


def compute_cwaves_from_ocn_sinh(
    input_nc: str,
    lag: int = 2,
    **cwave: Any,
) -> xr.DataArray:
    """Compute the CWAVE parameters of an OCN WV tile from the sinh cartesian grid.

    Args:
        input_nc: path to the OCN WV measurement NetCDF file.
        lag: the ``oswLag`` index (default 2 = the 2-tau cross-spectrum).
        **cwave: ``cwave_*`` parameters forwarded to
            :func:`cwaves.compute_cwave_parameters` (defaults from the packaged
            ``default_config.yml`` if not given).

    Returns:
        The CWAVE parameters ``xarray.DataArray`` (``k_gp`` x ``phi_hf``).
    """
    from cwaves.compute_cwaves import compute_cwave_parameters
    from cwaves.config import get_default_config

    if not cwave:
        cwave = get_default_config()["cwave"]
    with xr.open_dataset(input_nc) as ds:
        xs = compute_ocn_sinh_cartesian(ds, lag=lag)
        return cast("xr.DataArray", compute_cwave_parameters(xs, **cwave))
