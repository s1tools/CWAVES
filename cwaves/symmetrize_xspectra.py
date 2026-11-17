"""Symmetrization of one-sided (positive range) SAR cross-spectra."""

import warnings

import numpy as np
import xarray as xr


def symmetrize_xspectrum(xs, dim_range="freq_sample", dim_azimuth="freq_line"):
    """
    Symmetrize half (positive range only) cross-spectrum. xs should be complex and assumed to contain only positive wavenumbers in range.

    Args:
        xs (xarray.DataArray or xarray.Dataset): complex half xspectra (it is supposed to have only positive wavenumber along range axis)
        dim_range (str) : name of the dimension (index) in the range direction
        dim_azimuth (str) : name of the dimension (index) in the azimuth direction
    Return:
        (xarray.DataArray or xarray.Dataset): symmetrized spectra (as they were generated in the first place using fft)
    """
    if not np.iscomplexobj(xs):
        warnings.warn(
            "You are trying to symmetrize a real spectrum. It should probably be complex.",
            stacklevel=2,
        )
    if dim_range not in xs.sizes or dim_azimuth not in xs.sizes:
        raise ValueError(
            f"dim_range and dim_azimuth must be indexers {xs.dims}. ({dim_range} and/or {dim_azimuth} are probably only coordinates)"
        )

    even_azimuth = False
    if (
        not xs.sizes[dim_azimuth] % 2
    ):  # if even, xs has to be forced to be perfectly symmetric to enable correct merging of mirror with xs
        even_azimuth = True
        kaz2add = xs[{dim_azimuth: 0}]  # azimuth wavenumber to be symmetrized
        for cname, c in xs.coords.items():
            if dim_azimuth in c.dims:
                kaz2add = kaz2add.assign_coords({cname: -kaz2add[cname]})
        xs = xr.concat([xs, kaz2add], dim=dim_azimuth)

    mirror = np.conj(
        xs[{dim_azimuth: slice(None, None, -1), dim_range: slice(None, 0, -1)}]
    )  # This line remove zero value on freq_sample
    for cname, c in mirror.coords.items():
        if dim_range in c.dims or dim_azimuth in c.dims:
            mirror = mirror.assign_coords({cname: -mirror[cname]})
    res = xr.concat(
        [mirror, xs[{dim_range: slice(None, -1)}]], dim=dim_range
    )  # concat and drop last positive k_range to keep fft freq convention

    if even_azimuth:  # To ensure to always return an even spectrum on azimuth dimension
        res = res[{dim_azimuth: slice(None, -1)}]
    return res
