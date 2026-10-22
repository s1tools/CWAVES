"""Test fixtures for the cwaves unit tests."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def mock_xspectra_da():
    """A small mock one-sided (positive range) cross-spectrum.

    Matches the structure produced by ``xsar_slc`` (``freq_sample`` /
    ``freq_line`` are plain dimensions, ``k_rg`` / ``k_az`` carry the values).
    """
    rng = np.random.default_rng(42)
    dims = ("burst", "tile_sample", "tile_line", "pol", "freq_line", "freq_sample")
    shape = (1, 5, 2, 2, 20, 100)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    coords = {
        "burst": [0],
        "pol": ["VH", "VV"],
        "tile_sample": np.arange(5),
        "tile_line": np.arange(2),
        "k_az": ("freq_line", np.linspace(-0.0447, 0.0429, 20)),
    }
    k_rg_base = np.linspace(0, 0.896, 100)
    k_rg_data = np.zeros((2, 5, 100))
    for i in range(2):
        for j in range(5):
            k_rg_data[i, j, :] = k_rg_base + (i * 0.00001) + (j * 0.000005)
    coords["k_rg"] = (("tile_line", "tile_sample", "freq_sample"), k_rg_data)
    return xr.DataArray(data, dims=dims, coords=coords, name="xspectra_2tau")
