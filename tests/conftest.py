"""Test fixtures for the cwaves unit tests.

The mocks reproduce the structure of the ``xsar_slc`` cross-spectra (see
``xsar_slc/xsarslc/tests/test_methods_related_to_xspectra.py``) so that the
``cwaves`` functions can be exercised without network access or large files.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

DIMS = ("burst", "tile_sample", "tile_line", "pol", "freq_line", "freq_sample")

#: path of the ESA OCN WV Sentinel-1 measurement fixture shipped in assets/
ASSETS_NC = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "s1d-wv1-ocn-vv-20260802t194456-20260802t194459-003951-00729e-013.nc"
)

#: path of the reduced SARWave L1B XSP cross-spectrum fixture shipped in assets/
ASSETS_L1B_NC = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "s1a_wv1_xsp_20250501t060101_058996_075117_dd24_a24.nc"
)

#: path of the reduced OCN IW osw per-subswath fixture (sinh grid, m/m units)
ASSETS_OCN_IW1_NC = (
    Path(__file__).resolve().parent.parent / "assets" / "s1d-iw1-osw-vv-fixture.nc"
)

#: path of the reduced SARWAVE L1B IW XSP per-subswath fixture (uniform, rad/m)
ASSETS_SARWAVE_IW2_NC = (
    Path(__file__).resolve().parent.parent / "assets" / "l1b-s1a-iw2-xsp-fixture.nc"
)


@pytest.fixture
def osw_measurement_nc():
    """Path of the ESA OCN WV Sentinel-1 measurement NetCDF fixture."""
    return ASSETS_NC


@pytest.fixture
def l1b_xsp_nc():
    """Path of the reduced SARWave L1B XSP cross-spectrum NetCDF fixture."""
    return ASSETS_L1B_NC


@pytest.fixture
def ocn_iw1_nc():
    """Path of the reduced OCN IW osw per-subswath NetCDF fixture."""
    return ASSETS_OCN_IW1_NC


@pytest.fixture
def sarwave_iw2_nc():
    """Path of the reduced SARWAVE L1B IW XSP per-subswath NetCDF fixture."""
    return ASSETS_SARWAVE_IW2_NC


def _make_mock(shape):
    """Build a mock one-sided (positive range) cross-spectrum."""
    n_burst, n_sample, n_line, n_pol, n_freq_line, n_freq_sample = shape
    rng = np.random.default_rng(42)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    coords = {
        "burst": [0] * n_burst,
        "pol": ["VH", "VV"][:n_pol],
        "tile_sample": np.arange(n_sample),
        "tile_line": np.arange(n_line),
        "k_az": ("freq_line", np.linspace(-0.0447, 0.0429, n_freq_line)),
    }
    k_rg_base = np.linspace(0, 0.896, n_freq_sample)
    k_rg_data = np.zeros((n_line, n_sample, n_freq_sample))
    for i in range(n_line):
        for j in range(n_sample):
            k_rg_data[i, j, :] = k_rg_base + (i * 0.00001) + (j * 0.000005)
    coords["k_rg"] = (("tile_line", "tile_sample", "freq_sample"), k_rg_data)
    coords["line"] = (("burst", "tile_line"), np.arange(n_line)[None, :])
    coords["sample"] = (
        ("burst", "tile_sample"),
        np.arange(n_sample)[None, :],
    )
    coords["longitude"] = (
        ("burst", "tile_line", "tile_sample"),
        rng.uniform(41, 42, (n_burst, n_line, n_sample)),
    )
    coords["latitude"] = (
        ("burst", "tile_line", "tile_sample"),
        rng.uniform(-16, -15, (n_burst, n_line, n_sample)),
    )
    return xr.DataArray(data, dims=DIMS, coords=coords, name="xspectra_2tau")


@pytest.fixture
def mock_xspectra_da():
    """A mock cross-spectrum close to the original ``xsar_slc`` size."""
    return _make_mock((1, 15, 4, 2, 50, 506))


@pytest.fixture
def mock_xspectra_da_small():
    """A small mock cross-spectrum for fast unit testing."""
    return _make_mock((1, 5, 2, 2, 20, 100))
