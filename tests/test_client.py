"""Tests for the CWAVE CLI client (``cwaves.client``): schema detection and the
single entry point ``compute_cwaves_from_file`` / ``cwaves --input-nc``.

Fixtures shipped in ``assets/``: the ESA OCN WV measurement (cartesian
``oswCartSpec`` on its sinh grid and polar ``oswPolSpec``), the reduced SARWave L1B XSP WV cross-spectrum, and the tiled
IW products (spectra in NetCDF groups).
"""

import numpy as np
import pytest
import xarray as xr
from cwaves import compute_cwaves
from cwaves.client import (
    compute_cwaves_from_file,
    detect_schema,
    entrypoint,
    ocn_cartesian_spectrum,
    polar_spectrum,
)


def test_detect_schema(osw_measurement_nc, l1b_xsp_nc):
    assert detect_schema(osw_measurement_nc) == "polar"
    assert detect_schema(l1b_xsp_nc) == "xsp"


def test_detect_schema_grouped_product(ocn_iw1_nc, sarwave_iw2_nc):
    for nc in (ocn_iw1_nc, sarwave_iw2_nc):
        with pytest.raises(ValueError, match="iw_cwaves_map"):
            detect_schema(nc)


def test_detect_schema_unsupported(tmp_path):
    nc = tmp_path / "foo.nc"
    xr.Dataset({"foo": ("x", [1.0])}).to_netcdf(nc)
    with pytest.raises(ValueError, match="no supported spectrum"):
        detect_schema(nc)


def test_detect_schema_missing_file():
    with pytest.raises(FileNotFoundError):
        detect_schema("/nonexistent.nc")


def test_ocn_cartesian_file(osw_measurement_nc):
    cp = compute_cwaves_from_file(osw_measurement_nc)
    assert cp.attrs["schema"] == "ocn_cartesian"
    assert cp.attrs["grid"] == "cartesian"
    assert cp.attrs["coverage"] == "half"
    assert np.isfinite(cp.values).all()
    with xr.open_dataset(osw_measurement_nc) as ds:
        direct = compute_cwaves(ocn_cartesian_spectrum(ds))
    np.testing.assert_allclose(cp, direct)


def test_ocn_cartesian_spectrum_calibration(osw_measurement_nc):
    """Normalised m/m axes -> rad/m: central step = 2 pi / (1024 ground pixels)."""
    with xr.open_dataset(osw_measurement_nc) as ds:
        spec = ocn_cartesian_spectrum(ds)
        cell = {"oswAzSize": 0, "oswRaSize": 0}
        px_rg = float(ds["oswGroundRngSize"].isel(cell)) / float(
            ds["oswRaSizeSLC"].isel(cell)
        )
        px_az = float(ds["oswAziSize"].isel(cell)) / float(
            ds["oswAzSizeSLC"].isel(cell)
        )
    assert spec.dims == ("freq_line", "freq_sample")
    assert float(spec.k_rg.min()) == 0.0
    step_rg = float(spec.k_rg[1] - spec.k_rg[0])
    centre = spec.sizes["freq_line"] // 2
    step_az = float(spec.k_az[centre + 1] - spec.k_az[centre])
    np.testing.assert_allclose(step_rg, 2 * np.pi / (1024 * px_rg), rtol=1e-3)
    np.testing.assert_allclose(step_az, 2 * np.pi / (1024 * px_az), rtol=1e-3)


def test_ocn_cartesian_spectrum_errors(osw_measurement_nc):
    with xr.open_dataset(osw_measurement_nc) as ds:
        with pytest.raises(ValueError, match="not an OCN WV"):
            ocn_cartesian_spectrum(ds.drop_vars("oswGroundRngSize"))
        bad = ds.copy()
        bad["oswCartSpecRe"] = bad["oswCartSpecRe"] * np.nan
        with pytest.raises(ValueError, match="NaN"):
            ocn_cartesian_spectrum(bad)
    with pytest.raises(ValueError, match="ocn_spectrum"):
        compute_cwaves_from_file(osw_measurement_nc, ocn_spectrum="sinh")


def test_polar_file(osw_measurement_nc):
    cp = compute_cwaves_from_file(osw_measurement_nc, ocn_spectrum="polar")
    assert cp.attrs["schema"] == "polar"
    assert cp.attrs["grid"] == "polar"
    assert cp.attrs["coverage"] == "full"
    assert (cp.sizes["k_gp"], cp.sizes["phi_hf"]) == (4, 5)
    assert np.isfinite(cp.values).all()
    # same result as calling the library on the extracted spectrum
    with xr.open_dataset(osw_measurement_nc) as ds:
        direct = compute_cwaves(polar_spectrum(ds))
    np.testing.assert_allclose(cp, direct)


def test_polar_spectrum_radians(osw_measurement_nc):
    with xr.open_dataset(osw_measurement_nc) as ds:
        spec = polar_spectrum(ds)
    assert spec.dims == ("phi", "k")
    assert float(spec.phi.max()) < 2 * np.pi


def test_xsp_file(l1b_xsp_nc):
    cp = compute_cwaves_from_file(l1b_xsp_nc)
    assert cp.attrs["schema"] == "xsp"
    assert cp.attrs["coverage"] == "half"
    assert "time" in cp.dims
    assert np.isfinite(cp.values).all()


def test_xsp_file_lag1_custom_basis(l1b_xsp_nc):
    cp = compute_cwaves_from_file(l1b_xsp_nc, lag=1, cwave_Nk=3, cwave_Nphi=7)
    assert (cp.sizes["k_gp"], cp.sizes["phi_hf"]) == (3, 7)


def test_xsp_missing_lag(l1b_xsp_nc, tmp_path):
    with xr.open_dataset(l1b_xsp_nc) as ds:
        ds.drop_vars(["xspectra_1tau_Re", "xspectra_1tau_Im"]).to_netcdf(
            tmp_path / "x.nc"
        )
    with pytest.raises(ValueError, match="xspectra_1tau_Re"):
        compute_cwaves_from_file(tmp_path / "x.nc", lag=1)


@pytest.mark.parametrize("fixture", ["osw_measurement_nc", "l1b_xsp_nc"])
def test_entrypoint(fixture, request, tmp_path, capsys):
    out = tmp_path / "cw.nc"
    rc = entrypoint(
        [
            "--input-nc",
            str(request.getfixturevalue(fixture)),
            "--cwave-nk",
            "3",
            "--cwave-nphi",
            "3",
            "--cwave-wlmax",
            "500",
            "--cwave-wlmin",
            "30",
            "--output-nc",
            str(out),
        ]
    )
    assert rc == 0
    with xr.open_dataarray(out) as cp:
        assert (cp.sizes["k_gp"], cp.sizes["phi_hf"]) == (3, 3)
    assert "native" in capsys.readouterr().out


def test_entrypoint_errors(ocn_iw1_nc, capsys):
    assert entrypoint(["--input-nc", "/nonexistent.nc"]) == 1
    assert entrypoint(["--input-nc", str(ocn_iw1_nc)]) == 1
    assert "iw_cwaves_map" in capsys.readouterr().out


def test_entrypoint_complex_output(l1b_xsp_nc, tmp_path):
    out = tmp_path / "cw.nc"
    rc = entrypoint(
        ["--input-nc", str(l1b_xsp_nc), "--part", "complex", "--output-nc", str(out)]
    )
    assert rc == 0
    with xr.open_dataset(out) as ds:
        assert {"cwave_params_re", "cwave_params_im"} <= set(ds.data_vars)
        assert ds.attrs["part"] == "complex"
        assert np.abs(ds["cwave_params_im"]).max() > 0
