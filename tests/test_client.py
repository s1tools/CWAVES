"""Tests for the CWAVE CLI client (``cwaves.client``).

Covers both supported input products :
- the ESA OCN WV Sentinel-1 *measurement* fixture (``oswPolSpec``, polar to
  cartesian) shipped in ``assets/``,
- the reduced SARWave *L1B XSP* cross-spectrum fixture (``xspectra_{lag}tau``)
  also shipped in ``assets/``.
"""

import numpy as np
import pytest
import xarray as xr
from cwaves.client import (
    compute_cwaves_from_l1b,
    compute_cwaves_from_osw,
    entrypoint,
    polar_to_cartesian,
)

# ---------------------------------------------------------------------------
# OCN WV measurement product (oswPolSpec, polar -> cartesian)
# ---------------------------------------------------------------------------


def test_polar_to_cartesian_shape(osw_measurement_nc):
    with xr.open_dataset(osw_measurement_nc) as ds:
        xs = polar_to_cartesian(ds, n_range=32, n_azimuth=64)
    assert xs.sizes["freq_sample"] == 32
    assert xs.sizes["freq_line"] == 64
    # one-sided range (k_rg >= 0), two-sided azimuth
    assert xs.k_rg.min() >= 0
    assert xs.k_az.min() < 0 < xs.k_az.max()
    assert np.iscomplexobj(xs)
    assert np.isfinite(xs.values).any()


def test_compute_cwaves_from_osw(osw_measurement_nc):
    cp = compute_cwaves_from_osw(str(osw_measurement_nc), n_range=32, n_azimuth=64)
    assert cp.name == "cwave_params"
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).all()


def test_compute_cwaves_from_osw_custom_cwave(osw_measurement_nc):
    cp = compute_cwaves_from_osw(
        str(osw_measurement_nc),
        n_range=32,
        n_azimuth=64,
        cwave_wlmax=800,
        cwave_wlmin=30,
        cwave_Nk=3,
        cwave_Nphi=3,
    )
    assert cp.sizes["k_gp"] == 3
    assert cp.sizes["phi_hf"] == 3


def test_entrypoint_osw(osw_measurement_nc, capsys):
    rc = entrypoint(
        [
            "--input-nc-osw",
            str(osw_measurement_nc),
            "--n-range",
            "32",
            "--n-azimuth",
            "64",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # logger.info lines are written to stdout
    assert "CWAVE parameters computed" in out
    assert "CWAVE parameters" in out


def test_entrypoint_osw_with_cwave_options(osw_measurement_nc, capsys):
    rc = entrypoint(
        [
            "--input-nc-osw",
            str(osw_measurement_nc),
            "--n-range",
            "32",
            "--n-azimuth",
            "64",
            "--cwave-wlmax",
            "800",
            "--cwave-wlmin",
            "30",
            "--cwave-nk",
            "3",
            "--cwave-nphi",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 kernels (k_gp) x 3 (phi_hf)" in out


# ---------------------------------------------------------------------------
# SARWave L1B XSP cross-spectrum product (xspectra_{lag}tau)
# ---------------------------------------------------------------------------


def test_compute_cwaves_from_l1b(l1b_xsp_nc):
    cp = compute_cwaves_from_l1b(str(l1b_xsp_nc))
    assert cp.name == "cwave_params"
    assert cp.sizes["time"] == 2
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).all()


def test_compute_cwaves_from_l1b_lag1(l1b_xsp_nc):
    cp = compute_cwaves_from_l1b(str(l1b_xsp_nc), lag=1)
    assert cp.sizes["time"] == 2
    assert cp.sizes["k_gp"] == 4
    assert cp.sizes["phi_hf"] == 5
    assert np.isfinite(cp.values).all()


def test_compute_cwaves_from_l1b_custom_cwave(l1b_xsp_nc):
    cp = compute_cwaves_from_l1b(
        str(l1b_xsp_nc),
        cwave_wlmax=800,
        cwave_wlmin=30,
        cwave_Nk=3,
        cwave_Nphi=3,
    )
    assert cp.sizes["k_gp"] == 3
    assert cp.sizes["phi_hf"] == 3


def test_entrypoint_l1b(l1b_xsp_nc, capsys):
    rc = entrypoint(
        [
            "--input-nc-l1b",
            str(l1b_xsp_nc),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "CWAVE parameters computed" in out
    assert "CWAVE parameters" in out


def test_entrypoint_l1b_output_nc(l1b_xsp_nc, tmp_path, capsys):
    out_nc = tmp_path / "cwave_out.nc"
    rc = entrypoint(
        [
            "--input-nc-l1b",
            str(l1b_xsp_nc),
            "--output-nc",
            str(out_nc),
        ]
    )
    assert rc == 0
    assert out_nc.exists()
    with xr.open_dataset(out_nc) as out:
        assert out.sizes["time"] == 2
        assert "cwave_params" in out
    assert "saved CWAVE parameters to" in capsys.readouterr().out


def test_entrypoint_l1b_with_cwave_options(l1b_xsp_nc, capsys):
    rc = entrypoint(
        [
            "--input-nc-l1b",
            str(l1b_xsp_nc),
            "--cwave-wlmax",
            "800",
            "--cwave-wlmin",
            "30",
            "--cwave-nk",
            "3",
            "--cwave-nphi",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 kernels (k_gp) x 3 (phi_hf)" in out


# ---------------------------------------------------------------------------
# entrypoint error handling
# ---------------------------------------------------------------------------


def test_entrypoint_missing_file(capsys):
    rc = entrypoint(["--input-nc-osw", "/nonexistent/file.nc"])
    assert rc == 1
    err_out = capsys.readouterr().out
    assert "failed to compute CWAVE parameters" in err_out


def test_compute_cwaves_from_l1b_missing_file():
    with pytest.raises(FileNotFoundError):
        compute_cwaves_from_l1b("/nonexistent/l1b.nc")


def test_entrypoint_no_input(capsys):
    rc = entrypoint([])
    assert rc == 1
    err_out = capsys.readouterr().out
    assert "exactly one of" in err_out


def test_entrypoint_both_inputs(capsys, osw_measurement_nc, l1b_xsp_nc):
    rc = entrypoint(
        [
            "--input-nc-osw",
            str(osw_measurement_nc),
            "--input-nc-l1b",
            str(l1b_xsp_nc),
        ]
    )
    assert rc == 1
    assert "exactly one of" in capsys.readouterr().out
