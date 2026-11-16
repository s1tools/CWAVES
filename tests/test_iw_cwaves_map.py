"""Tests for the IW subswath CWAVE map script (``cwaves.scripts.iw_cwaves_map``).

Uses two reduced per-subswath fixtures shipped in ``assets/`` :
- an OCN IW osw file (sinh grid, ``m/m`` units),
- a SARWAVE L1B IW XSP file (uniform grid, ``rad/m``).
"""

import shutil

import numpy as np
import pytest
import xarray as xr
from cwaves.scripts.iw_cwaves_map import (
    compute_ocn_subswath,
    compute_sarwave_subswath,
    default_pol,
    detect_product,
    entrypoint,
    ocn_tile_spectrum,
    plot_map,
)

# ---------------------------------------------------------------------------
# detect_product / default_pol
# ---------------------------------------------------------------------------


def test_detect_product_ocn(ocn_iw1_nc):
    # the fixture directory contains s1a-iw1-osw-*.nc only
    assert detect_product(str(ocn_iw1_nc.parent)) == "ocn"


def test_detect_product_sarwave(sarwave_iw2_nc, tmp_path):
    # build a tiny SAFE-like dir containing a l1b-s1?-iw?-xsp file
    safe = tmp_path / "S1A_IW_XSP__1SDV_fake.SAFE"
    safe.mkdir()
    shutil.copy(sarwave_iw2_nc, safe / "l1b-s1b-iw2-xsp-1sdv-fake.nc")
    assert detect_product(str(safe)) == "sarwave"


def test_detect_product_not_found(tmp_path):
    with pytest.raises(ValueError):
        detect_product(str(tmp_path))


def test_default_pol_by_mode():
    assert default_pol("SDV") == "VV"
    assert default_pol("SSV") == "VV"
    assert default_pol("SDH") == "HH"
    assert default_pol("SSH") == "HH"
    assert default_pol("XXX") == "VV"  # fallback


# ---------------------------------------------------------------------------
# ocn_tile_spectrum (native sinh grid, no resampling)
# ---------------------------------------------------------------------------


def test_ocn_tile_spectrum_native_sinh_grid(ocn_iw1_nc):
    with xr.open_dataset(ocn_iw1_nc, group="intraburst") as ds:
        spec = ocn_tile_spectrum(ds, 0, 0, 2)
        kx = ds["oswKx"].isel(oswAzSize=0, oswRaSize=0).values
    # same samples as the product (rad/m), non-uniform spacing kept as is
    np.testing.assert_allclose(spec.k_rg.values, kx[np.isfinite(kx)])
    d = np.diff(spec.k_rg.values)
    assert d.std() / d.mean() > 0.05  # sinh grid: genuinely non-uniform
    assert spec.k_az.min() < 0 < spec.k_az.max()
    assert np.iscomplexobj(spec)


def test_ocn_iw_fixture_wavenumber_contract(ocn_iw1_nc):
    # the fixture must satisfy the cross-spectrum contract : one-sided range
    # (k >= 0) and a two-sided, symmetric azimuth (min ~= -max). A lopsided
    # azimuth crop (the historical defect) would bias the CWAVE projection.
    for group in ("intraburst", "interburst"):
        with xr.open_dataset(ocn_iw1_nc, group=group) as ds:
            kx = ds["oswKx"].isel(oswAzSize=0, oswRaSize=0).values
            ky = ds["oswKy"].isel(oswAzSize=0, oswRaSize=0).values
        assert float(kx.min()) >= 0.0  # one-sided range
        assert ky.min() < 0 < ky.max()  # two-sided azimuth
        assert np.isclose(ky.min(), -ky.max(), atol=1e-5)  # symmetric about 0


def test_ocn_tile_spectrum_drops_nan_bins(ocn_iw1_nc):
    # some OCN interburst tiles carry trailing NaN oswKx/oswKy bins: the grid
    # lines are dropped so that the quadrature sees a finite monotonic grid
    with xr.open_dataset(ocn_iw1_nc, group="intraburst") as ds:
        ds = ds.load()
    kx = ds["oswKx"].values.copy()
    ky = ds["oswKy"].values.copy()
    kx[0, 0, -2:] = np.nan
    ky[0, 0, -2:] = np.nan
    ds["oswKx"] = (ds["oswKx"].dims, kx)
    ds["oswKy"] = (ds["oswKy"].dims, ky)
    spec = ocn_tile_spectrum(ds, 0, 0, 2)
    assert np.isfinite(spec.k_rg.values).all()
    assert np.isfinite(spec.k_az.values).all()
    n_rg = np.isfinite(ds["oswKx"].isel(oswAzSize=0, oswRaSize=0).values).sum()
    assert spec.k_rg.size == n_rg


# ---------------------------------------------------------------------------
# per-subswath computation
# ---------------------------------------------------------------------------


def test_compute_ocn_subswath(ocn_iw1_nc):
    lons, lats, params = compute_ocn_subswath(str(ocn_iw1_nc), "intraburst", 2)
    assert lons.shape == (2, 2)
    assert lats.shape == (2, 2)
    assert params.shape == (2, 2, 4, 5)
    assert np.isfinite(params).mean() > 0.5


def test_compute_ocn_subswath_lag1(ocn_iw1_nc):
    _, _, params = compute_ocn_subswath(str(ocn_iw1_nc), "intraburst", 1)
    assert params.shape == (2, 2, 4, 5)


def test_sarwave_iw_fixture_wavenumber_contract(sarwave_iw2_nc):
    # the fixture must satisfy the cross-spectrum contract : one-sided range
    # (k_rg >= 0) and a two-sided, symmetric azimuth (min ~= -max).
    with xr.open_dataset(sarwave_iw2_nc, group="intraburst") as ds:
        k_rg = ds["k_rg"].isel(tile_line=0, tile_sample=0).values
        k_az = ds["k_az"].values
    assert float(k_rg.min()) >= 0.0  # one-sided range
    assert k_az.min() < 0 < k_az.max()  # two-sided azimuth
    assert np.isclose(k_az.min(), -k_az.max(), atol=1e-5)  # symmetric about 0


def test_compute_sarwave_subswath(sarwave_iw2_nc):
    lons, lats, params = compute_sarwave_subswath(
        str(sarwave_iw2_nc), "intraburst", 2, "VV"
    )
    assert lons.shape == (3, 2)
    assert lats.shape == (3, 2)
    assert params.shape == (3, 2, 4, 5)
    assert np.isfinite(params).mean() > 0.5


def test_compute_sarwave_subswah_bad_pol(sarwave_iw2_nc):
    with pytest.raises(ValueError, match="polarization"):
        compute_sarwave_subswath(str(sarwave_iw2_nc), "intraburst", 2, "XX")


# ---------------------------------------------------------------------------
# plot_map / entrypoint
# ---------------------------------------------------------------------------


def test_plot_map(tmp_path, ocn_iw1_nc):
    lons, lats, params = compute_ocn_subswath(str(ocn_iw1_nc), "intraburst", 2)
    out = tmp_path / "map.png"
    plot_map({"iw1": (lons, lats, params)}, 1, 1, str(out), "test map")
    assert out.exists()
    assert out.stat().st_size > 0


def test_entrypoint_ocn(ocn_iw1_nc, tmp_path):
    out = tmp_path / "ocn_map.png"
    rc = entrypoint(
        [
            "--input-safe",
            str(ocn_iw1_nc.parent),
            "--group",
            "intraburst",
            "--lag",
            "2",
            "--pol",
            "VV",
            "--cwave-nk",
            "3",
            "--cwave-nphi",
            "7",
            "--output-png",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_compute_ocn_subswath_interburst(ocn_iw1_nc):
    _, _, params = compute_ocn_subswath(str(ocn_iw1_nc), "interburst", 2)
    assert params.shape[2] == 4
    assert params.shape[3] == 5


def test_subswath_params_shape_follows_basis_size(ocn_iw1_nc, sarwave_iw2_nc):
    # the output size follows cwave_Nk x cwave_Nphi (no hard-coded 4 x 5)
    _, _, p_ocn = compute_ocn_subswath(
        str(ocn_iw1_nc), "intraburst", 2, cwave_Nk=3, cwave_Nphi=7
    )
    assert p_ocn.shape[2:] == (3, 7)
    assert np.isfinite(p_ocn).any()
    _, _, p_sar = compute_sarwave_subswath(
        str(sarwave_iw2_nc), "intraburst", 2, "VV", cwave_Nk=6, cwave_Nphi=3
    )
    assert p_sar.shape[2:] == (6, 3)


def test_entrypoint_sarwave(sarwave_iw2_nc, tmp_path):
    safe = tmp_path / "S1A_IW_XSP__1SDV_fake.SAFE"
    safe.mkdir()
    shutil.copy(sarwave_iw2_nc, safe / "l1b-s1b-iw2-xsp-1sdv-fake.nc")
    out = tmp_path / "sarwave_map.png"
    rc = entrypoint(
        [
            "--input-safe",
            str(safe),
            "--group",
            "intraburst",
            "--lag",
            "2",
            "--pol",
            "VV",
            "--output-png",
            str(out),
        ]
    )
    # iw1/iw3 are absent from this fixture but iw2 is processed -> success
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_entrypoint_missing_input(tmp_path):
    rc = entrypoint(["--input-safe", str(tmp_path / "nope.SAFE")])
    assert rc == 1


def test_entrypoint_no_matching_pol(ocn_iw1_nc, tmp_path):
    # the osw fixture is VV only ; asking for HH yields no per-subswath file
    rc = entrypoint(["--input-safe", str(ocn_iw1_nc.parent), "--pol", "HH"])
    assert rc == 1
