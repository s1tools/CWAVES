"""Tests for :mod:`cwaves.scripts.cwave_listing_compare`.

A tiny synthetic SARWAVE L1B XSP product (intraburst) is built in ``tmp_path`` :
it carries both the cross-spectra (so the *library* can compute CWAVE) and the
historical ``cwave_params`` (so the *old-vs-new* comparison can be exercised).
The full ``entrypoint`` is run over a one-SAFE listing, so the listing reader,
the file detection, the old-parameter reader, the map and the scatter are all
covered.
"""

import numpy as np
import xarray as xr
from cwaves.scripts.cwave_listing_compare import (
    _plot_scatter,
    _read_old_cwave,
    _xsp_files,
    entrypoint,
    read_listing,
)


def _make_xsp_product(path: str) -> None:
    """Write a minimal SARWAVE L1B XSP intraburst product (1x1 tile)."""
    rng = np.random.default_rng(7)
    n_tl, n_ts = 1, 1
    n_fr, n_fs = 11, 40
    k_rg = np.linspace(0.0, 0.5, n_fs)[None, None, :] * np.ones((n_tl, n_ts, 1))
    k_az = np.linspace(-0.05, 0.05, n_fr)
    re = rng.standard_normal((n_tl, n_ts, n_fr, n_fs, 2, 1)).astype(np.float32)
    im = rng.standard_normal((n_tl, n_ts, n_fr, n_fs, 2, 1)).astype(np.float32)
    ds = xr.Dataset(
        {
            "xspectra_2tau_Re": (
                ("tile_line", "tile_sample", "freq_line", "freq_sample", "pol", "2tau"),
                re,
            ),
            "xspectra_2tau_Im": (
                ("tile_line", "tile_sample", "freq_line", "freq_sample", "pol", "2tau"),
                im,
            ),
            "k_rg": (("tile_line", "tile_sample", "freq_sample"), k_rg),
            "k_az": (("freq_line",), k_az),
            "longitude": (("tile_line", "tile_sample"), np.full((n_tl, n_ts), -9.0)),
            "latitude": (("tile_line", "tile_sample"), np.full((n_tl, n_ts), 34.0)),
            "land_flag": (
                ("tile_line", "tile_sample"),
                np.zeros((n_tl, n_ts), np.int8),
            ),
            "pol": (("pol",), ["VH", "VV"]),
            "cwave_params": (
                ("tile_line", "tile_sample", "k_gp", "phi_hf", "pol"),
                np.ones((n_tl, n_ts, 4, 5, 2), np.float32)
                + rng.standard_normal((n_tl, n_ts, 4, 5, 2)).astype(np.float32),
            ),
        },
        coords={"k_gp": [1, 2, 3, 4], "phi_hf": [1, 2, 3, 4, 5]},
    )
    ds.to_netcdf(path, group="intraburst")


def test_read_listing(tmp_path):
    listing = tmp_path / "listing.txt"
    listing.write_text("/a/b.SAFE\n\n# comment\n/c/d.SAFE\n")
    assert read_listing(str(listing)) == ["/a/b.SAFE", "/c/d.SAFE"]


def test_xsp_files(tmp_path):
    safe = tmp_path / "S1B_IW_XSP__1SDV_test.SAFE"
    safe.mkdir()
    (safe / "l1b-s1b-iw1-xsp-1sdv-001.nc").touch()
    (safe / "l1b-s1b-iw2-xsp-1sdv-001.nc").touch()
    (safe / "l1b-s1b-iw3-xsp-1sdv-001.nc").touch()
    (safe / "unrelated.nc").touch()
    files = _xsp_files(str(safe))
    assert set(files) == {"iw1", "iw2", "iw3"}
    assert all(p.name.startswith("l1b-s1b-iw") for p in files.values())


def test_read_old_cwave(tmp_path):
    p = tmp_path / "l1b-s1b-iw2-xsp-1sdv-001.nc"
    _make_xsp_product(str(p))
    lons, lats, params = _read_old_cwave(str(p), "intraburst", "VV")
    assert lons.shape == (1, 1)
    assert lats.shape == (1, 1)
    assert params.shape == (1, 1, 4, 5)
    assert np.isfinite(params).all()


def test_plot_scatter(tmp_path):
    rng = np.random.default_rng(0)
    old = rng.standard_normal(200)
    new = old + 0.5 * rng.standard_normal(200) + 0.2
    out = tmp_path / "scatter.png"
    st = _plot_scatter(old, new, 1, 1, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert st["n"] == 200
    assert st["r"] > 0.8
    assert np.isfinite(st["bias"]) and np.isfinite(st["std"])


def test_entrypoint_full_listing(tmp_path):
    safe = tmp_path / "S1B_IW_XSP__1SDV_test.SAFE"
    safe.mkdir()
    prod = safe / "l1b-s1b-iw1-xsp-1sdv-001.nc"
    _make_xsp_product(str(prod))
    listing = tmp_path / "listing.txt"
    listing.write_text(str(safe) + "\n")
    out_dir = tmp_path / "out"
    rc = entrypoint(
        [
            "--input-listing",
            str(listing),
            "--output-dir",
            str(out_dir),
            "--subswath",
            "iw1",
        ]
    )
    assert rc == 0
    for kg in range(1, 5):
        for ph in range(1, 6):
            assert (out_dir / f"cwave_map_k{kg}_phi{ph}.png").exists()
            assert (out_dir / f"compare_cwave_k{kg}_phi{ph}.png").exists()
    assert (out_dir / "compare_summary.txt").exists()


def test_entrypoint_missing_safe(tmp_path):
    listing = tmp_path / "listing.txt"
    listing.write_text(str(tmp_path / "nope.SAFE") + "\n")
    out_dir = tmp_path / "out"
    rc = entrypoint(
        [
            "--input-listing",
            str(listing),
            "--output-dir",
            str(out_dir),
            "--cwave-nk",
            "1",
            "--cwave-nphi",
            "1",
        ]
    )
    assert rc == 1  # nothing processed
