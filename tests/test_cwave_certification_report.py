"""Tests for :mod:`cwaves.scripts.cwave_certification_report`.

A tiny synthetic SARWAVE L1B XSP product (one time slice, smooth spectrum) is
built in ``tmp_path`` so the full ``entrypoint`` can be exercised without the
real asset. The pure helpers (basis parsing, normalized density, per-basis
sweep, figures, HTML assembly) are tested on a synthetic cross-spectrum.
"""

import numpy as np
import pytest
import xarray as xr
from cwaves.scripts.cwave_certification_report import (
    _build_html,
    _fig_basis_panel,
    _fig_curves,
    _fmt_pct,
    _normalized_density,
    _parse_bases,
    _png_bytes,
    _sweep_basis,
    entrypoint,
)

KMIN = 2.0 * np.pi / 600.0
KMAX = 2.0 * np.pi / 25.0


def _synthetic_xs() -> xr.DataArray:
    """Build a smooth one-sided cross-spectrum (25 x 40 grid)."""
    n_fr, n_fs = 25, 40
    k_rg_1d = np.linspace(0.0, 0.4, n_fs)
    k_az_1d = np.linspace(-0.15, 0.15, n_fr)
    krg, kaz = np.meshgrid(k_rg_1d, k_az_1d, indexing="xy")
    kk = np.hypot(krg, kaz)
    vals = np.exp(-(((kk - 0.10) / 0.04) ** 2)) + 0.02j
    return xr.DataArray(
        vals,
        dims=("freq_line", "freq_sample"),
        coords={"k_rg": ("freq_sample", k_rg_1d), "k_az": ("freq_line", k_az_1d)},
        name="xspectra_2tau",
    )


def _make_xsp_product(path: str) -> None:
    """Write a minimal SARWAVE L1B XSP product (1 time slice, smooth)."""
    n_time, n_pol, n_lag = 1, 1, 1
    n_fr, n_fs = 25, 40
    k_rg_1d = np.linspace(0.0, 0.4, n_fs)
    k_az_1d = np.linspace(-0.15, 0.15, n_fr)
    krg, kaz = np.meshgrid(k_rg_1d, k_az_1d, indexing="xy")
    kk = np.hypot(krg, kaz)
    re = np.exp(-(((kk - 0.10) / 0.04) ** 2))[None, :, :, None, None]
    im = np.full((n_fr, n_fs), 0.02)[None, :, :, None, None]
    ds = xr.Dataset(
        {
            "xspectra_2tau_Re": (
                ("time", "freq_line", "freq_sample", "pol", "2tau"),
                np.broadcast_to(re, (n_time, n_fr, n_fs, n_pol, n_lag)).astype(
                    np.float32
                ),
            ),
            "xspectra_2tau_Im": (
                ("time", "freq_line", "freq_sample", "pol", "2tau"),
                np.broadcast_to(im, (n_time, n_fr, n_fs, n_pol, n_lag)).astype(
                    np.float32
                ),
            ),
            "k_rg": (
                ("time", "freq_sample"),
                np.broadcast_to(k_rg_1d, (n_time, n_fs)).astype(np.float32),
            ),
            "k_az": (
                ("time", "freq_line"),
                np.broadcast_to(k_az_1d, (n_time, n_fr)).astype(np.float32),
            ),
            "sensing_time": ("time", ["2025-05-01T06:01:01"]),
        },
        coords={"pol": ["VV"]},
    )
    ds.to_netcdf(path)


def _res(nk: int, nphi: int, rel_l2: float) -> dict[str, object]:
    """Build a minimal sweep-result row for the HTML tests."""
    return {
        "nk": nk,
        "nphi": nphi,
        "n_coeffs": nk * nphi,
        "rel_l2": rel_l2,
        "capture": 1.0 - rel_l2**2,
        "max_pix": 0.5 * rel_l2,
        "within_half": 0.99,
        "parseval_gap": 0.01,
    }


# ---------------------------------------------------------------------------
# _parse_bases / _fmt_pct
# ---------------------------------------------------------------------------


def test_parse_bases():
    assert _parse_bases("4 5 8 8") == [(4, 5), (8, 8)]
    assert _parse_bases("  32  30 ") == [(32, 30)]


def test_parse_bases_odd():
    with pytest.raises(ValueError, match="even"):
        _parse_bases("4 5 8")


def test_fmt_pct():
    assert _fmt_pct(0.1533) == "15.33%"
    assert _fmt_pct(0.0) == "0.00%"


# ---------------------------------------------------------------------------
# _normalized_density
# ---------------------------------------------------------------------------


def test_normalized_density_unit_mass():
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    assert f.dims == ("freq_line", "freq_sample")
    # the density is non-negative and masked outside the annulus
    assert float(f.min(skipna=True)) >= 0.0
    assert bool(f.isnull().any())
    # unit mass under the constant-cell-area quadrature
    dkrg = float(xs.k_rg.diff(dim="freq_sample").mean())
    dkaz = float(xs.k_az.diff(dim="freq_line").mean())
    mass = float(f.sum(skipna=True)) * dkrg * dkaz
    assert mass == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# _sweep_basis
# ---------------------------------------------------------------------------


def test_sweep_basis_metrics():
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    # the kernel is evaluated on the (masked / dropna'd) density grid
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    res = _sweep_basis(f, k_rg, k_az, KMIN, KMAX, 4, 5, with_fields=True)
    assert res["n_coeffs"] == 20
    assert 0.0 < res["rel_l2"] < 1.0
    assert res["capture"] == pytest.approx(1.0 - res["rel_l2"] ** 2)
    assert 0.0 <= res["within_half"] <= 1.0
    # the discrete Parseval identity holds up to quadrature on a smooth field
    assert res["parseval_gap"] < 0.1
    # the 2-D fields are consistent with f - fhat
    fhat = np.asarray(res["fhat_2d"], dtype=float)
    diff = np.asarray(res["diff_2d"], dtype=float)
    assert fhat.shape == f.shape
    fv = np.asarray(f, dtype=float)
    assert np.allclose(diff, fv - fhat, equal_nan=True)


def test_sweep_basis_no_fields():
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    res = _sweep_basis(f, k_rg, k_az, KMIN, KMAX, 2, 3, with_fields=False)
    assert "fhat_2d" not in res
    assert "diff_2d" not in res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def test_png_bytes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    raw = _png_bytes(fig, dpi=50)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(raw) > 1000


def test_fig_basis_panel_and_curves():
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    res = _sweep_basis(f, k_rg, k_az, KMIN, KMAX, 4, 5, with_fields=True)
    f_2d = np.asarray(f, dtype=float)
    panel = _fig_basis_panel(
        f_2d,
        np.asarray(res["fhat_2d"], dtype=float),
        np.asarray(res["diff_2d"], dtype=float),
        np.asarray(f.k_rg, dtype=float),
        np.asarray(f.k_az, dtype=float),
        KMAX,
        res,
    )
    assert panel[:8] == b"\x89PNG\r\n\x1a\n"
    curves = _fig_curves([_res(2, 3, 0.4), _res(4, 5, 0.3)])
    assert curves[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# _build_html (key-finding branches)
# ---------------------------------------------------------------------------


def test_build_html_beats_reference():
    results = [_res(32, 30, 0.1533), _res(30, 28, 0.145)]
    html = _build_html(results, b"png", [], 0.25, (25, 40), 1000, "x.nc")
    assert "does" in html
    assert "percentage points" in html
    # both rows rendered, the reference row tagged
    assert html.count("<tr class=") == 2
    assert "32x30 ref" in html


def test_build_html_plateau():
    results = [_res(32, 30, 0.1533), _res(32, 32, 0.1529)]
    html = _build_html(results, b"png", [], 0.25, (25, 40), 1000, "x.nc")
    assert "flat plateau" in html
    assert "No basis close to 32 x 30 beats it" in html


def test_build_html_no_beat():
    results = [_res(32, 30, 0.1533), _res(30, 32, 0.1553)]
    html = _build_html(results, b"png", [], 0.25, (25, 40), 1000, "x.nc")
    assert "no basis in that window" in html


def test_build_html_without_reference_row():
    results = [_res(2, 3, 0.4), _res(4, 5, 0.3)]
    html = _build_html(results, b"png", [], 0.25, (25, 40), 1000, "x.nc")
    # the reference value falls back to the constant (15.33 %)
    assert "15.33%" in html
    assert "32x30 ref" not in html
    # the best row is tagged
    assert "best" in html


def test_build_html_gallery():
    results = [_res(2, 3, 0.4), _res(4, 5, 0.3)]
    panels = [(results[0], b"aaa"), (results[1], b"bbb")]
    html = _build_html(results, b"png", panels, 0.25, (25, 40), 1000, "x.nc")
    assert html.count("data:image/png;base64,") == 3  # 2 panels + curves


# ---------------------------------------------------------------------------
# entrypoint (end to end, synthetic product)
# ---------------------------------------------------------------------------


def test_entrypoint_end_to_end(tmp_path):
    nc = tmp_path / "xsp.nc"
    _make_xsp_product(str(nc))
    out_dir = tmp_path / "out"
    rc = entrypoint(
        ["--xsp-nc", str(nc), "--output-dir", str(out_dir), "--bases", "2 3 4 5"]
    )
    assert rc == 0
    html = (out_dir / "index.html").read_text()
    assert "2 x 3" in html
    assert "4 x 5" in html
    # two per-basis panels + one curves figure
    assert html.count("data:image/png;base64,") == 3


def test_entrypoint_no_panels(tmp_path):
    nc = tmp_path / "xsp.nc"
    _make_xsp_product(str(nc))
    out_dir = tmp_path / "out"
    rc = entrypoint(
        [
            "--xsp-nc",
            str(nc),
            "--output-dir",
            str(out_dir),
            "--bases",
            "2 3",
            "--no-panels",
        ]
    )
    assert rc == 0
    html = (out_dir / "index.html").read_text()
    # only the curves figure is embedded
    assert html.count("data:image/png;base64,") == 1


def test_entrypoint_default_output(tmp_path, monkeypatch):
    nc = tmp_path / "xsp.nc"
    _make_xsp_product(str(nc))
    monkeypatch.chdir(tmp_path)
    rc = entrypoint(["--xsp-nc", str(nc), "--no-panels"])
    assert rc == 0
    produced = tmp_path / "cwave_certification_report" / "index.html"
    assert produced.exists()
    assert produced.stat().st_size > 0


def test_entrypoint_missing_nc(tmp_path):
    rc = entrypoint(
        [
            "--xsp-nc",
            str(tmp_path / "nope.nc"),
            "--output-dir",
            str(tmp_path / "out"),
            "--bases",
            "2 3",
        ]
    )
    assert rc == 1


def test_entrypoint_odd_bases(tmp_path):
    nc = tmp_path / "xsp.nc"
    _make_xsp_product(str(nc))
    rc = entrypoint(
        ["--xsp-nc", str(nc), "--output-dir", str(tmp_path / "out"), "--bases", "2 3 4"]
    )
    assert rc == 1
