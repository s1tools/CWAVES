"""Tests for :mod:`cwaves.scripts.cwave_certification_report`.

Tiny synthetic products are built in ``tmp_path`` so the full ``entrypoint``
can be exercised without the real assets : a SARWAVE L1B XSP (one time slice,
smooth spectrum), an Envisat/xsar_slc L1C (singleton tile dims) and an OCN WV
osw product (cartesian ``oswCartSpec`` on a sinh grid, ``m/m`` units). The pure
helpers (basis parsing, normalized density, per-basis sweep, figures, HTML
assembly, schema dispatch) are tested on synthetic cross-spectra.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr
from cwaves.scripts.cwave_certification_report import (
    _build_html,
    _fig_basis_panel,
    _fig_curves,
    _fmt_pct,
    _kernel_domain_mask,
    _l1b_cross_spectrum_compat,
    _load_cross_spectrum,
    _normalized_density,
    _parse_bases,
    _png_bytes,
    _product_kind,
    _sweep_basis,
    _sweep_product,
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


def _make_xsp_product_envisat(path: str) -> None:
    """Write a minimal xsar_slc-style L1C product (singleton tile dims)."""
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
                ("time", "tile_sample", "tile_line", "freq_sample"),
                np.broadcast_to(k_rg_1d, (n_time, 1, 1, n_fs)).astype(np.float32),
            ),
            "k_az": (
                ("time", "tile_sample", "tile_line", "freq_line"),
                np.broadcast_to(k_az_1d, (n_time, 1, 1, n_fr)).astype(np.float32),
            ),
        },
        coords={"pol": ["VV"], "tile_sample": [0], "tile_line": [0]},
    )
    ds.to_netcdf(path)


def _make_ocn_product(path: str) -> None:
    """Write a minimal OCN WV osw product (cartesian sinh grid, ``m/m`` units).

    The ``oswCartSpecRe/Im`` are on a sinh (non-uniform) wavenumber grid in
    normalized (``m/m``) units, decoded to a regular rad/m grid by
    :func:`cwaves.ocn_sinh.compute_ocn_sinh_cartesian`. Single-tile layout
    (no ``oswAzSize`` / ``oswRaSize`` dims) with the calibration inputs the
    decoder needs (``oswKy``, ``oswAziSize``, ``owiIncidenceAngle`` and the
    ``raSamplingFreq`` attribute).
    """
    from cwaves.ocn_sinh import DEFAULT_GROWTH_RATE

    n_kx, n_ky, n_lag = 16, 9, 3
    growth = DEFAULT_GROWTH_RATE
    azi_size = 20433.0
    dky_true = 1.3e-3  # rad/m per unit of the normalized (m/m) azimuth grid
    kappa_az = np.arange(n_ky) - n_ky // 2
    kky = np.sinh(growth * kappa_az) / growth
    osw_ky = (dky_true * azi_size * kky).astype(np.float64)
    osw_kx = (dky_true * azi_size * np.sinh(growth * np.arange(n_kx)) / growth).astype(
        np.float64
    )
    kx_idx = np.arange(n_kx)
    ky_idx = np.arange(n_ky) - n_ky // 2
    ky, kx = np.meshgrid(ky_idx, kx_idx, indexing="ij")  # (n_ky, n_kx)
    gauss = np.exp(-((kx * kx + ky * ky) / 30.0))
    re = np.repeat(gauss[:, :, None], n_lag, axis=2)
    im = np.repeat(0.05 * gauss[:, :, None], n_lag, axis=2)
    ds = xr.Dataset(
        {
            "oswCartSpecRe": (
                ("oswKyBinSize", "oswKxBinSize", "oswLag"),
                re.astype(np.float32),
            ),
            "oswCartSpecIm": (
                ("oswKyBinSize", "oswKxBinSize", "oswLag"),
                im.astype(np.float32),
            ),
            "oswKx": ("oswKxBinSize", osw_kx.astype(np.float32)),
            "oswKy": ("oswKyBinSize", osw_ky.astype(np.float32)),
            "oswAziSize": np.float64(azi_size),
            "owiIncidenceAngle": np.float64(30.0),
        },
        attrs={"raSamplingFreq": np.float64(1.0e8)},
    )
    ds.to_netcdf(path)


def _make_ww3_product(path: str) -> None:
    """Write a minimal WW3 colocation product (polar directional wave
    elevation spectrum E(k, phi)).

    ``polSpec`` is a smooth (noise-free) directional wave elevation spectrum
    on a log-spaced wavenumber grid in 1/m (cycles/m) and a full-circle
    30-degree angular grid, like the IFREMER colocation WW3 products.
    """
    n_k, n_phi = 8, 12
    k_m1 = np.logspace(np.log10(0.005), np.log10(0.1), n_k)  # 1/m
    phi_deg = np.arange(n_phi) * (360.0 / n_phi)  # 0..330 deg
    kk = k_m1[:, None] * 2.0 * np.pi  # rad/m
    phip = np.deg2rad(phi_deg)[None, :]
    # smooth model spectrum: gaussian ridge in k, single azimuthal lobe
    # centered at phi = pi/4 (45 deg, upper-right diagonal)
    pol = np.exp(-(((kk - 0.10) / 0.03) ** 2)) * (0.5 + 0.5 * np.cos(phip - 0.785))
    ds = xr.Dataset(
        {
            "polSpec": (
                ("directions", "wavenumbers", "time"),
                pol.T[:, :, None].astype(np.float32),
            ),
            "k": ("wavenumbers", k_m1.astype(np.float32)),
            "phi": ("directions", phi_deg.astype(np.float32)),
            "time": ("time", [np.float32(3.5e8)]),
        },
    )
    ds.to_netcdf(path)


def _res(nk: int, nphi: int, rel_l2: float) -> dict[str, int | float]:
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
        "n_valid": 1000,
        "n_match": 900,
        "max_abs_diff": 0.5 * rel_l2,
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
    # the point-to-point match metrics are consistent with the difference
    valid = ~np.isnan(diff)
    assert res["n_valid"] == int(valid.sum())
    assert 0 <= res["n_match"] <= res["n_valid"]
    assert res["n_match"] == int(np.sum(np.abs(diff[valid]) <= 1e-3))
    assert res["max_abs_diff"] == pytest.approx(float(np.nanmax(np.abs(diff))))


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


def test_fig_basis_panel_domain_shading():
    # the optional ``domain`` mask shades the out-of-kernel-domain region gray
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    res = _sweep_basis(f, k_rg, k_az, KMIN, KMAX, 4, 5, with_fields=True)
    f_2d = np.asarray(f, dtype=float)
    domain = (
        _kernel_domain_mask(k_rg, k_az, KMIN, KMAX)
        .transpose("freq_line", "freq_sample")
        .values
    )
    panel = _fig_basis_panel(
        f_2d,
        np.asarray(res["fhat_2d"], dtype=float),
        np.asarray(res["diff_2d"], dtype=float),
        np.asarray(f.k_rg, dtype=float),
        np.asarray(f.k_az, dtype=float),
        KMAX,
        res,
        domain=domain,
    )
    assert panel[:8] == b"\x89PNG\r\n\x1a\n"


def test_fig_curves_domain_curve():
    # the rel-L2 panel carries a dashed in-domain curve when the key is present
    res = [
        dict(_res(2, 3, 0.4), rel_l2_domain=0.2),
        dict(_res(4, 5, 0.3), rel_l2_domain=0.15),
    ]
    assert _fig_curves(res)[:8] == b"\x89PNG\r\n\x1a\n"


def test_kernel_domain_mask():
    # the mask is True on the range axis (inside the annulus) and False at
    # 45 deg where the elliptic (gamma=2) domain is azimuthally compressed
    k_rg_1d = np.array([0.05, 0.10, 0.1414, 0.20])
    k_az_1d = np.array([0.0, 0.1414])
    k_rg = xr.DataArray(k_rg_1d, dims="freq_sample")
    k_az = xr.DataArray(k_az_1d, dims="freq_line")
    mask = _kernel_domain_mask(k_rg, k_az, KMIN, KMAX)
    m = np.asarray(mask.transpose("freq_line", "freq_sample"), dtype=bool)
    # (k_az=0.0, k_rg=0.10) : on the range axis, inside the annulus -> in domain
    assert bool(m[0, 1])
    # (k_az=0.1414, k_rg=0.1414) : 45 deg, r = 0.20 -> outside the elliptic domain
    assert not bool(m[1, 2])
    # a far point on the range axis (k_rg=0.20 < kmax) is still in domain
    assert bool(m[0, 3])


# ---------------------------------------------------------------------------
# _build_html (key-finding branches)
# ---------------------------------------------------------------------------


def _prod(
    name: str, results: list[dict[str, int | float]], **kw: Any
) -> dict[str, Any]:
    """Build a minimal product dict for the HTML tests."""
    results = list(results)
    best = min(results, key=lambda r: r["rel_l2"])
    ref = next((r for r in results if (r["nk"], r["nphi"]) == (32, 30)), None)
    defaults: dict[str, Any] = {
        "results": results,
        "kind": "SARWAVE L1B XSP",
        "panels": [],
        "curves_png": b"png",
        "kmax_display": 0.25,
        "grid_shape": (25, 40),
        "n_pixels": 1000,
        "best": best,
        "best_near": best,
        "near": [r for r in results if 780 <= r["n_coeffs"] <= 1200],
        "ref_val": ref["rel_l2"] if ref is not None else 0.1533,
    }
    defaults.update(kw)
    return {"name": name, **defaults}


def test_build_html_beats_reference():
    results = [_res(32, 30, 0.1533), _res(30, 28, 0.145)]
    html = _build_html([_prod("x.nc", results)], 600.0, 25.0)
    assert "does" in html
    assert "percentage points" in html
    # both rows rendered, the reference row tagged
    assert html.count("<tr class=") == 2
    assert "32x30 ref" in html


def test_build_html_plateau():
    results = [_res(32, 30, 0.1533), _res(32, 32, 0.1529)]
    html = _build_html([_prod("x.nc", results)], 600.0, 25.0)
    assert "flat plateau" in html
    assert "No basis close to 32 x 30 beats it" in html


def test_build_html_no_beat():
    results = [_res(32, 30, 0.1533), _res(30, 32, 0.1553)]
    html = _build_html([_prod("x.nc", results)], 600.0, 25.0)
    assert "no basis in that window" in html


def test_build_html_without_reference_row():
    results = [_res(2, 3, 0.4), _res(4, 5, 0.3)]
    html = _build_html([_prod("x.nc", results)], 600.0, 25.0)
    # the reference value falls back to the constant (15.33 %)
    assert "15.33%" in html
    assert "32x30 ref" not in html
    # the best row is tagged
    assert "best" in html


def test_build_html_gallery():
    results = [_res(2, 3, 0.4), _res(4, 5, 0.3)]
    panels = [(results[0], b"aaa"), (results[1], b"bbb")]
    html = _build_html([_prod("x.nc", results, panels=panels)], 600.0, 25.0)
    assert html.count("data:image/png;base64,") == 3  # 2 panels + curves
    # the fourth column carries the per-basis point-to-point metrics
    assert html.count("panel-metrics") == 3  # 2 panels + the CSS rule
    assert "points matching the original (|&Delta;| &le; 1e-03 a.u.)" in html
    assert "highest pixel-to-pixel difference" in html
    # 900 / 1000 valid pixels -> 90.00 % match, both panels
    assert html.count("<strong>900 / 1000</strong> (90.00 %)") == 2


def test_build_html_kernel_domain_breakdown():
    # a product whose sweep carried the in-domain breakdown gets the extra
    # table column, the gallery line and the findings-card paragraph
    results = [
        dict(_res(32, 30, 0.683), rel_l2_domain=0.0578),
        dict(_res(40, 35, 0.682), rel_l2_domain=0.0460),
    ]
    panels = [(results[0], b"aaa"), (results[1], b"bbb")]
    prod = _prod(
        "ww3.nc",
        results,
        panels=panels,
        kind="WW3 model (directional wave elevation spectrum E(k, phi), "
        "resampled to cartesian rad/m)",
        domain_in_mass=0.5352,
        domain_mask=None,
    )
    html = _build_html([prod], 600.0, 25.0)
    # the conditional table column (header + one cell per row)
    assert "rel. L2 (in kernel domain)" in html
    # the findings card carries the kernel-domain breakdown paragraph : the
    # in-domain L2 mass, the hard out-of-domain floor and the best in-domain basis
    assert "Kernel-domain breakdown" in html
    assert "53.5 %</strong> of this spectrum's L2 mass" in html
    assert "68.2 %" in html  # 100 * sqrt(1 - 0.5352)
    assert "4.60 %" in html  # best in-domain rel L2 (40 x 35)
    assert "at <strong>40 x 35</strong>" in html
    # the WW3-specific tail
    assert "out-of-domain mass, not noise" in html
    # the gallery fourth column carries the per-basis in-domain rel L2
    assert "in-kernel-domain rel. L2" in html
    assert "<strong>4.60 %</strong>" in html


def test_build_html_no_domain_breakdown():
    # without the in-domain breakdown the extra column / note are absent
    results = [_res(32, 30, 0.1533), _res(32, 32, 0.1529)]
    prod = _prod("s1a.nc", results, domain_in_mass=0.999, domain_mask=None)
    html = _build_html([prod], 600.0, 25.0)
    assert "rel. L2 (in kernel domain)" not in html
    assert "Kernel-domain breakdown" not in html
    assert "in-kernel-domain rel. L2" not in html


def test_build_html_multi_product():
    p1 = _prod("s1a.nc", [_res(32, 30, 0.1533), _res(32, 32, 0.1529)])
    p2 = _prod("envisat.nc", [_res(32, 30, 0.1533), _res(30, 30, 0.1528)])
    html = _build_html([p1, p2], 600.0, 25.0)
    # one section per product
    assert 'class="product"' in html
    assert "s1a.nc" in html
    assert "envisat.nc" in html
    # the unified finding lists one verdict per product (the side nav also
    # carries <li> items, so count inside the summary list only)
    summary = html.split('class="summary"')[1].split("</ul>")[0]
    assert summary.count("<li>") == 2
    assert "flat plateau" in html
    # the unified closing is present because every product is flat
    assert "not of a single product" in html
    # the footer lists both source products
    assert "<code>s1a.nc</code>, <code>envisat.nc</code>" in html
    # the wavelength window is rendered from wlmax / wlmin
    assert "2&pi;/600" in html
    assert "2&pi;/25" in html
    # one curves figure per product + two tables
    assert html.count("data:image/png;base64,") == 2
    assert html.count("<table>") == 2


def test_build_html_multi_product_not_all_flat():
    p1 = _prod("s1a.nc", [_res(32, 30, 0.1533), _res(32, 32, 0.1529)])
    p2 = _prod("envisat.nc", [_res(32, 30, 0.1533), _res(30, 28, 0.145)])
    html = _build_html([p1, p2], 600.0, 25.0)
    # one product clearly beats the reference
    assert "beats the 15.33% reference by 0.83 pp" in html
    # but no unified closing since not every product is flat
    assert "not of a single product" not in html


def test_build_html_single_product_no_closing():
    p1 = _prod("s1a.nc", [_res(32, 30, 0.1533), _res(30, 28, 0.145)])
    html = _build_html([p1], 600.0, 25.0)
    # no multi-product closing for a single product
    assert "not of a single product" not in html


def test_build_html_nav():
    p1 = _prod("s1a.nc", [_res(32, 30, 0.1533), _res(32, 32, 0.1529)])
    p2 = _prod("envisat.nc", [_res(32, 30, 0.1533), _res(30, 30, 0.1528)])
    html = _build_html([p1, p2], 600.0, 25.0)
    # the side navigation panel is present
    assert "<nav class='sidenav'>" in html
    assert "href='#top'" in html
    assert "href='#interpretation'" in html
    assert 'id="interpretation"' in html
    # one nav entry and one anchor per product section
    assert "href='#product-1'" in html
    assert "href='#product-2'" in html
    assert 'id="product-1"' in html
    assert 'id="product-2"' in html
    assert html.count('id="product-') == 2
    # the nav entries carry the product names
    assert ">1. s1a.nc</a>" in html
    assert ">2. envisat.nc</a>" in html


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
    # the gallery rows carry the fourth-column metrics
    assert html.count("<div class='panel-metrics'>") == 2
    assert "highest pixel-to-pixel difference" in html


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


# ---------------------------------------------------------------------------
# Envisat / xsar_slc L1C schema (singleton tile dims)
# ---------------------------------------------------------------------------


def test_compat_l1c_schema(tmp_path):
    nc = tmp_path / "envisat.nc"
    _make_xsp_product_envisat(str(nc))
    with xr.open_dataset(nc) as ds:
        xs = _l1b_cross_spectrum_compat(ds, lag=2)
    # the tile_sample / tile_line dims are squeezed away
    assert xs.dims == ("freq_line", "freq_sample")
    assert xs["k_rg"].dims == ("freq_sample",)
    assert xs["k_az"].dims == ("freq_line",)
    assert xs.shape == (25, 40)
    assert np.iscomplexobj(xs.values)


def test_compat_l1b_schema(tmp_path):
    # the SARWAVE L1B schema (no tile dims) goes through the same path
    nc = tmp_path / "s1a.nc"
    _make_xsp_product(str(nc))
    with xr.open_dataset(nc) as ds:
        xs = _l1b_cross_spectrum_compat(ds, lag=2)
    assert xs.dims == ("freq_line", "freq_sample")
    assert xs.shape == (25, 40)


def test_entrypoint_multi_product(tmp_path):
    nc1 = tmp_path / "s1a.nc"
    nc2 = tmp_path / "envisat.nc"
    nc3 = tmp_path / "ocn.nc"
    nc4 = tmp_path / "ww3.nc"
    _make_xsp_product(str(nc1))
    _make_xsp_product_envisat(str(nc2))
    _make_ocn_product(str(nc3))
    _make_ww3_product(str(nc4))
    out_dir = tmp_path / "out"
    rc = entrypoint(
        [
            "--xsp-nc",
            str(nc1),
            str(nc2),
            str(nc3),
            str(nc4),
            "--output-dir",
            str(out_dir),
            "--bases",
            "2 3",
        ]
    )
    assert rc == 0
    html = (out_dir / "index.html").read_text()
    # one section per product, all in the summary list and the footer
    assert html.count('class="product"') == 4
    assert "s1a.nc" in html
    assert "envisat.nc" in html
    assert "ocn.nc" in html
    assert "ww3.nc" in html
    # the OCN and WW3 sections carry their schema labels
    assert "cartesian sinh grid, decoded to regular rad/m" in html
    assert (
        "directional wave elevation spectrum E(k, phi), resampled to cartesian "
        "rad/m" in html
    )
    # one panel + one curves figure per product
    assert html.count("data:image/png;base64,") == 8
    # the side nav carries one entry per product
    assert "href='#product-4'" in html
    # the gallery rows carry the fourth-column metrics
    assert html.count("<div class='panel-metrics'>") == 4


def test_entrypoint_ww3_product(tmp_path):
    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    out_dir = tmp_path / "out"
    rc = entrypoint(
        ["--xsp-nc", str(nc), "--output-dir", str(out_dir), "--bases", "2 3"]
    )
    assert rc == 0
    html = (out_dir / "index.html").read_text()
    assert "2 x 3" in html
    assert (
        "directional wave elevation spectrum E(k, phi), resampled to cartesian "
        "rad/m" in html
    )
    # one per-basis panel + one curves figure
    assert html.count("data:image/png;base64,") == 2
    # the synthetic WW3 spectrum carries out-of-domain mass, so the
    # kernel-domain breakdown is active (table column + findings paragraph)
    assert "rel. L2 (in kernel domain)" in html
    assert "Kernel-domain breakdown" in html


# ---------------------------------------------------------------------------
# OCN WV osw schema (cartesian sinh grid, m/m units)
# ---------------------------------------------------------------------------


def test_product_kind(tmp_path):
    nc_l1b = tmp_path / "s1a.nc"
    nc_l1c = tmp_path / "envisat.nc"
    nc_ocn = tmp_path / "ocn.nc"
    nc_ww3 = tmp_path / "ww3.nc"
    _make_xsp_product(str(nc_l1b))
    _make_xsp_product_envisat(str(nc_l1c))
    _make_ocn_product(str(nc_ocn))
    _make_ww3_product(str(nc_ww3))
    with xr.open_dataset(nc_l1b) as ds:
        assert _product_kind(ds) == "SARWAVE L1B XSP"
    with xr.open_dataset(nc_l1c) as ds:
        assert _product_kind(ds) == "ASAR WVI XSP L1C (xsar_slc)"
    with xr.open_dataset(nc_ocn) as ds:
        assert (
            _product_kind(ds)
            == "OCN WV (cartesian sinh grid, decoded to regular rad/m)"
        )
    with xr.open_dataset(nc_ww3) as ds:
        assert (
            _product_kind(ds)
            == "WW3 model (directional wave elevation spectrum E(k, phi), "
            "resampled to cartesian rad/m)"
        )


def test_load_cross_spectrum_ocn(tmp_path):
    nc = tmp_path / "ocn.nc"
    _make_ocn_product(str(nc))
    with xr.open_dataset(nc) as ds:
        xs = _load_cross_spectrum(ds, lag=2)
    # the sinh grid is decoded to a regular rad/m grid, canonical dims
    assert xs.dims == ("freq_line", "freq_sample")
    assert xs.shape == (256, 128)  # 256 azimuth x 128 one-sided range
    assert np.iscomplexobj(xs.values)
    assert bool(np.isfinite(np.abs(xs.values)).all())
    # one-sided range wavenumbers, calibrated to rad/m
    k_rg = np.asarray(xs["k_rg"], dtype=float)
    k_az = np.asarray(xs["k_az"], dtype=float)
    assert float(k_rg.min()) == pytest.approx(0.0, abs=1e-9)
    assert float(k_rg.max()) == pytest.approx(0.2599, rel=1e-3)
    assert float(k_az.min()) < 0.0 < float(k_az.max())
    assert float(np.abs(k_az).max()) == pytest.approx(0.1664, rel=1e-3)


def test_load_cross_spectrum_l1b(tmp_path):
    # the SARWAVE L1B / Envisat path is unchanged by the OCN dispatch
    nc = tmp_path / "s1a.nc"
    _make_xsp_product(str(nc))
    with xr.open_dataset(nc) as ds:
        xs = _load_cross_spectrum(ds, lag=2)
    assert xs.dims == ("freq_line", "freq_sample")
    assert xs.shape == (25, 40)
    assert np.iscomplexobj(xs.values)


def test_load_cross_spectrum_ww3(tmp_path):
    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    with xr.open_dataset(nc) as ds:
        xs = _load_cross_spectrum(ds, lag=2, kmax=KMAX)
    # the polar model spectrum is resampled to a regular cartesian rad/m grid
    assert xs.dims == ("freq_line", "freq_sample")
    assert xs.shape == (256, 256)  # WW3_N_AZIMUTH x WW3_N_RANGE
    assert np.iscomplexobj(xs.values)
    assert bool(np.isfinite(np.abs(xs.values)).all())
    # one-sided range wavenumbers in rad/m, covering [0, kmax]
    k_rg = np.asarray(xs["k_rg"], dtype=float)
    k_az = np.asarray(xs["k_az"], dtype=float)
    assert float(k_rg.min()) == pytest.approx(0.0, abs=1e-12)
    assert float(k_rg.max()) == pytest.approx(KMAX, rel=1e-12)
    assert float(k_az.min()) == pytest.approx(-KMAX, rel=1e-12)
    assert float(k_az.max()) == pytest.approx(KMAX, rel=1e-12)
    vals = np.abs(xs.values)
    # the model energy sits near the gaussian ridge (k = 0.10 rad/m)
    peak_k = float(k_rg[int(np.nanargmax(np.nansum(vals, axis=0)))])
    assert 0.07 < peak_k < 0.14
    # the azimuthal lobe (centered at phi = pi/4, in the k_az > 0 half-plane)
    # makes the two azimuth half-planes asymmetric
    upper = float(vals[k_az > 0, :].sum())
    lower = float(vals[k_az < 0, :].sum())
    assert upper > 1.5 * lower
    # the WW3 branch requires kmax
    with xr.open_dataset(nc) as ds, pytest.raises(ValueError, match="kmax"):
        _load_cross_spectrum(ds, lag=2)


def test_sweep_product_kernel_domain(tmp_path):
    # the WW3 product dict carries the kernel-domain breakdown : a float
    # in-domain L2 mass, a 2-D domain mask and per-basis in-domain rel L2
    from pathlib import Path

    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    prod = _sweep_product(Path(nc), KMIN, KMAX, [(2, 3), (4, 5)], no_panels=True)
    assert isinstance(prod["domain_in_mass"], float)
    # the synthetic WW3 lobe (phi = pi/4) pushes mass outside the domain
    assert prod["domain_in_mass"] < 1.0 - 5e-3
    assert prod["domain_mask"] is not None
    # every sweep result carries the in-domain relative L2
    for res in prod["results"]:
        assert "rel_l2_domain" in res
        assert 0.0 <= float(res["rel_l2_domain"]) < float(res["rel_l2"])


def test_entrypoint_ocn_product(tmp_path):
    nc = tmp_path / "ocn.nc"
    _make_ocn_product(str(nc))
    out_dir = tmp_path / "out"
    rc = entrypoint(
        ["--xsp-nc", str(nc), "--output-dir", str(out_dir), "--bases", "2 3"]
    )
    assert rc == 0
    html = (out_dir / "index.html").read_text()
    assert "2 x 3" in html
    assert "cartesian sinh grid, decoded to regular rad/m" in html
    # one per-basis panel + one curves figure
    assert html.count("data:image/png;base64,") == 2
