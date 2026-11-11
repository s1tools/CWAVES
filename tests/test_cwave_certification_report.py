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
from cwaves.scripts import cwave_certification_report as ccr
from cwaves.scripts.cwave_certification_report import (
    GRAM_BASIS,
    SCALING_RATIO_MAX,
    _build_html,
    _fig_basis_panel,
    _fig_curves,
    _fmt_pct,
    _gram_diagnostic,
    _idempotence_scaling,
    _kernel_domain_mask,
    _l1b_cross_spectrum_compat,
    _load_cross_spectrum,
    _metrics_section,
    _native_eta,
    _native_f1d,
    _native_g1d,
    _native_grid_sweep,
    _native_inverse,
    _normalized_density,
    _parse_bases,
    _png_bytes,
    _product_kind,
    _refinement_demo,
    _sweep_basis,
    _sweep_product,
    _ww3_polar_interpolator,
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


def _res_idemp(
    nk: int, nphi: int, rel_l2: float, idemp_abs: float, idemp_rel: float
) -> dict[str, int | float]:
    """Build a sweep-result row carrying the Part 2 idempotence metrics."""
    return dict(
        _res(nk, nphi, rel_l2), idemp_max_abs=idemp_abs, idemp_max_rel=idemp_rel
    )


def _idemp_scale(
    e1: float, e2: float, *, truncated: bool = False
) -> dict[str, float | bool | str]:
    """Build the Part 2 per-product scaling diagnostic, with the same
    status derivation as :func:`_idempotence_scaling` (pass / expected /
    fail)."""
    ratio = e2 / e1 if e1 > 0.0 else 0.0
    floor = max(0.0, 2.0 * e2 - e1)
    floor_frac = floor / e1 if e1 > 0.0 else 0.0
    if ratio <= SCALING_RATIO_MAX:
        status = "pass"
    elif truncated:
        status = "expected"
    else:
        status = "fail"
    return {
        "e1": e1,
        "e2": e2,
        "ratio": ratio,
        "floor": floor,
        "floor_frac": floor_frac,
        "truncated": truncated,
        "status": status,
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


def test_sweep_basis_idempotence_metrics():
    # Part 2 : the coefficient round-trip C -> fhat -> C' is measured on
    # every basis of the sweep
    xs = _synthetic_xs()
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    res = _sweep_basis(f, k_rg, k_az, KMIN, KMAX, 4, 5, with_fields=False)
    assert np.isfinite(res["idemp_max_abs"])
    assert np.isfinite(res["idemp_max_rel"])
    assert res["idemp_max_abs"] >= 0.0
    assert res["idemp_max_rel"] >= 0.0
    # the coarse 25 x 40 test grid is far from the continuum limit : the
    # first-order quadrature error is well above machine precision
    assert res["idemp_max_abs"] > 1e-3


def test_refinement_demo_converges():
    # Part 2 : on a full-domain grid the idempotence error halves with every
    # grid doubling (first-order quadrature, not a systematic bug) and the
    # discrete Gram matrix of the 4 x 5 basis converges to the identity
    rows = _refinement_demo(
        KMIN, KMAX, plans=[(1, 128, 64, [(4, 5)]), (2, 256, 128, [(4, 5)])]
    )
    assert len(rows) == 2
    err1, err2 = (r["idemp_max_abs"] for r in rows)
    assert err2 < 0.75 * err1
    # the scaling ratio is reported per doubling (None on the coarsest grid)
    assert rows[0]["ratio"] is None
    assert rows[1]["ratio"] == pytest.approx(err2 / err1)
    assert rows[1]["ratio"] <= SCALING_RATIO_MAX
    # the Gram deviations (on the GRAM_BASIS rows) shrink with refinement
    assert rows[0]["gram_diag"] < 0.25
    assert rows[0]["gram_off"] < 0.25
    assert rows[1]["gram_diag"] < rows[0]["gram_diag"]
    assert rows[1]["gram_off"] < rows[0]["gram_off"]
    assert rows[1]["gram_diag"] < 0.1
    assert rows[1]["gram_off"] < 0.1
    # the non-GRAM_BASIS rows carry no Gram keys
    rows32 = _refinement_demo(KMIN, KMAX, plans=[(1, 128, 64, [(32, 30)])])
    assert "gram_diag" not in rows32[0]
    assert "gram_off" not in rows32[0]
    assert rows32[0]["ratio"] is None


def test_idempotence_scaling_full_domain():
    # Part 2 : on a grid whose range contains the kernel domain the
    # idempotence error halves per grid doubling (status "pass", ratio
    # <= SCALING_RATIO_MAX) and no truncation floor is implied
    n_fr, n_fs = 128, 64
    k_rg = xr.DataArray(np.linspace(0.0, KMAX, n_fs), dims="freq_sample")
    k_az = xr.DataArray(np.linspace(-KMAX, KMAX, n_fr), dims="freq_line")
    krg, kaz = np.meshgrid(np.asarray(k_rg), np.asarray(k_az), indexing="xy")
    vals = np.exp(-(((np.hypot(krg, kaz) - 0.10) / 0.04) ** 2))
    xs = xr.DataArray(
        vals,
        dims=("freq_line", "freq_sample"),
        coords={
            "k_rg": ("freq_sample", np.asarray(k_rg)),
            "k_az": ("freq_line", np.asarray(k_az)),
        },
        name="xspectra_2tau",
    )
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    domain = _kernel_domain_mask(k_rg, k_az, KMIN, KMAX)
    scale = _idempotence_scaling(f, k_rg, k_az, KMIN, KMAX, domain)
    assert scale["ratio"] == pytest.approx(scale["e2"] / scale["e1"])
    assert scale["ratio"] <= SCALING_RATIO_MAX
    assert scale["status"] == "pass"
    assert scale["truncated"] is False
    assert 0.0 <= scale["floor_frac"] < 0.5


def test_idempotence_scaling_truncated():
    # Part 2 : a grid whose range cuts the kernel domain (k_rg max < kmax)
    # carries a range-truncation floor : the error plateaus (ratio >
    # SCALING_RATIO_MAX) and the status is "expected", not "fail"
    n_fr, n_fs = 128, 64
    k_rg = xr.DataArray(np.linspace(0.0, 0.5 * KMAX, n_fs), dims="freq_sample")
    k_az = xr.DataArray(np.linspace(-KMAX, KMAX, n_fr), dims="freq_line")
    krg, kaz = np.meshgrid(np.asarray(k_rg), np.asarray(k_az), indexing="xy")
    vals = np.exp(-(((np.hypot(krg, kaz) - 0.10) / 0.04) ** 2))
    xs = xr.DataArray(
        vals,
        dims=("freq_line", "freq_sample"),
        coords={
            "k_rg": ("freq_sample", np.asarray(k_rg)),
            "k_az": ("freq_line", np.asarray(k_az)),
        },
        name="xspectra_2tau",
    )
    f = _normalized_density(xs, KMIN, KMAX)
    k_rg = xr.DataArray(np.asarray(f.k_rg, dtype=float), dims="freq_sample")
    k_az = xr.DataArray(np.asarray(f.k_az, dtype=float), dims="freq_line")
    domain = _kernel_domain_mask(k_rg, k_az, KMIN, KMAX)
    scale = _idempotence_scaling(f, k_rg, k_az, KMIN, KMAX, domain)
    assert scale["truncated"] is True
    assert scale["status"] in {"expected", "pass"}
    if scale["status"] == "expected":
        assert scale["ratio"] > SCALING_RATIO_MAX
        assert scale["floor"] > 0.0


def test_gram_diagnostic_identity_like():
    # Part 2 : the 4 x 5 basis is nearly orthonormal on a fine full-domain
    # grid (M close to the identity) and the diagnostic is NaN-free (the
    # out-of-domain kernel values are zero-filled)
    n_fr, n_fs = 256, 128
    k_rg = xr.DataArray(np.linspace(0.0, KMAX, n_fs), dims="freq_sample")
    k_az = xr.DataArray(np.linspace(-KMAX, KMAX, n_fr), dims="freq_line")
    diag, off = _gram_diagnostic(k_rg, k_az, KMIN, KMAX, *GRAM_BASIS)
    assert np.isfinite(diag)
    assert np.isfinite(off)
    assert diag < 0.1
    assert off < 0.1


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
# _metrics_section
# ---------------------------------------------------------------------------


def test_metrics_section_present_and_self_contained():
    html = _metrics_section()
    # the section has the anchor used by the side nav
    assert "id='metrics'" in html
    assert "Metrics &amp; definitions" in html
    # the report is a self-contained page : no raw LaTeX dollar signs
    assert "$" not in html
    # both metric families are defined, with their equations as entities
    assert "Part 1" in html
    assert "Part 2" in html
    assert "discrete Gram matrix" in html
    # every Part 1 table metric is named
    for label in (
        "rel. L2 error",
        "captured L2 energy",
        "worst pixel / peak",
        "pixels within 50 % peak",
        "Parseval gap",
        "rel. L2 (in kernel domain)",
        "out-of-domain floor",
    ):
        assert label in html


def test_build_html_metrics_section():
    # the metrics section is wired into the assembled report (before Part 1)
    # and its nav entry is present
    results = [_res(32, 30, 0.1533), _res(32, 32, 0.1529)]
    html = _build_html([_prod("x.nc", results)], 600.0, 25.0)
    assert "id='metrics'" in html
    assert "Metrics &amp; definitions" in html
    assert "href='#metrics'" in html
    # the section sits between the top-level finding and Part 1
    assert html.find("id='metrics'") < html.find('id="part-1"')
    assert "$" not in html


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
    # the per-product table column header is absent (the "Metrics &
    # definitions" section names the metric in a <td> row, not a <th>)
    assert "<th>rel. L2 (in kernel domain)</th>" not in html
    assert "Kernel-domain breakdown" not in html
    assert "in-kernel-domain rel. L2" not in html


def test_build_html_part2_fail():
    # Part 2 : a product whose idempotence error does not halve per doubling
    # on a range-complete grid (status "fail") renders the Part 2 section,
    # the FAIL verdict and the per-product scaling table
    results = [
        _res_idemp(32, 30, 0.1533, 0.46, 0.05),
        _res_idemp(40, 35, 0.1500, 0.70, 0.075),
    ]
    prod = _prod(
        "s1a.nc",
        results,
        idemp_worst=results[1],
        idemp_scale=_idemp_scale(0.46, 0.458, truncated=False),
        gram_grid=(512, 256),
        gram_diag=3.3e-2,
        gram_off=6.0e-2,
    )
    html = _build_html([prod], 600.0, 25.0)
    assert 'id="part-2"' in html
    assert "Part 2 &mdash; Coefficient idempotence" in html
    # the gate verdict is a FAIL (defect signature on a range-complete grid)
    assert "<strong>FAIL.</strong>" in html
    assert "s1a.nc" in html
    # the per-product scaling diagnostic table carries the FAIL tag, the
    # ratio and the implied floor
    assert "Per-product scaling diagnostic" in html
    assert "4.600e-01" in html
    assert "4.580e-01" in html
    assert "0.996" in html  # ratio 0.458 / 0.46
    assert "4.560e-01" in html  # implied floor 2 * 0.458 - 0.46
    assert html.count("<span class='tag fail'>FAIL</span>") == 1
    # the Gram diagnostic table is rendered
    assert "Discrete Gram matrix M of the 4 x 5 basis" in html
    assert "3.300e-02" in html
    assert "6.000e-02" in html
    # the nav carries the Part 1 and Part 2 entries
    assert "href='#part-1'" in html
    assert 'id="part-1"' in html
    assert "href='#part-2'" in html


def test_build_html_part2_fail_refinement():
    # Part 2 : a scaling ratio above the gate on the full-domain refinement
    # grids (plateau instead of halving) fails the global gate
    results = [_res_idemp(4, 5, 0.30, 3.5e-2, 1.0e-2)]
    prod = _prod(
        "s1a.nc",
        results,
        idemp_worst=results[0],
        idemp_scale=_idemp_scale(3.5e-2, 1.7e-2, truncated=False),
    )
    refinement = [
        {
            "scale": 1,
            "grid": (256, 128),
            "nk": 4,
            "nphi": 5,
            "idemp_max_abs": 3.521e-2,
            "idemp_max_rel": 1.022e-2,
            "ratio": None,
            "gram_diag": 4.887e-2,
            "gram_off": 5.626e-2,
        },
        {
            "scale": 2,
            "grid": (512, 256),
            "nk": 4,
            "nphi": 5,
            "idemp_max_abs": 3.49e-2,
            "idemp_max_rel": 1.012e-2,
            "ratio": 0.991,
            "gram_diag": 4.8e-2,
            "gram_off": 5.6e-2,
        },
    ]
    html = _build_html([prod], 600.0, 25.0, refinement=refinement)
    assert "<strong>FAIL.</strong>" in html
    # the plateau is called out on the refinement gate
    assert "does not halve per grid doubling" in html
    assert "0.991" in html
    # the ratio cell above the gate is tagged fail
    assert html.count("<span class='tag fail'>0.991</span>") == 1


def test_build_html_part2_expected():
    # Part 2 : a product whose range cuts the kernel domain (status
    # "expected") is flagged, not failed : the verdict stays a PASS and the
    # scaling table carries the amber EXPECTED tag with the implied floor
    results = [
        _res_idemp(4, 5, 0.30, 7.3e-1, 7.0e-2),
        _res_idemp(32, 30, 0.1533, 8.2e-1, 8.0e-2),
    ]
    prod = _prod(
        "xsp.nc",
        results,
        idemp_worst=results[1],
        idemp_scale=_idemp_scale(7.263e-1, 7.117e-1, truncated=True),
    )
    html = _build_html([prod], 600.0, 25.0)
    assert "<strong>PASS.</strong>" in html
    assert "<span class='tag fail'>FAIL</span>" not in html
    assert "EXPECTED (range floor)" in html
    # the implied floor 2*E2 - E1 and the truncation flag are reported
    assert "6.971e-01" in html  # 2 * 7.117e-1 - 7.263e-1
    assert "yes" in html


def test_build_html_part2_pass():
    # Part 2 : a range-complete product that halves per doubling passes the
    # per-product scaling test and the verdict is a PASS
    results = [
        _res_idemp(4, 5, 0.30, 2.2e-1, 5.0e-2),
        _res_idemp(32, 30, 0.1533, 5e-2, 1e-2),
    ]
    prod = _prod(
        "s1a.nc",
        results,
        idemp_worst=results[0],
        idemp_scale=_idemp_scale(2.164e-1, 1.032e-1, truncated=False),
        gram_grid=(512, 256),
        gram_diag=1e-3,
        gram_off=1e-3,
    )
    html = _build_html([prod], 600.0, 25.0)
    assert "<strong>PASS.</strong>" in html
    assert "PASS (E &prop; h)" in html
    assert "<span class='tag fail'>FAIL</span>" not in html
    # the ratio and the (near-zero) floor share are reported
    assert "0.477" in html
    assert "Discrete Gram matrix M of the 4 x 5 basis" in html


def test_build_html_part2_refinement_table():
    # Part 2 : the refinement demo rows are rendered (with the scaling ratio
    # column and the Gram columns on the GRAM_BASIS rows)
    results = [_res_idemp(4, 5, 0.30, 3.5e-2, 1.0e-2)]
    prod = _prod(
        "s1a.nc",
        results,
        idemp_worst=results[0],
        idemp_scale=_idemp_scale(3.5e-2, 1.7e-2, truncated=False),
        gram_grid=(512, 256),
        gram_diag=3.3e-2,
        gram_off=6.0e-2,
    )
    refinement = [
        {
            "scale": 1,
            "grid": (256, 128),
            "nk": 4,
            "nphi": 5,
            "idemp_max_abs": 3.521e-2,
            "idemp_max_rel": 1.022e-2,
            "ratio": None,
            "gram_diag": 4.887e-2,
            "gram_off": 5.626e-2,
        },
        {
            "scale": 2,
            "grid": (512, 256),
            "nk": 4,
            "nphi": 5,
            "idemp_max_abs": 1.746e-2,
            "idemp_max_rel": 5.069e-3,
            "ratio": 0.4959,
            "gram_diag": 2.297e-2,
            "gram_off": 2.076e-2,
        },
    ]
    html = _build_html([prod], 600.0, 25.0, refinement=refinement)
    assert "Grid-refinement evidence" in html
    assert "3.521e-02" in html
    assert "1.746e-02" in html
    assert "4.887e-02" in html
    assert "2.297e-02" in html
    # the ratio column is rendered (dash on the coarsest grid, value below
    # the gate tagged pass)
    assert "ratio per doubling" in html
    assert html.count("<span class='tag pass'>0.496</span>") == 1
    # the 4x5 basis row count (the GRAM_BASIS rows carry the Gram cells)
    assert html.count("512 x 256 (2x)") >= 1


def test_build_html_no_part2_without_idempotence_metrics():
    # backward compatibility : a product sweep without the idempotence keys
    # (e.g. built before Part 2) renders no Part 2 section at all
    results = [_res(32, 30, 0.1533)]
    prod = _prod("s1a.nc", results)
    html = _build_html([prod], 600.0, 25.0)
    assert 'id="part-2"' not in html
    assert "Coefficient idempotence" not in html
    assert "href='#part-2'" not in html


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
    # one curves figure per product + two per-product tables (plus the two
    # static "Metrics & definitions" tables in the introduction)
    assert html.count("data:image/png;base64,") == 2
    assert html.count("<table>") == 4


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
    # Part 2 is rendered : the error-proportionality gate, the verdict card,
    # the per-product scaling diagnostic and the full-domain refinement gate
    assert 'id="part-2"' in html
    assert "Coefficient idempotence" in html
    assert "error proportionality" in html
    assert "Per-product scaling diagnostic" in html
    assert "Grid-refinement evidence" in html
    assert "Discrete Gram matrix M" in html
    # the per-product scaling diagnostic carries one row per product with a
    # status tag (pass / expected / fail)
    assert "ratio E(2x)/E(1x)" in html


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
    # Part 2 : every sweep result carries the idempotence metrics and the
    # product dict carries the aggregation, the scaling diagnostic and the
    # Gram diagnostic
    for res in prod["results"]:
        assert "idemp_max_abs" in res
        assert "idemp_max_rel" in res
        assert res["idemp_max_abs"] >= 0.0
    assert prod["idemp_worst"] in prod["results"]
    scale = prod["idemp_scale"]
    assert scale["ratio"] == pytest.approx(scale["e2"] / scale["e1"])
    assert scale["e1"] >= 0.0
    assert scale["e2"] >= 0.0
    assert scale["floor"] >= 0.0
    assert scale["floor_frac"] >= 0.0
    assert scale["status"] in {"pass", "expected", "fail"}
    assert isinstance(scale["truncated"], bool)
    # the Gram grid is 2x the (masked / dropna'd) density grid of the sweep
    with xr.open_dataset(nc) as ds:
        xs = _load_cross_spectrum(ds, lag=2, kmax=KMAX)
    f = _normalized_density(xs, KMIN, KMAX)
    assert prod["gram_grid"] == (
        2 * int(f.sizes["freq_line"]),
        2 * int(f.sizes["freq_sample"]),
    )
    assert np.isfinite(prod["gram_diag"])
    assert np.isfinite(prod["gram_off"])


# ---------------------------------------------------------------------------
# native (log-polar) grid round-trip
# ---------------------------------------------------------------------------


def test_native_inverse_roundtrip():
    # the inverse map lands inside the kernel domain and round-trips the
    # (alpha_k, alpha_phi) coordinates
    rng = np.random.default_rng(0)
    ak = rng.uniform(-0.99, 0.99, 500)
    # alpha_phi = arctan2(k_rg, k_az) spans (0, pi) on the one-sided domain
    ap = rng.uniform(1e-3, np.pi - 1e-3, 500)
    krg, kaz = _native_inverse(ak, ap, KMIN, KMAX)
    assert (krg >= 0.0).all()
    gamma = 2.0
    a1 = (gamma**2 - gamma**4) / (gamma**2 * KMIN**2 - KMAX**2)
    a2 = (KMAX**2 - gamma**4 * KMIN**2) / (KMAX**2 - gamma**2 * KMIN**2)
    tmp = a1 * kaz**4 + a2 * kaz**2 + krg**2
    ak_back = (
        2
        * (np.log10(np.sqrt(tmp)) - np.log10(KMIN))
        / (np.log10(KMAX) - np.log10(KMIN))
        - 1
    )
    ap_back = np.arctan2(krg, kaz)
    assert float(np.max(np.abs(ak_back - ak))) < 1e-9
    assert float(np.max(np.abs(ap_back - ap))) < 1e-9


def test_native_kernel_factors_match_library():
    # the factorized kernel G F eta on the native nodes equals the library
    # compute_kernel evaluated on the same (k_rg, k_az) nodes
    from cwaves.compute_cwaves import compute_kernel

    mr = mp = 40
    ak = -1.0 + (np.arange(mr) + 0.5) * (2.0 / mr)
    # the native angular grid is the midpoint nodes of (0, pi)
    ap = (np.arange(mp) + 0.5) * (np.pi / mp)
    ak_2d, ap_2d = np.meshgrid(ak, ap, indexing="ij")
    krg, kaz = _native_inverse(ak_2d, ap_2d, KMIN, KMAX)
    nk, nphi = 6, 5
    g1d = _native_g1d(ak, nk)  # (nk, mr)
    f1d = _native_f1d(ap, nphi)  # (nphi, mp)
    eta2d = _native_eta(krg, kaz, KMIN, KMAX)
    lib = compute_kernel(
        xr.DataArray(krg, dims=("r", "p")),
        xr.DataArray(kaz, dims=("r", "p")),
        KMIN,
        KMAX,
        cwave_Nk=nk,
        cwave_Nphi=nphi,
    )["cwave_kernel"]  # dims (k_gp, r, p, phi_hf)
    our = g1d[:, None, :, None] * f1d[None, :, None, :] * eta2d[None, None, :, :]
    theirs = np.transpose(lib.values, (0, 3, 1, 2))  # (nk, nphi, r, p)
    assert np.allclose(our, theirs, rtol=1e-9, atol=1e-12)


def test_native_grid_sweep_synthetic(tmp_path, monkeypatch):
    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    with xr.open_dataset(nc) as ds:
        interp = _ww3_polar_interpolator(ds)[2]
    # small grid + a tiny Gram basis to keep the test fast / light
    monkeypatch.setattr(ccr, "NATIVE_GRAM_BASES", [(4, 5)])
    nat = _native_grid_sweep(
        interp, KMIN, KMAX, [(2, 3), (4, 5), (8, 8)], n_range=128, n_azimuth=64
    )
    for r in nat["results"]:
        assert 0.0 <= r["capture"] <= 1.0 + 1e-9
        assert r["idemp_max_abs"] >= 0.0
    # the smooth synthetic spectrum : rel L2 falls as the basis grows
    rel = [r["rel_l2"] for r in sorted(nat["results"], key=lambda r: r["n_coeffs"])]
    assert rel[-1] < rel[0]
    # the eta^2 J self-check is tiny (the library identity)
    assert nat["eta_j_check"] < 1e-5
    # the Gram diagnostics are present for the (patched) reference bases
    assert set(nat["gram"].keys()) == {(4, 5)}
    assert nat["gram"][(4, 5)]["max_diag"] >= 0.0


def test_sweep_product_native_ww3(tmp_path):
    # the WW3 product dict carries the native sweep ; a SAR product does not
    from pathlib import Path

    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    # an explicit (non-default) base list is honoured as given (no extension)
    prod = _sweep_product(Path(nc), KMIN, KMAX, [(2, 3)], no_panels=True)
    assert prod["native"] is not None
    assert prod["native"]["n_range"] == ccr.NATIVE_N_RANGE
    assert len(prod["results"]) == 1
    # the native sweep runs on the same (given) base list
    assert len(prod["native"]["results"]) == 1
    # the SAR L1B XSP product carries no native sweep
    nc2 = tmp_path / "l1b.nc"
    _make_xsp_product(str(nc2))
    prod2 = _sweep_product(Path(nc2), KMIN, KMAX, [(2, 3)], no_panels=True)
    assert prod2["native"] is None


def test_sweep_product_ww3_default_bases_extended(tmp_path, monkeypatch):
    # the *default* WW3 sweep is extended with NATIVE_EXTRA_BASES (an explicit
    # user sweep is not); the heavy per-basis work is stubbed for speed
    from pathlib import Path

    def _sweep_basis_stub(f, k_rg, k_az, kmin, kmax, nk, nphi, with_fields):
        return {
            "nk": nk,
            "nphi": nphi,
            "n_coeffs": nk * nphi,
            "rel_l2": 0.1,
            "capture": 0.99,
            "max_pix": 0.01,
            "within_half": 0.5,
            "parseval_gap": 1e-6,
            "idemp_max_abs": 1e-3,
            "idemp_max_rel": 1e-4,
        }

    monkeypatch.setattr(ccr, "_sweep_basis", _sweep_basis_stub)
    monkeypatch.setattr(
        ccr,
        "_idempotence_scaling",
        lambda *a, **kw: {
            "e1": 1e-3,
            "e2": 5e-4,
            "ratio": 0.5,
            "floor": 0.0,
            "floor_frac": 0.0,
            "truncated": False,
            "status": "PASS",
        },
    )
    monkeypatch.setattr(ccr, "_gram_diagnostic", lambda *a, **kw: (1e-2, 1e-2))
    monkeypatch.setattr(ccr, "_fig_curves", lambda *a, **kw: b"png")
    monkeypatch.setattr(
        ccr,
        "_native_grid_sweep",
        lambda *a, **kw: {
            "results": [],
            "gram": {},
            "eta_j_check": 1e-9,
            "n_range": 1,
            "n_azimuth": 1,
        },
    )

    nc = tmp_path / "ww3.nc"
    _make_ww3_product(str(nc))
    expected = len(dict.fromkeys(list(ccr.DEFAULT_BASES) + ccr.NATIVE_EXTRA_BASES))
    prod = _sweep_product(Path(nc), KMIN, KMAX, list(ccr.DEFAULT_BASES), no_panels=True)
    assert len(prod["results"]) == expected
    assert expected > len(ccr.DEFAULT_BASES)  # the extension did happen
    # an explicit (non-default) WW3 sweep is not extended
    prod_small = _sweep_product(Path(nc), KMIN, KMAX, [(4, 5)], no_panels=True)
    assert len(prod_small["results"]) == 1


def test_build_html_native_note():
    # a product carrying a native sweep gets the note + table ; one without
    # does not
    results = [
        dict(_res(32, 30, 0.683), rel_l2_domain=0.0578),
        dict(_res(40, 35, 0.683), rel_l2_domain=0.0457),
    ]
    native = {
        "results": [
            {
                "nk": 32,
                "nphi": 30,
                "n_coeffs": 960,
                "rel_l2": 0.0562,
                "capture": 0.9968,
                "idemp_max_abs": 3.2e-3,
                "idemp_max_rel": 1e-4,
            },
            {
                "nk": 40,
                "nphi": 35,
                "n_coeffs": 1400,
                "rel_l2": 0.0437,
                "capture": 0.9981,
                "idemp_max_abs": 4.4e-3,
                "idemp_max_rel": 1e-4,
            },
        ],
        "gram": {(40, 35): {"max_diag": 1.75e-2, "max_off": 1.75e-2}},
        "eta_j_check": 4.4e-9,
        "n_range": 512,
        "n_azimuth": 256,
    }
    html = _build_html([_prod("ww3.nc", results, native=native)], 600.0, 25.0)
    assert "Native (log-polar) grid check" in html
    assert "rel. L2 (native grid)" in html  # metrics-section row + table header
    assert "idempotence max |C &minus; C&#8242;|" in html  # native table header
    # the cartesian in-domain column is filled for the matching bases
    assert "5.78%" in html and "4.57%" in html
    # a product without a native sweep has no native note / table
    html2 = _build_html([_prod("l1b.nc", results)], 600.0, 25.0)
    assert "Native (log-polar) grid check" not in html2
    assert "rel. L2 (native grid)" in html2  # the metrics row is always present


def test_fig_curves_native():
    res = [dict(_res(32, 30, 0.056), rel_l2_domain=0.0578)]
    native = {
        "results": [{"nk": 32, "nphi": 30, "n_coeffs": 960, "rel_l2": 0.0562}],
    }
    png = _fig_curves(res, native=native)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # without a native sweep the figure still renders
    assert _fig_curves(res)[:8] == b"\x89PNG\r\n\x1a\n"


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
