"""Tests of the grid detection / quadrature (``cwaves.grid``) and of the single
entry point ``cwaves.compute_cwaves`` on every supported grid.

The same analytic spectrum is sampled on a fine uniform cartesian grid, a
log-polar grid (OCN-like), a sinh cartesian grid (OCN ``oswCartSpec``-like),
half and full planes: all must give the same CWAVE parameters (they are
quadratures of the same integral).
"""

import numpy as np
import pytest
import xarray as xr
from cwaves import compute_cwaves
from cwaves.grid import (
    build_quadrature,
    cell_widths,
    coordinate_extra_dims,
    polar_k_weights,
    polar_phi_weights,
)

NK, NPHI = 6, 9
SIG = 0.01
#: two wave systems, not centrally symmetric
SYSTEMS = ((0.04, 0.03, 1.0), (-0.02, 0.05, 0.3))


def _f(krg, kaz):
    """Analytic (not centrally symmetric) spectrum."""
    return sum(
        a * np.exp(-((krg - x0) ** 2 + (kaz - y0) ** 2) / (2 * SIG**2))
        for x0, y0, a in SYSTEMS
    )


def _f_even(krg, kaz):
    """Its centrally-symmetric part (f(k) + f(-k)) / 2."""
    return 0.5 * (_f(krg, kaz) + _f(-krg, -kaz))


def _cartesian(krg, kaz, func=_f_even):
    kr, ka = np.meshgrid(krg, kaz, indexing="ij")
    return xr.DataArray(
        func(kr, ka),
        dims=("x", "y"),
        coords={"k_rg": ("x", krg), "k_az": ("y", kaz)},
    )


def _polar(k, phi, func=_f):
    kk, pp = np.meshgrid(k, phi, indexing="ij")
    return xr.DataArray(
        func(kk * np.cos(pp), kk * np.sin(pp)),
        dims=("k", "phi"),
        coords={"k": k, "phi": phi},
    )


@pytest.fixture(scope="module")
def reference():
    """CWAVE parameters of the even part on a fine uniform half-plane grid."""
    spec = _cartesian(np.linspace(0, 0.3, 600), np.linspace(-0.3, 0.3, 1201))
    return compute_cwaves(spec, cwave_Nk=NK, cwave_Nphi=NPHI)


def _rel(a, b):
    return float(abs(a - b).max() / abs(b).max())


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_cell_widths_uniform_and_clipping():
    v = np.arange(5.0)
    np.testing.assert_allclose(cell_widths(v), 1.0)
    # sample on the domain edge: its cell keeps only the inner half
    np.testing.assert_allclose(cell_widths(v, lower=0.0), [0.5, 1, 1, 1, 1])
    # first sample one step away from the edge: the edge strip is covered
    np.testing.assert_allclose(cell_widths(v + 1.0, lower=0.0), [1.5, 1, 1, 1, 1])
    # decreasing axis
    np.testing.assert_allclose(cell_widths(v[::-1], lower=0.0), [1, 1, 1, 1, 0.5])


def test_cell_widths_non_uniform_sum_to_extent():
    v = np.sinh(np.linspace(0, 3, 40))
    w = cell_widths(v, lower=0.0)
    assert np.isclose(w.sum(), v[-1] + 0.5 * (v[-1] - v[-2]))


def test_polar_k_weights_log_grid_exact():
    k = np.exp(np.linspace(np.log(0.01), np.log(0.2), 30))
    w = polar_k_weights(k)
    dlog = np.log(k[1] / k[0])
    lo, hi = k[0] * np.exp(-dlog / 2), k[-1] * np.exp(dlog / 2)
    assert np.isclose(w.sum(), 0.5 * (hi**2 - lo**2))


def test_polar_k_weights_linear_grid():
    k = np.linspace(0.01, 0.2, 20)
    w = polar_k_weights(k)
    dk = k[1] - k[0]
    assert np.isclose(w.sum(), 0.5 * ((k[-1] + dk / 2) ** 2 - (k[0] - dk / 2) ** 2))


def test_polar_phi_weights_full_circle_and_arc():
    phi = np.deg2rad(np.arange(0, 360, 5.0))
    w, full = polar_phi_weights(phi)
    assert full and np.isclose(w.sum(), 2 * np.pi)
    arc = np.deg2rad(np.arange(-90, 90.1, 5.0))
    w, full = polar_phi_weights(arc, arc=(-np.pi / 2, np.pi / 2))
    assert not full and np.isclose(w.sum(), np.pi)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detection_kinds_and_coverage():
    half = _cartesian(np.linspace(0, 0.3, 50), np.linspace(-0.3, 0.3, 61))
    full = _cartesian(np.linspace(-0.3, 0.3, 61), np.linspace(-0.3, 0.3, 61))
    pol = _polar(np.geomspace(0.01, 0.3, 30), np.deg2rad(np.arange(0, 360, 10.0)))
    pol_half = _polar(np.geomspace(0.01, 0.3, 30), np.deg2rad(np.arange(-90, 91, 10.0)))
    assert (build_quadrature(half).kind, build_quadrature(half).coverage) == (
        "cartesian",
        "half",
    )
    assert build_quadrature(full).coverage == "full"
    assert (build_quadrature(pol).kind, build_quadrature(pol).coverage) == (
        "polar",
        "full",
    )
    assert build_quadrature(pol_half).coverage == "half"


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (xr.DataArray(np.ones((3, 3)), dims=("a", "b")), "unsupported grid"),
        (
            xr.DataArray(
                np.ones((3, 3)),
                dims=("k", "phi"),
                coords={"k": [0.1, 0.2, 0.3], "phi": [0.0, 90.0, 300.0]},
            ),
            "RADIANS",
        ),
        (
            xr.DataArray(
                np.ones((3, 3)),
                dims=("x", "y"),
                coords={"k_rg": ("x", [0.0, 0.2, 0.1]), "k_az": ("y", [-1, 0, 1.0])},
            ),
            "monotonic",
        ),
        (
            xr.DataArray(
                np.ones((4, 3)),
                dims=("x", "y"),
                coords={
                    "k_rg": ("x", [-0.01, 0.1, 0.2, 0.3]),
                    "k_az": ("y", [-1, 0, 1.0]),
                },
            ),
            "not symmetrically",
        ),
        (
            xr.DataArray(
                np.ones((3, 3)),
                dims=("x", "y"),
                coords={
                    "k_rg": ("x", [0.0, np.nan, 0.2]),
                    "k_az": ("y", [-1, 0, 1.0]),
                },
            ),
            "NaN",
        ),
    ],
)
def test_detection_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        build_quadrature(spec)


def test_not_a_dataarray():
    with pytest.raises(TypeError, match="DataArray"):
        build_quadrature(np.ones((3, 3)))  # type: ignore[arg-type]


def test_zero_mass_raises():
    spec = _cartesian(np.linspace(0, 0.3, 50), np.linspace(-0.3, 0.3, 61)) * 0
    with pytest.raises(ValueError, match="zero"):
        compute_cwaves(spec)


def test_coordinate_extra_dims():
    spec = xr.DataArray(
        np.ones((2, 4, 3)),
        dims=("tile", "x", "y"),
        coords={
            "k_rg": (("tile", "x"), np.tile(np.linspace(0, 0.3, 4), (2, 1))),
            "k_az": ("y", [-0.1, 0, 0.1]),
        },
    )
    assert coordinate_extra_dims(spec) == ["tile"]


# ---------------------------------------------------------------------------
# one integral, every grid
# ---------------------------------------------------------------------------


def test_polar_full_circle_equals_even_part(reference):
    # full circle of a NON symmetric spectrum == CWAVE of its even part
    spec = _polar(np.geomspace(0.005, 0.3, 120), np.deg2rad(np.arange(0, 360, 2.0)))
    c = compute_cwaves(spec, cwave_Nk=NK, cwave_Nphi=NPHI)
    assert c.attrs["coverage"] == "full"
    assert _rel(c, reference) < 5e-3


def test_polar_half_equals_full_circle():
    k = np.geomspace(0.005, 0.3, 80)
    full = _polar(k, np.deg2rad(np.arange(0, 360, 2.5)), func=_f)
    half = _polar(k, np.deg2rad(np.arange(-90, 90.1, 2.5)), func=_f_even)
    c_full = compute_cwaves(full, cwave_Nk=NK, cwave_Nphi=NPHI)
    c_half = compute_cwaves(half, cwave_Nk=NK, cwave_Nphi=NPHI)
    assert _rel(c_half, c_full) < 1e-10


def test_sinh_grid_matches_reference(reference):
    s = 0.004 * np.sinh(np.linspace(0, np.arcsinh(0.3 / 0.004), 70))
    spec = _cartesian(s, np.concatenate([-s[::-1][:-1], s]))
    c = compute_cwaves(spec, cwave_Nk=NK, cwave_Nphi=NPHI)
    assert _rel(c, reference) < 1e-3


def test_cartesian_full_plane_equals_even_part(reference):
    krg = np.linspace(-0.3, 0.3, 601)
    spec = _cartesian(krg, np.linspace(-0.3, 0.3, 601), func=_f)
    c = compute_cwaves(spec, cwave_Nk=NK, cwave_Nphi=NPHI)
    assert c.attrs["coverage"] == "full"
    assert _rel(c, reference) < 5e-3


def test_half_grid_without_dc_column(reference):
    # an FFT grid whose k_rg = 0 column was dropped (SARWave L1B XSP)
    dk = 0.3 / 300
    spec = _cartesian(np.arange(1, 301) * dk, np.linspace(-0.3, 0.3, 601))
    c = compute_cwaves(spec, cwave_Nk=NK, cwave_Nphi=NPHI)
    assert _rel(c, reference) < 5e-3


def test_extra_dims_broadcast():
    base = _cartesian(np.linspace(0, 0.3, 100), np.linspace(-0.3, 0.3, 121))
    stacked = xr.concat([base, 2 * base], dim="pol")
    c = compute_cwaves(stacked)
    assert c.sizes["pol"] == 2
    # normalised spectrum: a global factor does not change the parameters
    np.testing.assert_allclose(c.isel(pol=0), c.isel(pol=1))


def test_modulus_only():
    base = _cartesian(np.linspace(0, 0.3, 100), np.linspace(-0.3, 0.3, 121))
    np.testing.assert_allclose(
        compute_cwaves(base), compute_cwaves(base * np.exp(1j * 0.7))
    )


# ---------------------------------------------------------------------------
# cell-integrated kernel weights (quadrature="cell")
# ---------------------------------------------------------------------------


def test_cell_bounds_tile_the_axis():
    from cwaves.grid import cell_bounds

    lo, hi = cell_bounds(np.sinh(np.linspace(0, 3, 20)), lower=0.0)
    np.testing.assert_allclose(lo[1:], hi[:-1])
    assert lo[0] == 0.0
    lo, hi = cell_bounds(np.linspace(1, 0, 11), lower=0.0)  # decreasing axis
    assert np.all(hi >= lo) and np.isclose(lo[-1], 0.0)
