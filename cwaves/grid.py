"""Wavenumber-grid detection and quadrature weights for the CWAVE projection.

The CWAVE parameters are integrals over the wavenumber plane,
:math:`c_i = \\int_A K_i(\\mathbf{k})\\, f(\\mathbf{k})\\, \\mathrm{d}A`. The grid on
which the spectrum ``f`` is sampled only enters through the **quadrature rule**
used to evaluate these integrals: a set of sample points
:math:`(k_{rg}, k_{az})` and one area weight per sample. This module turns any
supported input grid into such a rule, so that a single entry point
(:func:`cwaves.compute_cwaves`) accepts every grid without resampling the
spectrum:

* **cartesian** grids -- coordinates ``k_rg`` (range) and ``k_az`` (azimuth),
  each 1-D along its own dimension, in rad/m. The spacing may be **uniform**
  (FFT grids, SARWave L1B XSP) or **non-uniform** (e.g. the sinh grid of the
  ESA OCN ``oswCartSpec``): the cell of each sample extends half-way to its
  neighbours, so the weight is the exact cell area of that tessellation
  (for a uniform grid it is the constant ``dk_rg * dk_az``, i.e. the historical
  behaviour).
* **polar** grids -- coordinates ``k`` (wavenumber magnitude, rad/m) and
  ``phi`` (direction, **radians**), each 1-D along its own dimension, with the
  convention ``k_rg = k cos(phi)``, ``k_az = k sin(phi)`` (``phi = 0`` is the
  range axis, ``phi = pi/2`` the positive azimuth axis). The weight is the
  polar cell area ``k dk dphi`` (log-spaced ``k`` grids use geometric cell
  boundaries, linear grids arithmetic ones; a full-circle ``phi`` axis is
  treated as periodic).

Each grid covers either the **half plane** ``k_rg >= 0`` (the domain of the
2007 paper, on which the kernel basis is orthonormal) or the **full plane**.
On the full plane the integral is taken with the kernels evaluated as they
are, which are even (``K(-k) = K(k)``, pi-periodic harmonics): the result is
exactly the CWAVE parameters of the centrally-symmetric part
``(f(k) + f(-k)) / 2`` of the spectrum (see ``repository_specifications.md``
section 6). Coverage is detected from the coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import xarray as xr

#: accepted coordinate names, per grid type
CARTESIAN_COORDS = ("k_rg", "k_az")
POLAR_COORDS = ("k", "phi")

#: relative tolerance used to detect a log-uniform (geometric) ``k`` axis
_LOG_UNIFORM_RTOL = 1.0e-3


@dataclass(frozen=True)
class Quadrature:
    """A quadrature rule on the wavenumber plane for a given input grid.

    Attributes:
        kind: ``"cartesian"`` or ``"polar"``.
        coverage: ``"half"`` (``k_rg >= 0`` only) or ``"full"`` plane.
        dims: the two grid dimensions of the input spectrum.
        k_rg: range wavenumber of every sample (rad/m), over ``dims``.
        k_az: azimuth wavenumber of every sample (rad/m), over ``dims``.
        weight: area weight of every sample (rad^2/m^2), over ``dims``.
        cells: per grid axis (same order as ``dims``), the ``(lower, upper)``
            bounds of the cell of every sample -- in (k_rg, k_az) for a
            cartesian grid, in (k, phi) for a polar grid. ``weight`` is the
            exact area of these cells; they tile the integration domain.
    """

    kind: str
    coverage: str
    dims: tuple[str, str]
    k_rg: xr.DataArray
    k_az: xr.DataArray
    weight: xr.DataArray
    cells: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

    @property
    def k_norm(self) -> xr.DataArray:
        """Wavenumber magnitude ``sqrt(k_rg^2 + k_az^2)`` of every sample."""
        return cast("xr.DataArray", np.sqrt(self.k_rg**2 + self.k_az**2))


def _coord_dim(spectrum: xr.DataArray, name: str) -> str:
    """Return the grid dimension of the coordinate ``name`` (its last dim)."""
    coord = spectrum.coords[name]
    if coord.ndim != 1:
        raise ValueError(
            f"coordinate {name!r} must be 1-D (one value per grid line), got "
            f"dims {coord.dims}. Select a single tile/time first."
        )
    return str(coord.dims[0])


def grid_coord_names(spectrum: xr.DataArray) -> tuple[str, str] | None:
    """Return the wavenumber coordinate pair of ``spectrum`` (or ``None``)."""
    if all(c in spectrum.coords for c in CARTESIAN_COORDS):
        return CARTESIAN_COORDS
    if all(c in spectrum.coords for c in POLAR_COORDS):
        return POLAR_COORDS
    return None


def coordinate_extra_dims(spectrum: xr.DataArray) -> list[str]:
    """Dimensions along which the wavenumber grid itself varies.

    A product may store one wavenumber axis per tile / time (e.g. ``k_rg`` with
    dims ``(tile_line, tile_sample, freq_sample)``). The **last** dimension of
    each wavenumber coordinate is its grid dimension; the other ones are
    returned here, so that the caller can process the spectrum slice by slice.
    """
    names = grid_coord_names(spectrum)
    if names is None:
        return []
    extra: list[str] = []
    for name in names:
        for d in spectrum.coords[name].dims[:-1]:
            if d not in extra:
                extra.append(str(d))
    return extra


def _check_axis(values: np.ndarray, name: str) -> None:
    """Raise unless ``values`` is a finite, strictly monotonic 1-D axis."""
    if values.size < 2:
        raise ValueError(f"coordinate {name!r} needs at least 2 samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"coordinate {name!r} contains NaN/inf values; drop those grid lines "
            "before calling the CWAVE computation"
        )
    d = np.diff(values)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"coordinate {name!r} must be strictly monotonic")


def cell_bounds(
    values: np.ndarray, lower: float | None = None, upper: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds ``(lo, hi)`` (``lo <= hi``) of the cell of every sample of an axis.

    The cell boundaries are the mid-points between adjacent samples; the two
    outer cells extend half of the adjacent spacing beyond the first / last
    sample. The cells are then fitted to the domain ``[lower, upper]``: they
    are clipped to it (on a half-plane grid the cell of a ``k_rg = 0`` sample
    only keeps its ``k_rg >= 0`` half, the trapezoidal rule at the domain
    edge) and, when the outermost sample lies within one grid step of the edge,
    its cell is extended to the edge (a grid starting at ``k_rg = dk`` still
    covers the ``[0, dk/2]`` strip).

    Args:
        values: strictly monotonic 1-D array.
        lower: optional lower bound of the integration domain.
        upper: optional upper bound of the integration domain.

    Returns:
        ``(lo, hi)``, each of the shape of ``values`` (original sample order).
    """
    v = np.asarray(values, dtype=float)
    sign = 1.0 if v[-1] > v[0] else -1.0
    u = sign * v  # increasing
    mid = 0.5 * (u[1:] + u[:-1])
    b = np.concatenate(
        [[u[0] - 0.5 * (u[1] - u[0])], mid, [u[-1] + 0.5 * (u[-1] - u[-2])]]
    )
    lo, hi = lower, upper
    if sign < 0:
        lo, hi = (
            (-upper if upper is not None else None),
            (-lower if lower is not None else None),
        )
    # snap the outer boundary onto the domain edge when the first / last sample
    # lies within one grid step of it (e.g. an FFT grid whose k_rg = 0 column
    # was dropped): the edge strip is integrated instead of being lost
    if lo is not None and u[0] - lo <= (u[1] - u[0]) * (1.0 + 1.0e-9):
        b[0] = lo
    if hi is not None and hi - u[-1] <= (u[-1] - u[-2]) * (1.0 + 1.0e-9):
        b[-1] = hi
    if lo is not None or hi is not None:
        b = np.clip(b, lo, hi)
    if sign > 0:
        return b[:-1], b[1:]
    return -b[1:], -b[:-1]


def cell_widths(
    values: np.ndarray, lower: float | None = None, upper: float | None = None
) -> np.ndarray:
    """Width of the cell of every sample of a monotonic 1-D axis.

    See :func:`cell_bounds` (mid-point cells fitted to ``[lower, upper]``);
    for a uniform axis without bounds every width equals the spacing.
    """
    lo, hi = cell_bounds(values, lower, upper)
    return np.asarray(hi - lo)


def _is_log_uniform(k: np.ndarray) -> bool:
    """True if ``k`` is (to ``_LOG_UNIFORM_RTOL``) a geometric progression."""
    if np.any(k <= 0):
        return False
    r = k[1:] / k[:-1]
    return bool(np.allclose(r, r[0], rtol=_LOG_UNIFORM_RTOL))


def polar_k_bounds(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial cell bounds of a strictly increasing, positive ``k`` axis.

    For a log-uniform axis (e.g. the ESA OCN ``oswK``) the boundaries are the
    geometric means of adjacent samples (the samples are the cell centres in
    log space); otherwise the arithmetic mid-points. The outer boundaries
    extend half of the adjacent step.
    """
    k = np.asarray(k, dtype=float)
    if _is_log_uniform(k):
        lnk = np.log(k)
        lnb = np.concatenate(
            [
                [lnk[0] - 0.5 * (lnk[1] - lnk[0])],
                0.5 * (lnk[1:] + lnk[:-1]),
                [lnk[-1] + 0.5 * (lnk[-1] - lnk[-2])],
            ]
        )
        b = np.exp(lnb)
    else:
        b = np.concatenate(
            [
                [k[0] - 0.5 * (k[1] - k[0])],
                0.5 * (k[1:] + k[:-1]),
                [k[-1] + 0.5 * (k[-1] - k[-2])],
            ]
        )
        b = np.clip(b, 0.0, None)
    return b[:-1], b[1:]


def polar_k_weights(k: np.ndarray) -> np.ndarray:
    """Radial weights ``w_i`` such that ``sum_i w_i f(k_i) ~ int f(k) k dk``.

    The weight is the exact ``k dk`` measure of the cell
    ``[lo_i, hi_i]`` of :func:`polar_k_bounds`, ``(hi_i^2 - lo_i^2) / 2``.

    Args:
        k: strictly increasing, strictly positive 1-D array (rad/m).

    Returns:
        The radial weights, same shape as ``k``.
    """
    lo, hi = polar_k_bounds(k)
    return np.asarray(0.5 * (hi**2 - lo**2))


def polar_phi_bounds(
    phi: np.ndarray, arc: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Angular cell bounds of a strictly increasing ``phi`` axis (radians).

    If the axis spans the full circle (its samples plus one more step reach
    ``2 pi``) the cells are periodic (wrap-around between the last and the
    first sample); otherwise the axis is an arc and the outer cells extend half
    of the adjacent step, fitted to ``arc`` if given (see :func:`cell_bounds`).

    Returns:
        ``(lo, hi, full_circle)``.
    """
    phi = np.asarray(phi, dtype=float)
    d = np.diff(phi)
    wrap = phi[0] + 2.0 * np.pi - phi[-1]
    full_circle = bool(wrap <= 1.5 * float(np.max(d)) + 1.0e-9)
    if full_circle:
        dd = np.concatenate([d, [wrap]])  # dd[i] = phi[i+1] - phi[i] (wrapped)
        return phi - 0.5 * np.roll(dd, 1), phi + 0.5 * dd, True
    if arc is None:
        lo, hi = cell_bounds(phi)
    else:
        lo, hi = cell_bounds(phi, lower=arc[0], upper=arc[1])
    return lo, hi, False


def polar_phi_weights(
    phi: np.ndarray, arc: tuple[float, float] | None = None
) -> tuple[np.ndarray, bool]:
    """Angular cell widths of a strictly increasing ``phi`` axis (radians).

    If the axis spans the full circle (its samples plus one more step reach
    ``2 pi``) the widths are periodic (wrap-around between the last and the
    first sample); otherwise the axis is an arc and the outer cells extend half
    of the adjacent step, clipped to ``arc`` if given.

    Args:
        phi: strictly increasing 1-D array of angles in radians.
        arc: optional ``(lower, upper)`` bounds of the integration domain (for
            an arc).

    Returns:
        ``(widths, full_circle)``.
    """
    lo, hi, full_circle = polar_phi_bounds(phi, arc)
    return np.asarray(hi - lo), full_circle


def _cartesian(spectrum: xr.DataArray) -> Quadrature:
    dim_rg = _coord_dim(spectrum, "k_rg")
    dim_az = _coord_dim(spectrum, "k_az")
    if dim_rg == dim_az:
        raise ValueError("k_rg and k_az must be defined along two different dims")
    krg = np.asarray(spectrum.coords["k_rg"].values, dtype=float)
    kaz = np.asarray(spectrum.coords["k_az"].values, dtype=float)
    _check_axis(krg, "k_rg")
    _check_axis(kaz, "k_az")
    cells_az = cell_bounds(kaz)
    if krg.min() >= 0.0:
        coverage = "half"
        # clip the cells at the k_rg = 0 domain edge (half-plane integral)
        cells_rg = cell_bounds(krg, lower=0.0)
    else:
        cells_rg = cell_bounds(krg)
        coverage = "full"
    w_rg = cells_rg[1] - cells_rg[0]
    w_az = cells_az[1] - cells_az[0]
    # a full-plane grid must cover negative and positive range wavenumbers
    # symmetrically (to one cell, e.g. an even-sized FFT grid), otherwise the
    # normalisation mass would be biased towards one half
    if coverage == "full" and abs(krg.min() + krg.max()) > 2.0 * float(np.max(w_rg)):
        raise ValueError(
            "k_rg covers negative wavenumbers but not symmetrically "
            f"(min {krg.min():.4g}, max {krg.max():.4g} rad/m). Pass either a "
            "half-plane spectrum (k_rg >= 0) or a full-plane spectrum with "
            "k_rg symmetric around 0."
        )
    k_rg = xr.DataArray(krg, dims=(dim_rg,))
    k_az = xr.DataArray(kaz, dims=(dim_az,))
    k_rg, k_az = xr.broadcast(k_rg, k_az)
    weight = xr.DataArray(w_rg, dims=(dim_rg,)) * xr.DataArray(w_az, dims=(dim_az,))
    dims = (dim_rg, dim_az)
    return Quadrature(
        "cartesian",
        coverage,
        dims,
        k_rg.transpose(*dims),
        k_az.transpose(*dims),
        weight.transpose(*dims),
        (cells_rg, cells_az),
    )


def _polar(spectrum: xr.DataArray) -> Quadrature:
    dim_k = _coord_dim(spectrum, "k")
    dim_phi = _coord_dim(spectrum, "phi")
    if dim_k == dim_phi:
        raise ValueError("k and phi must be defined along two different dims")
    k = np.asarray(spectrum.coords["k"].values, dtype=float)
    phi = np.asarray(spectrum.coords["phi"].values, dtype=float)
    _check_axis(k, "k")
    _check_axis(phi, "phi")
    if np.any(np.diff(k) <= 0) or k[0] <= 0:
        raise ValueError("k must be strictly positive and strictly increasing (rad/m)")
    if np.any(np.diff(phi) <= 0):
        raise ValueError("phi must be strictly increasing (radians)")
    if phi[-1] - phi[0] >= 2.0 * np.pi:
        raise ValueError(
            f"phi spans {phi[-1] - phi[0]:.4g} >= 2*pi: phi must be in RADIANS "
            "(convert degrees with np.deg2rad) and cover at most one full circle"
        )
    phi_lo, phi_hi, full_circle = polar_phi_bounds(phi)
    k_lo, k_hi = polar_k_bounds(k)
    cos_phi = np.cos(phi)
    if full_circle or (cos_phi.min() < -1.0e-6 and cos_phi.max() > 1.0e-6):
        coverage = "full"
    elif cos_phi.min() >= -1.0e-6:
        coverage = "half"
    else:
        raise ValueError(
            "the phi axis only covers k_rg < 0 directions; the convention is "
            "k_rg = k cos(phi) (phi = 0 along +range). Pass the k_rg >= 0 half "
            "(phi in [-pi/2, pi/2]) or the full circle."
        )
    if coverage == "full" and not full_circle:
        raise ValueError(
            "the phi axis covers both range half planes but not the full circle; "
            "pass either the k_rg >= 0 half (phi in [-pi/2, pi/2]) or the full circle"
        )
    if coverage == "half":
        if phi[0] < -0.5 * np.pi - 1.0e-6 or phi[-1] > 0.5 * np.pi + 1.0e-6:
            raise ValueError(
                "a half-plane polar grid must have phi in [-pi/2, pi/2] "
                "(k_rg = k cos(phi) >= 0)"
            )
        # clip the arc cells at the k_rg = 0 domain edge (phi = -/+ pi/2)
        phi_lo, phi_hi, _ = polar_phi_bounds(phi, arc=(-0.5 * np.pi, 0.5 * np.pi))
    w_phi = phi_hi - phi_lo
    w_k = 0.5 * (k_hi**2 - k_lo**2)
    kk = xr.DataArray(k, dims=(dim_k,))
    pp = xr.DataArray(phi, dims=(dim_phi,))
    dims = (dim_k, dim_phi)
    k_rg = (kk * np.cos(pp)).transpose(*dims)
    k_az = (kk * np.sin(pp)).transpose(*dims)
    weight = (
        xr.DataArray(w_k, dims=(dim_k,)) * xr.DataArray(w_phi, dims=(dim_phi,))
    ).transpose(*dims)
    return Quadrature(
        "polar", coverage, dims, k_rg, k_az, weight, ((k_lo, k_hi), (phi_lo, phi_hi))
    )


def build_quadrature(spectrum: xr.DataArray) -> Quadrature:
    """Detect the grid of ``spectrum`` and build its quadrature rule.

    Args:
        spectrum: the spectrum, with either the cartesian coordinates
            ``k_rg`` / ``k_az`` or the polar coordinates ``k`` / ``phi``
            (see the module docstring for units and conventions).

    Returns:
        The :class:`Quadrature` of the input grid.

    Raises:
        TypeError: if ``spectrum`` is not an ``xarray.DataArray``.
        ValueError: if no supported coordinate pair is found, or the grid is
            malformed (non-monotonic, NaN coordinates, degrees instead of
            radians, partial plane coverage...).
    """
    if not isinstance(spectrum, xr.DataArray):
        raise TypeError(
            "the spectrum must be an xarray.DataArray carrying its wavenumber "
            f"coordinates, got {type(spectrum).__name__}"
        )
    has_cart = all(c in spectrum.coords for c in CARTESIAN_COORDS)
    has_pol = all(c in spectrum.coords for c in POLAR_COORDS)
    if has_cart and has_pol:
        raise ValueError(
            "ambiguous grid: the spectrum carries both cartesian (k_rg, k_az) and "
            "polar (k, phi) coordinates; drop one pair"
        )
    if has_cart:
        return _cartesian(spectrum)
    if has_pol:
        return _polar(spectrum)
    raise ValueError(
        "unsupported grid: the spectrum must carry either the cartesian "
        "coordinates ('k_rg', 'k_az') [rad/m] or the polar coordinates "
        "('k' [rad/m], 'phi' [rad], k_rg = k cos(phi)); found "
        f"{sorted(map(str, spectrum.coords))}"
    )
