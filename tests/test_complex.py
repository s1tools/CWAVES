"""Tests of the complex CWAVE extension (``compute_cwaves(part=...)``).

The real part of a SAR cross-spectrum is centrally symmetric, its imaginary
part anti-symmetric (Hermitian symmetry): the even (pi-periodic) harmonics of
the 2007 paper are the basis of the former, the odd (pi-antiperiodic)
harmonics of the latter. Both are orthonormal and complete on [0, pi].
"""

import numpy as np
import pytest
import xarray as xr
from cwaves import compute_cwaves, harmonic_functions
from cwaves.client import _xsp_spectrum

PHI = (np.arange(20000) + 0.5) * np.pi / 20000  # midpoint rule on [0, pi]


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_harmonics_orthonormal_on_half_circle(parity):
    f = np.array([harmonic_functions(PHI, p, parity=parity) for p in range(1, 10)])
    gram = f @ f.T * (np.pi / PHI.size)
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-6)


def test_harmonics_parity_under_point_reflection():
    # phi -> phi + pi is k -> -k: even set unchanged, odd set flips sign
    for p in range(1, 8):
        e = harmonic_functions(PHI, p, "even")
        o = harmonic_functions(PHI, p, "odd")
        np.testing.assert_allclose(harmonic_functions(PHI + np.pi, p, "even"), e)
        np.testing.assert_allclose(harmonic_functions(PHI + np.pi, p, "odd"), -o)


def test_harmonics_bad_parity():
    with pytest.raises(ValueError, match="parity"):
        harmonic_functions(PHI, 1, parity="foo")


def _hermitian(krg, kaz, theta=0.8):
    """Two wave systems, X(-k) = conj X(k); one close to the k_az axis."""

    def g(x0, y0, s):
        return np.exp(-((krg - x0) ** 2 + (kaz - y0) ** 2) / (2 * s**2))

    out = 0
    for x0, y0, s, a, th in (
        (0.004, 0.045, 0.008, 1.0, theta),
        (0.05, -0.02, 0.01, 0.6, -0.5),
    ):
        out = out + a * (
            g(x0, y0, s) * np.exp(1j * th) + g(-x0, -y0, s) * np.exp(-1j * th)
        )
    return out


def _spec(krg, kaz, func=_hermitian):
    kr, ka = np.meshgrid(krg, kaz, indexing="ij")
    return xr.DataArray(
        func(kr, ka), dims=("x", "y"), coords={"k_rg": ("x", krg), "k_az": ("y", kaz)}
    )


def test_parts_consistency():
    spec = _spec(np.linspace(0, 0.3, 200), np.linspace(-0.3, 0.3, 301))
    cw = {"cwave_Nk": 4, "cwave_Nphi": 5}
    c = compute_cwaves(spec, part="complex", **cw)
    assert np.iscomplexobj(c.values)
    np.testing.assert_allclose(c.real, compute_cwaves(spec, part="real", **cw))
    np.testing.assert_allclose(c.imag, compute_cwaves(spec, part="imag", **cw))
    assert c.attrs["part"] == "complex"
    # default unchanged: modulus on the even harmonics
    np.testing.assert_allclose(
        compute_cwaves(spec, **cw), compute_cwaves(np.abs(spec), part="real", **cw)
    )


def test_bad_part():
    spec = _spec(np.linspace(0, 0.3, 20), np.linspace(-0.3, 0.3, 31))
    with pytest.raises(ValueError, match="part"):
        compute_cwaves(spec, part="phase")


def test_complex_half_plane_equals_full_plane():
    # Hermitian spectrum: half-plane (Hermitian assumed) == full plane
    half = _spec(np.linspace(0, 0.3, 301), np.linspace(-0.3, 0.3, 601))
    full = _spec(np.linspace(-0.3, 0.3, 601), np.linspace(-0.3, 0.3, 601))
    a = compute_cwaves(half, part="complex", cwave_Nk=5, cwave_Nphi=7)
    b = compute_cwaves(full, part="complex", cwave_Nk=5, cwave_Nphi=7)
    assert float(abs(a - b).max() / abs(a).max()) < 5e-3


def test_even_kernels_blind_to_antisymmetric_part_on_full_plane():
    # on the full plane the even kernels only see the centrally symmetric
    # part: a purely anti-symmetric (real) field projects to ~0 on them, and
    # is captured by the odd kernels
    full = _spec(np.linspace(-0.3, 0.3, 401), np.linspace(-0.3, 0.3, 401))
    anti = np.imag(full)  # real-valued, f(-k) = -f(k)
    on_even = compute_cwaves(anti, part="real", cwave_Nk=5, cwave_Nphi=7)
    on_odd = compute_cwaves(1j * anti, part="imag", cwave_Nk=5, cwave_Nphi=7)
    assert float(abs(on_even).max()) < 1e-6 * float(abs(on_odd).max()) + 1e-12
    assert float(abs(on_odd).max()) > 0.1


def test_imag_cwave_scales_with_lag(l1b_xsp_nc):
    # phase of a cross-spectrum ~ omega(k) * tau: the imaginary CWAVE
    # parameters double from 1-tau to 2-tau, the real ones barely change
    with xr.open_dataset(l1b_xsp_nc) as ds:
        x1 = _xsp_spectrum(ds, 1, 0, 0).load()
        x2 = _xsp_spectrum(ds, 2, 0, 0).load()
    c1 = compute_cwaves(x1, part="complex", cwave_Nk=8, cwave_Nphi=9).values.ravel()
    c2 = compute_cwaves(x2, part="complex", cwave_Nk=8, cwave_Nphi=9).values.ravel()
    slope = np.dot(c1.imag, c2.imag) / np.dot(c1.imag, c1.imag)
    assert 1.8 < slope < 2.2
    assert np.corrcoef(c1.imag, c2.imag)[0, 1] > 0.99
    assert np.abs(c2.real - c1.real).max() / np.abs(c1.real).max() < 0.06
