"""Tests of the reconstruction of a spectrum from its CWAVE parameters."""

import numpy as np
import pytest
import xarray as xr
from cwaves import compute_cwaves, reconstruct_spectrum

KMAX = 2 * np.pi / 25


def _spectrum(full: bool = False, n: int = 161) -> xr.DataArray:
    """Smooth positive two-peak spectrum on a cartesian grid (rad/m)."""
    k_rg = np.linspace(-KMAX if full else 0.0, KMAX, n if full else n // 2 + 1)
    k_az = np.linspace(-KMAX, KMAX, n)
    krg, kaz = np.meshgrid(k_rg, k_az, indexing="ij")

    def peak(x0: float, y0: float) -> np.ndarray:
        return np.exp(-((krg - x0) ** 2 + (kaz - y0) ** 2) / (2 * 0.012**2))

    f = peak(0.05, 0.03) + peak(-0.05, -0.03) + 0.4 * peak(0.03, -0.06)
    f += 0.4 * peak(-0.03, 0.06)
    return xr.DataArray(
        f, dims=("x", "y"), coords={"k_rg": ("x", k_rg), "k_az": ("y", k_az)}
    )


def _truth_error(spec: xr.DataArray, rec: xr.DataArray) -> float:
    """Relative L2 error on the kernel domain (where rec != 0)."""
    from cwaves.grid import build_quadrature

    quad = build_quadrature(spec)
    kn = np.asarray(quad.k_norm.transpose(*spec.dims))
    mask = (kn > 2 * np.pi / 600) & (kn < KMAX)
    w = np.asarray(quad.weight.transpose(*spec.dims))
    mass = float((np.abs(spec.values) * w)[mask].sum())
    gt = (spec / mass).values
    dom = rec.values != 0
    err = (((rec.values - gt) ** 2) * w)[dom].sum()
    return float(np.sqrt(err / ((gt**2) * w)[dom].sum()))


@pytest.mark.parametrize("full", [False, True])
def test_mem_reproduces_parameters_and_is_positive(full):
    spec = _spectrum(full)
    c = compute_cwaves(spec, cwave_Nk=8, cwave_Nphi=9)
    rec = reconstruct_spectrum(c, spec, method="mem")
    assert rec.dims == spec.dims
    assert float(rec.min()) >= 0.0
    c2 = compute_cwaves(rec, cwave_Nk=8, cwave_Nphi=9)
    # the annulus mass of rec differs from 1 only by the out-of-domain mass
    # of the original spectrum: compare the parameters up to that scale
    scale = float((c2 * c).sum() / (c2 * c2).sum())
    np.testing.assert_allclose(scale * c2, c, atol=1e-6 * float(np.abs(c).max()))
    assert rec.attrs["constraint_residual"] < 1e-6 * float(np.abs(c).max())


def test_mem_beats_linear():
    spec = _spectrum()
    c = compute_cwaves(spec, cwave_Nk=8, cwave_Nphi=9)
    e_mem = _truth_error(spec, reconstruct_spectrum(c, spec, method="mem"))
    e_lin = _truth_error(spec, reconstruct_spectrum(c, spec, method="linear"))
    assert e_mem < e_lin


def test_linear_is_finite():
    spec = _spectrum()
    c = compute_cwaves(spec, cwave_Nk=4, cwave_Nphi=5)
    rec = reconstruct_spectrum(c, spec, method="linear")
    c2 = compute_cwaves(rec.where(rec != 0, 0.0) + 0.0, cwave_Nk=4, cwave_Nphi=5)
    # linear reconstruction may be negative: only check it is a finite field
    assert np.isfinite(rec).all()
    assert c2.shape == c.shape


def test_mem_with_noise_sigma():
    spec = _spectrum()
    c = compute_cwaves(spec, cwave_Nk=4, cwave_Nphi=5)
    noise = 0.01 * float(np.sqrt((c**2).mean()))
    rng = np.random.default_rng(0)
    cn = c + noise * rng.standard_normal(c.shape)
    rec = reconstruct_spectrum(cn, spec, method="mem", sigma=noise)
    assert float(rec.min()) >= 0.0
    assert _truth_error(spec, rec) < 1.0


def test_complex_linear():
    spec = _spectrum() * (1 + 0.1j)
    c = compute_cwaves(spec, cwave_Nk=4, cwave_Nphi=5, part="complex")
    rec = reconstruct_spectrum(c, spec, method="linear")
    assert np.iscomplexobj(rec.values)


def test_errors():
    spec = _spectrum()
    c = compute_cwaves(spec)
    with pytest.raises(ValueError, match="method"):
        reconstruct_spectrum(c, spec, method="transport")
    ci = compute_cwaves(spec * 1j, part="imag")
    with pytest.raises(ValueError, match="positive density"):
        reconstruct_spectrum(ci, spec, method="mem")
    with pytest.raises(ValueError, match="one parameter set"):
        reconstruct_spectrum(c.expand_dims(t=2), spec)
    bad = c.copy()
    bad.attrs["part"] = "phase"
    with pytest.raises(ValueError, match="unknown part"):
        reconstruct_spectrum(bad, spec)


def test_no_convergence_warns(caplog):
    spec = _spectrum()
    c = compute_cwaves(spec, cwave_Nk=8, cwave_Nphi=9)
    with caplog.at_level("WARNING"):
        reconstruct_spectrum(c, spec, method="mem", max_iter=1)
    assert "did not converge" in caplog.text
