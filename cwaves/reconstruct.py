"""Reconstruction of a spectrum from its CWAVE parameters.

The CWAVE parameters are a **projection**: the components of the spectrum
orthogonal to the kernel span are not in them, so no method recovers them from
the parameters alone (:ref:`limits`). Two reconstructions are provided, both on
a grid chosen by the user (any grid accepted by :func:`cwaves.compute_cwaves`):

* ``"linear"``: :math:`\\hat f = \\sum_i c_i K_i` (halved on a full plane, where
  the even kernels have norm :math:`\\sqrt 2`) -- the plain orthonormal
  expansion; it has negative lobes and Gibbs oscillations, and it reproduces
  the parameters only as far as the discrete kernels are orthonormal.
* ``"mem"`` (maximum entropy): the positive density that **reproduces the
  parameters exactly** (to the solver tolerance) and maximises the entropy
  :math:`-\\int \\hat f\\,(\\log(\\hat f/m) - 1)` relative to a flat prior
  :math:`m`. The solution is :math:`\\hat f = m \\exp(\\sum_i \\lambda_i K_i)`
  (dual Newton method). It adds a prior (positivity, smoothness), not
  information; with noisy parameters (``sigma``) the constraints are relaxed
  (Tikhonov on the dual), which is required to avoid fitting the noise.

Evaluated on 62 real spectra (S1A L1B XSP, S1A/S1D OCN WV polar, WW3 polar,
55 Envisat L1C) x 4 bases, with exact parameters, ``"mem"`` is closer to the
spectrum than ``"linear"`` (even clipped to positive values) in 99.6 % of the
cases (median relative-L2 gain 8 %, up to x4 on peaked spectra). With noisy
parameters the gain shrinks with the basis size and vanishes from about
16 x 16 on (see :ref:`limits`).

The reconstruction is the unit-annulus-mass density of
:func:`cwaves.compute_cwaves` (``|f| / mass``) restricted to the kernel domain;
it is zero outside the domain. Only ``part="modulus"`` parameters (positive
density) can be reconstructed by ``"mem"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from cwaves.compute_cwaves import PARTS, _cwave_bounds, compute_kernel
from cwaves.grid import build_quadrature

logger = logging.getLogger(__name__)

#: reconstruction methods of :func:`reconstruct_spectrum`.
METHODS = ("mem", "linear")


def _operator(
    grid: xr.DataArray,
    kmin: float,
    kmax: float,
    nk: int,
    nphi: int,
    parity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Kernel matrix on the kernel domain of ``grid``.

    Returns:
        ``(K, w, domain, quad)``: the ``(nk * nphi, n_domain)`` kernel values,
        the cell areas and the boolean domain mask over the flattened grid.
    """
    quad = build_quadrature(grid)
    in_annulus = (quad.k_norm > kmin) & (quad.k_norm < kmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = compute_kernel(
            krg=quad.k_rg.where(in_annulus),
            kaz=quad.k_az.where(in_annulus),
            kmin=kmin,
            kmax=kmax,
            cwave_Nk=nk,
            cwave_Nphi=nphi,
            parity=parity,
        ).cwave_kernel
    values = np.asarray(
        kernel.transpose("k_gp", "phi_hf", *quad.dims), dtype=float
    ).reshape(nk * nphi, -1)
    domain = np.all(np.isfinite(values), axis=0)
    weight = np.asarray(quad.weight.transpose(*quad.dims), dtype=float).ravel()
    return values[:, domain], weight[domain], domain, quad


def _mem_dual(
    kernel: np.ndarray,
    weight: np.ndarray,
    coeffs: np.ndarray,
    prior: np.ndarray,
    sigma: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Maximum-entropy density by Newton iterations on the convex dual.

    Dual objective ``F(l) = sum w m exp(l.K) - l.c + sigma^2 |l|^2 / 2``;
    its minimiser gives ``g = m exp(l.K)`` with ``K (g w) = c - sigma^2 l``
    (exact constraints when ``sigma = 0``).

    Returns:
        ``(g, max constraint residual, iterations)``.
    """
    reg = sigma**2
    lam = np.zeros(kernel.shape[0])

    def objective(lam_: np.ndarray) -> float:
        with np.errstate(over="ignore"):
            g_ = prior * np.exp(lam_ @ kernel)
        return float((weight * g_).sum() - lam_ @ coeffs + 0.5 * reg * lam_ @ lam_)

    it = 0
    while it < max_iter:
        it += 1
        g = prior * np.exp(lam @ kernel)
        grad = kernel @ (g * weight) - coeffs + reg * lam
        if np.abs(grad).max() < tol:
            break
        hess = (kernel * (g * weight)) @ kernel.T + reg * np.eye(len(lam))
        hess += 1e-12 * np.trace(hess) / len(lam) * np.eye(len(lam))
        step = np.linalg.solve(hess, grad)
        f0, t = objective(lam), 1.0
        while t > 1e-10 and not objective(lam - t * step) <= f0 - 1e-4 * t * (
            grad @ step
        ):
            t *= 0.5
        lam = lam - t * step
    g = prior * np.exp(lam @ kernel)
    residual = float(np.abs(kernel @ (g * weight) - coeffs).max())
    return g, residual, it


def reconstruct_spectrum(
    cwave_params: xr.DataArray,
    grid: xr.DataArray,
    method: str = "mem",
    sigma: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> xr.DataArray:
    """Reconstruct a spectrum from its CWAVE parameters on a given grid.

    Args:
        cwave_params: CWAVE parameters returned by :func:`cwaves.compute_cwaves`
            (dims ``(k_gp, phi_hf)``; the wavelength bounds and ``part`` are read
            from its attributes).
        grid: a ``DataArray`` carrying the wavenumber coordinates of the output
            grid (cartesian ``k_rg``/``k_az`` or polar ``k``/``phi``, half or
            full plane); its values are not used.
        method: ``"mem"`` (default, maximum entropy, positive, reproduces the
            parameters) or ``"linear"`` (plain orthonormal expansion).
        sigma: ``"mem"`` only: standard deviation of the noise on the parameters
            (e.g. parameters predicted by a model). ``None`` (default) imposes
            the parameters exactly; otherwise the constraints are relaxed with a
            Tikhonov term ``10 * sigma`` (calibrated on noisy tests).
        max_iter: ``"mem"`` only: maximum number of Newton iterations.
        tol: ``"mem"`` only: tolerance on the constraint residual, relative to
            the largest parameter.

    Returns:
        The reconstructed density on ``grid`` (same dims and coordinates), in
        the units of :func:`cwaves.compute_cwaves` (``|f|`` divided by its
        annulus mass); zero outside the kernel domain. Attributes ``method``
        and, for ``"mem"``, ``constraint_residual`` and ``iterations``.

    Raises:
        ValueError: on an unknown ``method``, a ``"mem"`` reconstruction of
            signed parameters (``part`` other than ``"modulus"``), or parameters
            with extra dimensions.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    part = cwave_params.attrs.get("part", "modulus")
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r} in the CWAVE parameter attributes")
    if method == "mem" and part != "modulus":
        raise ValueError(
            "the maximum-entropy reconstruction needs a positive density: only "
            f"part='modulus' parameters are supported (got part={part!r}); use "
            "method='linear'"
        )
    if set(cwave_params.dims) != {"k_gp", "phi_hf"}:
        raise ValueError(
            "reconstruct one parameter set at a time: dims must be (k_gp, phi_hf), "
            f"got {cwave_params.dims}"
        )
    nk, nphi = int(cwave_params.sizes["k_gp"]), int(cwave_params.sizes["phi_hf"])
    kmin, kmax = _cwave_bounds(
        cwave_params.attrs.get("cwave_wlmax", 600),
        cwave_params.attrs.get("cwave_wlmin", 25),
    )
    coeffs = np.asarray(cwave_params.transpose("k_gp", "phi_hf")).ravel()

    def solve(c: np.ndarray, parity: str) -> tuple[np.ndarray, Any, dict[str, Any]]:
        kernel, weight, domain, quad = _operator(grid, kmin, kmax, nk, nphi, parity)
        # even kernels on a full plane have norm sqrt(2) (both halves counted)
        scale = 0.5 if quad.coverage == "full" else 1.0
        info: dict[str, Any] = {}
        if method == "linear":
            dens = scale * (c @ kernel)
        else:
            linear = np.clip(scale * (c @ kernel), 0.0, None)
            level = (linear * weight).sum() / weight.sum()
            prior = np.full(weight.shape, level if level > 0 else 1.0 / weight.sum())
            noise = 0.0 if sigma is None else 10.0 * float(sigma)
            atol = tol * max(float(np.abs(c).max()), np.finfo(float).tiny)
            dens, residual, iters = _mem_dual(
                kernel, weight, c, prior, noise, max_iter, atol
            )
            info = {"constraint_residual": residual, "iterations": iters}
            if sigma is None and residual > 1e3 * atol:
                logger.warning(
                    "maximum-entropy reconstruction did not converge "
                    "(constraint residual %.1e after %d iterations)",
                    residual,
                    iters,
                )
        full = np.zeros(domain.size)
        full[domain] = dens
        return full, quad, info

    if part == "complex":
        re, quad, info = solve(coeffs.real, "even")
        im, _, _ = solve(coeffs.imag, "odd")
        values: np.ndarray = re + 1j * im
    else:
        values, quad, info = solve(np.real(coeffs), "odd" if part == "imag" else "even")
    shape = [grid.sizes[d] for d in quad.dims]
    out = xr.DataArray(
        values.reshape(shape), dims=quad.dims, coords=grid.coords
    ).transpose(*grid.dims)
    out.attrs.update(
        {
            "long_name": "spectrum reconstructed from CWAVE parameters",
            "method": method,
            "part": part,
            "cwave_wlmax": cwave_params.attrs.get("cwave_wlmax", 600),
            "cwave_wlmin": cwave_params.attrs.get("cwave_wlmin", 25),
            **info,
        }
    )
    return out.rename("cwave_reconstruction")
