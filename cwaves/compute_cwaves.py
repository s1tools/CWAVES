"""CWAVE parameter computation from SAR ocean cross-spectra.

Implements the orthonormal kernel decomposition of the reference :

    J. Schulz-Stellenfleth, T. Konig, and S. Lehner,
    "An empirical approach for the retrieval of integral ocean wave
    parameters from synthetic aperture radar data",
    J. Geophys. Res., 112, C03019, 2007.
    doi:10.1029/2006JC003970

The kernels are orthonormal on the elliptic wavenumber domain A
(equations B1 to B13 of the paper).

Mathematical basis : the kernel construction is the standard orthonormal
Gegenbauer theory of

    Gradshteyn, I. S., and Ryzhik, I. M. (2000),
    "Table of Integrals, Series and Products", 7th ed., Elsevier, New York
    (section 7.313 : orthonormality of C_n^(3/2) with the (1 - x^2) weight),

applied to the elliptic wavenumber domain A of the 2007 paper. This module
is the *exact* implementation of that construction :

* the Gegenbauer polynomials C_n^(3/2) are orthonormalized with
  ``coef(n) = sqrt((n + 3/2) / ((n + 2) (n + 1)))``, which makes
  ``coef(n)^2 * I_n = 1`` hold **exactly for every n**, where
  ``I_n = integral_{-1}^{1} (1 - x^2) C_n^(3/2)(x)^2 dx`` is the
  Gradshteyn-Ryzhik orthonormality integral (``I_0 = 4/3`` and
  ``I_n = 2 (n + 1) (n + 2) / (2 n + 3)`` for n >= 1);
* ``eta`` (paper equation B13) is identically the square root of the
  inverse Jacobian of the (alpha, phi) -> (k_rg, k_az) mapping
  (``eta^2 * J = 1``), so the 2-D kernel orthonormality over A reduces to
  the 1-D orthonormalities above and ``integral_A K_i K_j dA = delta_ij``
  holds **exactly in the continuum**.

See ``repository_specifications.md`` (sections 5 and 7) for the full
verification (polynomials against scipy, integrals against the analytic
G&R values, Jacobian identity -- all at machine precision).
"""

import copy

import numpy as np
import xarray as xr

from cwaves.grid import build_quadrature, coordinate_extra_dims


def _cwave_bounds(cwave_wlmax, cwave_wlmin):
    """Return ``(kmin, kmax)`` [rad/m] from the wavelength bounds [m]."""
    if not (cwave_wlmax > cwave_wlmin > 0):
        raise ValueError(
            f"need cwave_wlmax > cwave_wlmin > 0, got {cwave_wlmax} / {cwave_wlmin}"
        )
    return 2.0 * np.pi / cwave_wlmax, 2.0 * np.pi / cwave_wlmin


#: what part of the spectrum is projected (see compute_cwaves)
PARTS = ("modulus", "real", "imag", "complex")


def compute_cwaves(
    spectrum,
    cwave_wlmax=600,
    cwave_wlmin=25,
    cwave_Nk=4,
    cwave_Nphi=5,
    part="modulus",
    **kwargs,
):
    """Compute the CWAVE parameters of a spectrum sampled on any supported grid.

    This is the **single entry point** of the library. It evaluates the
    projections (paper equations 25-26)

    .. math::

        c_{n,p} = \\int_A K_{n,p}(\\mathbf{k})\\, \\hat f(\\mathbf{k})\\,
        \\mathrm{d}A, \\qquad
        \\hat f = |f| \\Big/ \\int_{k_{min}<|\\mathbf{k}|<k_{max}} |f|\\,
        \\mathrm{d}A

    with the quadrature rule of the input grid (:mod:`cwaves.grid`): no
    resampling of the spectrum is performed. Supported inputs:

    * **cartesian** grid, coordinates ``k_rg`` / ``k_az`` (rad/m), uniform or
      not (FFT grids, sinh grids...);
    * **polar** grid, coordinates ``k`` (rad/m) / ``phi`` (radians,
      ``k_rg = k cos(phi)``), linear or log-spaced ``k``;
    * either the **half plane** ``k_rg >= 0`` (the paper's domain, on which
      the kernel basis is orthonormal) or the **full plane**. On the full plane
      the result is exactly the CWAVE parameters of the centrally-symmetric
      part ``(f(k) + f(-k)) / 2`` of the spectrum (the kernels are even, so
      the anti-symmetric part does not contribute) -- see
      ``repository_specifications.md`` section 6.

    The projected quantity is chosen with ``part`` (the density is always
    normalised by the mass of the **modulus** ``|f|`` in the annulus, so the
    four options share the same scale):

    * ``"modulus"`` (default, the 2007 paper): ``|f|`` on the **even**
      (pi-periodic) harmonics -- the complex phase is discarded;
    * ``"real"``: ``Re f`` on the even harmonics;
    * ``"imag"``: ``Im f`` on the **odd** (pi-antiperiodic) harmonics
      (:func:`harmonic_functions` with ``parity="odd"``);
    * ``"complex"``: ``c = c_even[Re f] + 1j * c_odd[Im f]``.

    The real part of any SAR cross-spectrum (any lag, 0/1/2-tau) is
    centrally symmetric and its imaginary part anti-symmetric (Hermitian
    symmetry of the Fourier transform of a real image): the even harmonics are
    the natural basis of the former, the odd harmonics of the latter. On a
    half-plane input this Hermitian structure is *assumed*; on a full-plane
    input the even (odd) harmonics project the centrally symmetric
    (anti-symmetric) part of whatever is given. Using the even harmonics on
    ``Im f`` would converge very slowly (Gibbs effect at the ``k_rg = 0``
    edge, where ``Im f`` changes sign). Any extra dimension (``pol``,
    ``time``, ``tile``...) is broadcast.

    Args:
        spectrum (xarray.DataArray): real or complex spectrum with its
            wavenumber coordinates.
        cwave_wlmax (float, optional): maximum wavelength [m], sets
            ``kmin = 2 pi / cwave_wlmax``.
        cwave_wlmin (float, optional): minimum wavelength [m], sets
            ``kmax = 2 pi / cwave_wlmin``.
        cwave_Nk (int, optional): number of Gegenbauer (radial) functions.
        cwave_Nphi (int, optional): number of harmonic (angular) functions.
        part (str, optional): ``"modulus"`` (default), ``"real"``, ``"imag"``
            or ``"complex"`` (see above).

    Keyword Args:
        save_cwave_kernel (bool, optional): save the kernel to
            ``cwaves_kernel.nc`` in the working directory.

    Returns:
        xarray.DataArray: the CWAVE parameters ``cwave_params``, dims
        ``(k_gp, phi_hf)`` plus any extra input dimension. Attributes record
        the detected grid (``grid``) and plane coverage (``coverage``).

    Raises:
        TypeError / ValueError: unsupported or malformed grid (see
            :func:`cwaves.grid.build_quadrature`), or a spectrum with zero
            mass in the wavenumber annulus.
    """
    kmin, kmax = _cwave_bounds(cwave_wlmax, cwave_wlmin)
    cwave = {
        "cwave_wlmax": cwave_wlmax,
        "cwave_wlmin": cwave_wlmin,
        "cwave_Nk": cwave_Nk,
        "cwave_Nphi": cwave_Nphi,
        "part": part,
    }
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    if isinstance(spectrum, xr.DataArray):
        extra = coordinate_extra_dims(spectrum)
        if extra:
            # one wavenumber grid per tile / time: process slice by slice
            stacked = spectrum.stack(_cwave_slice=extra)
            index_names = list(stacked.indexes["_cwave_slice"].names)
            parts = [
                compute_cwaves(
                    stacked.isel(_cwave_slice=i).drop_vars(
                        ["_cwave_slice", *index_names], errors="ignore"
                    ),
                    **cwave,
                    **kwargs,
                )
                for i in range(stacked.sizes["_cwave_slice"])
            ]
            out = xr.concat(
                parts, dim="_cwave_slice", coords="minimal", compat="override"
            )
            out = out.assign_coords(
                xr.Coordinates.from_pandas_multiindex(
                    stacked.indexes["_cwave_slice"], "_cwave_slice"
                )
            ).unstack("_cwave_slice")
            out.attrs.update(parts[0].attrs)
            return out.rename("cwave_params")
    quad = build_quadrature(spectrum)
    dims = list(quad.dims)

    # wavenumber annulus kmin < |k| < kmax (circular), used for the
    # normalisation mass; the kernel itself is NaN outside the elliptic domain A
    in_annulus = (quad.k_norm > kmin) & (quad.k_norm < kmax)
    save = kwargs.pop("save_cwave_kernel", False)
    amplitude = np.abs(spectrum).where(in_annulus)
    mass = (amplitude * quad.weight).sum(dim=dims, skipna=True)
    if not bool(np.all(np.isfinite(mass) & (mass > 0))):
        raise ValueError(
            "the spectrum has zero (or non-finite) mass in the wavenumber annulus "
            f"{kmin:.4g} < |k| < {kmax:.4g} rad/m: nothing to normalise. Check "
            "cwave_wlmax / cwave_wlmin, the wavenumber units (rad/m) and the input."
        )

    def project(field, parity):
        with np.errstate(invalid="ignore", divide="ignore"):
            kernel = compute_kernel(
                krg=quad.k_rg.where(in_annulus),
                kaz=quad.k_az.where(in_annulus),
                kmin=kmin,
                kmax=kmax,
                cwave_Nk=cwave_Nk,
                cwave_Nphi=cwave_Nphi,
                save_cwave_kernel=save,
                parity=parity,
            )
        density = field.where(in_annulus) / mass
        return (kernel.cwave_kernel * density * quad.weight).sum(
            dim=dims, skipna=True, min_count=1
        )

    if part == "modulus":
        cwave_parameters = project(amplitude, "even")
    elif part == "real":
        cwave_parameters = project(np.real(spectrum), "even")
    elif part == "imag":
        cwave_parameters = project(np.imag(spectrum), "odd")
    else:
        cwave_parameters = project(np.real(spectrum), "even") + 1j * project(
            np.imag(spectrum), "odd"
        )

    name = str(spectrum.name) if spectrum.name is not None else ""
    loc = name.find("tau")
    tau = name[loc - 1 : loc + 3] if loc != -1 else None
    cwave_parameters.attrs.update(
        {
            "long_name": "CWAVE parameters" + (" (" + tau + ")" if tau else ""),
            "grid": quad.kind,
            "coverage": quad.coverage,
            "cwave_wlmax": cwave_wlmax,
            "cwave_wlmin": cwave_wlmin,
            "part": part,
        }
    )
    return cwave_parameters.rename("cwave_params")


# copy from xsarslc commit https://github.com/umr-lops/xsar_slc/commit/14f80b4663d82efe40d2243b65f02a835500d3d7
def compute_cwave_parameters(
    xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs
):
    """Compute CWAVE parameters (``xsar_slc``-compatible signature).

    Kept with the historical ``xsar_slc`` signature for API conservativeness
    (``repository_specifications.md`` section 3); it simply delegates to
    :func:`compute_cwaves`, which accepts every supported grid. New code should
    call :func:`compute_cwaves` directly.

    Args:
        xs (xarray.DataArray) : real or complex spectrum (see :func:`compute_cwaves`)
        cwave_wlmax (float, optional) : maximum wavelength [m]
        cwave_wlmin (float, optional) : minimum wavelength [m]
        cwave_Nk (int, optional) : number of Gegenbauer functions
        cwave_Nphi (int, optional) : number of harmonic functions
    Keywords Args:
        save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory

    Return:
        (xarray.DataArray) : cwave_parameters
    """
    return compute_cwaves(
        xs,
        cwave_wlmax=cwave_wlmax,
        cwave_wlmin=cwave_wlmin,
        cwave_Nk=cwave_Nk,
        cwave_Nphi=cwave_Nphi,
        **kwargs,
    )


def compute_kernel(
    krg,
    kaz,
    kmin,
    kmax,
    cwave_Nk=4,
    cwave_Nphi=5,
    save_cwave_kernel=False,
    parity="even",
):
    """
    Compute CWAVE kernels
    Args:
        krg (xarray.DataArray) : spectrum wavenumbers in range direction
        kaz (xarray.DataArray) : spectrum wavenumbers in azimuth direction

    Keywords Args:
        cwave_Nk (int) :
        cwave_Nphi (int) :
        save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory
        parity (str, optional) : angular parity, ``"even"`` (paper, default) or ``"odd"`` (see :func:`harmonic_functions`)

    Return:
        (xarray.Dataset) : kernels
    """
    # Kernel Computation
    #

    def coef(nk):
        # orthonormalization coefficient of the Gegenbauer polynomials C_n^(3/2)
        # on [-1, 1] with weight (1 - x^2)  (paper equation B9)
        return np.sqrt((nk + 3 / 2.0) / ((nk + 2.0) * (nk + 1.0)))

    def nu(x):
        return np.sqrt(1 - x**2.0)

    gamma = 2
    a1 = (gamma**2 - np.power(gamma, 4)) / (gamma**2 * kmin**2 - kmax**2)
    a2 = (kmax**2 - np.power(gamma, 4) * kmin**2) / (kmax**2 - gamma**2 * kmin**2)
    tmp = a1 * np.power(kaz, 4) + a2 * kaz**2 + krg**2
    # alpha k
    alpha_k = (
        2
        * (
            (np.log10(np.sqrt(tmp)) - np.log10(kmin))
            / (np.log10(kmax) - np.log10(kmin))
        )
        - 1
    )
    # alpha phi
    alpha_phi = np.arctan2(krg, kaz).rename(None)
    # eta : Jacobian of the wavenumber mapping (paper equation B13)
    # natural logarithm ; the numerator is 2 * S with
    # S = 2 a1 k_az^4 + a2 k_az^2 + k_rg^2 (not 2 * tmp)
    eta = np.sqrt(
        (2.0 * (2 * a1 * np.power(kaz, 4) + a2 * kaz**2 + krg**2))
        / ((kaz**2 + krg**2) * tmp * np.log(kmax / kmin))
    )

    Gnk = xr.combine_by_coords(
        [
            gegenbauer_polynoms(alpha_k, ik - 1, lbda=3 / 2.0)
            * coef(ik - 1)
            * nu(alpha_k).assign_coords({"k_gp": ik}).expand_dims("k_gp")
            for ik in np.arange(cwave_Nk) + 1
        ]
    )
    Fnphi = xr.combine_by_coords(
        [
            harmonic_functions(alpha_phi, iphi, parity=parity)
            .assign_coords({"phi_hf": iphi})
            .expand_dims("phi_hf")
            for iphi in np.arange(cwave_Nphi) + 1
        ]
    )

    Kernel = Gnk * Fnphi * eta
    Kernel.k_gp.attrs.update({"long_name": "Gegenbauer polynoms dimension"})
    Kernel.phi_hf.attrs.update(
        {"long_name": "Harmonic functions dimension (odd number)"}
    )

    _Kernel = Kernel.rename("cwave_kernel").to_dataset()
    if "pol" in _Kernel:
        _Kernel = _Kernel.drop_vars("pol")
    _Kernel["cwave_kernel"].attrs.update({"long_name": "CWAVE Kernel"})

    # NB: Gnk / Fnphi are DataArrays at runtime (xarray stubs type
    # combine_by_coords as returning a Dataset), so rename(str) is valid.
    ds_G = Gnk.rename("Gegenbauer_polynoms").to_dataset()  # type: ignore[arg-type]
    ds_F = Fnphi.rename("Harmonic_functions").to_dataset()  # type: ignore[arg-type]
    ds_eta = eta.rename("eta").to_dataset()
    if "pol" in ds_G:
        ds_G = ds_G.drop_vars("pol")
        ds_F = ds_F.drop_vars("pol")
        ds_eta = ds_eta.drop_vars("pol")

    Kernel = xr.merge([_Kernel, ds_G, ds_F, ds_eta])

    if save_cwave_kernel:
        Kernel.to_netcdf("cwaves_kernel.nc")

    return Kernel


# old version of gegenbauer polynoms (not working properly)
# def gegenbauer_polynoms(x, nk, lbda=3 / 2.):
#     """

#     Args:
#         x: np.ndarray
#         nk: int
#         lbda: float

#     Returns:
#         Cnk : np.ndarray
#     """
#     C0 = 1
#     if (nk == 0):
#         return C0 + x * 0
#     C1 = 3 * x
#     if (nk == 1):
#         return C1 + x * 0

#     Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
#             nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))
#     Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
#             nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))

#     return Cnk


def gegenbauer_polynoms(x, nk, lbda=1.5):
    """

    Compute Gegenbauer polynoms using iterative relation
    2*x*(n+lambda-1)*C(n-1) - (n+2*lambda-2)*C(n-2) = n*C(n)


    Args:
        x: xr.DataArray
        nk: int
        lbda: float [default 1.5]

    Returns:
        Cn : xr.DataArray
    """
    xcopy = copy.copy(x)
    x = np.asarray(x)
    if nk == 0:
        return np.ones_like(x)
    if nk == 1:
        return 2 * lbda * x

    Cnm2 = np.ones_like(x)
    Cnm1 = 2 * lbda * x
    for n in range(2, nk + 1):
        Cn = (1.0 / n) * (2 * x * (n + lbda - 1) * Cnm1 - (n + 2 * lbda - 2) * Cnm2)
        Cnm2, Cnm1 = Cnm1, Cn
    Cn = xr.DataArray(Cn, coords=xcopy.coords, dims=xcopy.dims)
    return Cn


#: angular parities of the harmonic basis
PARITIES = ("even", "odd")


def harmonic_functions(x, nphi, parity="even"):
    """Orthonormal angular functions on any interval of length pi.

    ``parity="even"`` (the 2007 paper, default) gives the pi-**periodic** set
    ``{1/sqrt(pi), sqrt(2/pi) sin(2 phi), sqrt(2/pi) cos(2 phi), sin(4 phi),
    cos(4 phi), ...}`` (``nphi`` = 1, 2, 3, ...): kernels built with it are
    even under ``k -> -k``.

    ``parity="odd"`` gives the pi-**antiperiodic** set ``{sqrt(2/pi)
    cos(phi), sqrt(2/pi) sin(phi), cos(3 phi), sin(3 phi), ...}``: kernels
    built with it are odd under ``k -> -k`` (they represent anti-symmetric
    content, e.g. the imaginary part of a Hermitian cross-spectrum).

    Both sets are orthonormal **and complete** on ``[0, pi]`` (the angular
    range of the half plane ``k_rg >= 0``); together they form the full
    Fourier basis of the circle.

    Args:
        x: angle(s) in radians (np.ndarray or xarray.DataArray).
        nphi: 1-based index of the function.
        parity: ``"even"`` (default) or ``"odd"``.

    Returns:
        Fn : the function evaluated at ``x`` (same type as ``x``).
    """
    if parity == "odd":
        m = 2 * ((nphi + 1) // 2) - 1
        if nphi % 2 == 1:
            return np.sqrt(2 / np.pi) * np.cos(m * x)
        return np.sqrt(2 / np.pi) * np.sin(m * x)
    if parity != "even":
        raise ValueError(f"parity must be one of {PARITIES}, got {parity!r}")
    if nphi == 1:
        Fn = np.sqrt(1 / np.pi) + x * 0
        return Fn

    # Even & Odd case
    if nphi % 2 == 0:
        Fn = np.sqrt(2 / np.pi) * np.sin((nphi) * x)
    else:
        Fn = np.sqrt(2 / np.pi) * np.cos((nphi - 1) * x)

    return Fn
