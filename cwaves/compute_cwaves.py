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


def _check_one_sided(xs):
    """Raise if ``xs`` is not a one-sided (positive range) cross-spectrum.

    The CWAVE parameters are computed directly on the one-sided (half) range
    domain, as in Schultz-Stellenfleth et al. (2007) (see
    ``repository_specifications.md`` section 6). A full / symmetrized spectrum
    (with negative range wavenumbers) must not be passed. The one-sided
    domain is a **requirement, not a convenience** :

    * on a full / zero-centered range grid the *same* kernels are only
      "2-normalized" : their Gram matrix converges to 2I (norm sqrt(2),
      not 1), so a signal built from the basis round-trips to 2x itself
      (100 % error, no convergence under refinement) -- verified in
      ``repository_specifications.md`` section 6.3;
    * the angular basis {1, sin 2p(phi), cos 2p(phi)} is pi-periodic :
      orthonormal on any interval of length pi (the one-sided domain maps
      phi to [0, pi] exactly), but **incomplete** on the full 2pi circle --
      it spans only the centrally-symmetric spectra, and a generic
      full-domain spectrum projects to ~0 (same section).
    """
    krg_min = float(np.min(np.asarray(xs.k_rg)))
    if krg_min < 0:
        raise ValueError(
            "compute_cwave_parameters expects a one-sided (k_rg >= 0) cross-spectrum. "
            f"A full / symmetrized spectrum was detected (k_rg.min() = {krg_min:.3e}). "
            "The one-sided (half) range domain is where the kernel basis is orthonormal; "
            "on a zero-centered grid the Gram matrix converges to 2I (norm sqrt(2), not 1) "
            "and the pi-periodic angular basis is incomplete for generic (non centrally-"
            "symmetric) spectra. Compute the CWAVE parameters directly on the one-sided "
            "range domain, as in Schultz-Stellenfleth et al. (2007); do not symmetrize "
            "first (see repository_specifications.md sections 6 and 7)."
        )


# copy from xsarslc commit https://github.com/umr-lops/xsar_slc/commit/14f80b4663d82efe40d2243b65f02a835500d3d7
def compute_cwave_parameters(
    xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs
):
    """
    Compute CWAVE parameters.

    Input contract (fully compliant with the 2007 paper, see
    ``repository_specifications.md`` section 6):

        - ``xs`` must be a **one-sided (half)** cross-spectrum : positive
          wavenumbers only in range (``k_rg >= 0``), two-sided in azimuth
          (``k_az``). A **full / symmetrized** (two-sided range) spectrum is
          rejected with a ``ValueError``.
        - Only the **real, non-negative modulus** ``|xs|`` of the spectrum is
          used : the complex phase is discarded, matching the paper's SAR image
          variance spectrum (periodogram). The computation is performed directly
          on the one-sided range domain (no symmetrization), as in
          Schultz-Stellenfleth et al. (2007) equations 25-26 and B1.
        - ``xs`` must be sampled on a **uniform (regular) wavenumber grid** in both
          ``k_rg`` (range) and ``k_az`` (azimuth) : the projection integral uses a
          single constant cell area (the mean spacing), which is exact only for a
          uniform grid (see ``repository_specifications.md`` section 6.6).

    Args:
        xs (xarray.Dataset) : complex ONE-SIDED xspectra (``k_rg >= 0``)
        cwave_wlmax (float, optional) : maximum wavelength
        cwave_wlmin (float, optional) : minimum wavelength
        cwave_Nk (int, optional) :
        cwave_Nphi (int, optional) :
    Keywords Args:
        kwargs (): keyword arguments passed to compute_kernel
            save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory

    Return:
        (xarray.DataArray) : cwave_parameters
    """
    kmin = 2.0 * np.pi / cwave_wlmax
    kmax = 2.0 * np.pi / cwave_wlmin
    _check_one_sided(xs)
    loc = xs.name.find("tau")
    tau = xs.name[loc - 1 : loc + 3] if loc != -1 else None

    # Cross-Spectra Frequency Filtering
    kk = np.sqrt((xs.k_rg) ** 2.0 + (xs.k_az) ** 2.0)
    xxs = xs.where(
        (kk > kmin) & (kk < kmax)
    )  # drop=True here would remove other dimensions we may want to keep
    xxs = xxs.dropna(dim="freq_line", how="all").dropna(
        dim="freq_sample", how="all"
    )  # removing only along the chosen dimensions

    # Cross-Spectra normalization
    xxsm = np.abs(xxs)
    dkrg = xxs.k_rg.diff(dim="freq_sample").mean(
        dim="freq_sample"
    )  # works even if there are other dimensions than expected
    dkaz = xxs.k_az.diff(dim="freq_line").mean(
        dim="freq_line"
    )  # works even if there are other dimensions than expected
    xxsmn = xxsm / (xxsm.sum(dim=["freq_line", "freq_sample"]) * dkrg * dkaz)

    # XS decomposition Kernel

    kernel = compute_kernel(
        krg=xxs.k_rg,
        kaz=xxs.k_az,
        kmin=kmin,
        kmax=kmax,
        cwave_Nk=cwave_Nk,
        cwave_Nphi=cwave_Nphi,
        save_cwave_kernel=kwargs.pop("save_cwave_kernel", False),
    )
    # CWAVE parameters computation
    cwave_parameters = ((kernel.cwave_kernel * xxsmn) * dkrg * dkaz).sum(
        dim=["freq_sample", "freq_line"], skipna=True, min_count=1
    )
    cwave_parameters.attrs.update(
        {"long_name": "CWAVE parameters" + (" (" + tau + ")" if tau else "")}
    )
    return cwave_parameters.rename("cwave_params")  # .to_dataset()


def _polar_k_weights(k):
    """Per-bin radial cell weights ``w_i`` for a polar ``k dk`` quadrature.

    ``sum_i w_i * f(k_i)`` approximates ``integral f(k) k dk``. The cell
    boundaries are the geometric means of adjacent sample values, which is
    exact for a log-spaced grid (where the samples are the cell centres in
    log-space); the two outer boundaries are extrapolated by half of the
    adjacent log step.
    """
    lnk = np.log(k)
    lnb = 0.5 * (lnk[1:] + lnk[:-1])
    lnb = np.concatenate(
        [
            [lnk[0] - 0.5 * (lnk[1] - lnk[0])],
            lnb,
            [lnk[-1] + 0.5 * (lnk[-1] - lnk[-2])],
        ]
    )
    return 0.5 * (np.exp(2.0 * lnb[1:]) - np.exp(2.0 * lnb[:-1]))


def _polar_phi_weights(phi):
    """Per-bin angular cell widths (periodic, full circle) for a ``dphi`` rule.

    The width of bin ``i`` is the distance to the midpoints of its two
    neighbours, with the :math:`2\\pi` wrap-around for the first and last bins.
    For a uniform grid this reduces to the constant :math:`2\\pi/n` for every
    bin.
    """
    n = phi.size
    if n == 1:
        return np.array([2.0 * np.pi])
    d = np.empty(n)
    d[:-1] = np.diff(phi)
    d[-1] = phi[0] + 2.0 * np.pi - phi[-1]
    return 0.5 * (d + np.roll(d, 1))


def compute_cwave_parameters_polar(
    spec, k, phi, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs
):
    """Compute the CWAVE parameters directly on a polar ``(k, phi)`` grid.

    This is the polar-grid counterpart of :func:`compute_cwave_parameters`.
    It integrates the same CWAVE projection
    :math:`P_{n,m} = \\int_A K_{n,m}\\, |X_S|_{\\rm norm}\\, \\mathrm{d}A`
    but **on the native polar wavenumber grid** (e.g. the ESA OCN WV product's
    ``oswPolSpec``, a real polar spectrum on a log-spaced ``k`` x uniform
    ``phi`` grid) instead of first resampling the spectrum onto a cartesian
    (``k_rg``, ``k_az``) grid. The area element is the polar one,
    :math:`\\mathrm{d}A = k\\, \\mathrm{d}k\\, \\mathrm{d}\\varphi`, so each
    cell weight is the exact polar cell area :math:`k_i\\, \\Delta k_i\\,
    \\Delta\\varphi_j` (see :func:`_polar_k_weights` and
    :func:`_polar_phi_weights`). This avoids the resampling step of
    :func:`cwaves.client.polar_to_cartesian` and is exact for the product's own
    (log-)polar sampling.

    The rest of the input contract is identical to
    :func:`compute_cwave_parameters` (``repository_specifications.md`` section
    6): the real, non-negative modulus :math:`|spec|` is used (the complex
    phase is discarded), the integration is over the elliptic wavenumber domain
    ``A`` (paper equations B1-B13) intersected with the wavenumber annulus
    :math:`k_{min} < k < k_{max}` and restricted to the **one-sided (half) range
    domain** :math:`k_x = k\\cos\\varphi \\ge 0` (in polar coordinates the
    circular annulus :math:`k_{min}<\\sqrt{k_x^2+k_y^2}<k_{max}` is exactly
    :math:`k_{min}<k<k_{max}`, and the one-sided half is :math:`\\cos\\varphi
    \\ge 0`). The normalized spectrum has unit mass over that one-sided
    annulus, as in the cartesian path.

    Args:
        spec: the (real or complex) polar cross-spectrum, 2-D with shape
            ``(len(phi), len(k))`` (first index = angle, second = wavenumber
            magnitude), matching the OCN product's ``oswPolSpec`` layout. Only
            its modulus is used.
        k: 1-D array of wavenumber magnitudes (rad/m), strictly positive and
            strictly increasing. A log-spaced grid is expected (OCN product).
        phi: 1-D array of angles (rad), strictly increasing, covering the full
            :math:`2\\pi` circle. The OCN product's ``oswPhi = 0`` is the
            **range** direction (``k_x = k cos(phi)``).
        cwave_wlmax (float, optional) : maximum wavelength [m] (sets kmin).
        cwave_wlmin (float, optional) : minimum wavelength [m] (sets kmax).
        cwave_Nk (int, optional) : number of Gegenbauer basis functions.
        cwave_Nphi (int, optional) : number of harmonic basis functions.
    Keywords Args:
        kwargs (): keyword arguments passed to :func:`compute_kernel`
            (e.g. ``save_cwave_kernel``).

    Returns:
        (xarray.DataArray) : the CWAVE parameters, dims ``(k_gp, phi_hf)``,
        named ``cwave_params``.

    Raises:
        ValueError: if the input grids are malformed (non-positive / non-
            increasing ``k``, ``phi`` not spanning the full circle, shape
            mismatch) or if the (masked) spectrum carries zero mass.
    """
    spec = np.asarray(spec)
    k = np.asarray(k, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if spec.ndim != 2:
        raise ValueError(f"spec must be 2-D (phi, k), got shape {spec.shape}")
    if k.ndim != 1 or phi.ndim != 1:
        raise ValueError("k and phi must be 1-D arrays")
    if k.size < 2 or phi.size < 2:
        raise ValueError("k and phi must each have at least 2 samples")
    if spec.shape != (phi.size, k.size):
        raise ValueError(
            f"spec shape {spec.shape} does not match (len(phi)={phi.size}, "
            f"len(k)={k.size})"
        )
    if np.any(k <= 0.0):
        raise ValueError("k must be strictly positive (rad/m)")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("k must be strictly increasing")
    if np.any(np.diff(phi) <= 0.0):
        raise ValueError("phi must be strictly increasing")
    if phi[-1] - phi[0] >= 2.0 * np.pi:
        raise ValueError("phi must span (but not exceed) one full 2*pi circle")

    kmin = 2.0 * np.pi / cwave_wlmax
    kmax = 2.0 * np.pi / cwave_wlmin

    # real, non-negative modulus (complex phase discarded), as in
    # compute_cwave_parameters
    S = np.abs(spec)

    # one-sided (half) range domain: kx = k cos(phi) >= 0  ->  cos(phi) >= 0.
    # The tiny tolerance keeps the kx = 0 boundary cells (phi = 90 / 270 deg),
    # which the cartesian path also integrates (its k_rg = 0 column).
    one_sided = np.cos(phi) >= -1.0e-6
    # circular annulus kmin < sqrt(kx^2 + ky^2) < kmax  ==  kmin < k < kmax
    # in polar coordinates (kk = k)
    ann = (k > kmin) & (k < kmax)

    # per-cell polar area weights: dA = k dk dphi
    w = _polar_phi_weights(phi)[:, None] * _polar_k_weights(k)[None, :]

    # unit-mass normalisation over the one-sided annulus (regions outside the
    # elliptic domain A are included in the mass, exactly as in the cartesian
    # path, where the kernel's NaN is only applied at the final sum)
    mask = one_sided[:, None] & ann[None, :] & ~np.isnan(S)
    mass = float(np.nansum(np.where(mask, S * w, np.nan)))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            "the polar spectrum has zero (or non-finite) mass on the one-sided "
            "annulus kmin < k < kmax, kx >= 0: nothing to normalise. Check "
            "cwave_wlmax / cwave_wlmin and the input spectrum."
        )
    Snorm = np.where(mask, S * w / mass, np.nan)

    # kernel evaluated at kx = k cos(phi), ky = k sin(phi); compute_kernel
    # returns NaN outside the elliptic domain A (via sqrt(1 - alpha_k^2)),
    # which the skipna sum below excludes from the integral
    kx = k[None, :] * np.cos(phi)[:, None]
    ky = k[None, :] * np.sin(phi)[:, None]
    krg = xr.DataArray(kx, dims=("phi", "k"), coords={"phi": phi, "k": k})
    kaz = xr.DataArray(ky, dims=("phi", "k"), coords={"phi": phi, "k": k})
    with np.errstate(invalid="ignore"):
        kernel = compute_kernel(
            krg=krg,
            kaz=kaz,
            kmin=kmin,
            kmax=kmax,
            cwave_Nk=cwave_Nk,
            cwave_Nphi=cwave_Nphi,
            save_cwave_kernel=kwargs.pop("save_cwave_kernel", False),
        )
    cwave_parameters = (
        kernel.cwave_kernel * xr.DataArray(Snorm, dims=("phi", "k"))
    ).sum(dim=("phi", "k"), skipna=True, min_count=1)
    cwave_parameters.attrs.update({"long_name": "CWAVE parameters (polar grid)"})
    return cwave_parameters.rename("cwave_params")


def compute_kernel(
    krg, kaz, kmin, kmax, cwave_Nk=4, cwave_Nphi=5, save_cwave_kernel=False
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
            harmonic_functions(alpha_phi, iphi)
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


def harmonic_functions(x, nphi):
    """

    Args:
        x: np.ndarray
        nphi: int

    Returns:
        Fn : np.ndarray
    """
    if nphi == 1:
        Fn = np.sqrt(1 / np.pi) + x * 0
        return Fn

    # Even & Odd case
    if nphi % 2 == 0:
        Fn = np.sqrt(2 / np.pi) * np.sin((nphi) * x)
    else:
        Fn = np.sqrt(2 / np.pi) * np.cos((nphi - 1) * x)

    return Fn
