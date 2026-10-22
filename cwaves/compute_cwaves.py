"""CWAVE parameter computation from SAR ocean cross-spectra.

Implements the orthonormal kernel decomposition of the reference :

    J. Schulz-Stellenfleth, T. Konig, and S. Lehner,
    "An empirical approach for the retrieval of integral ocean wave
    parameters from synthetic aperture radar data",
    J. Geophys. Res., 112, C03019, 2007.
    doi:10.1029/2006JC003970

The kernels are orthonormal on the elliptic wavenumber domain A
(equations B1 to B13 of the paper).
"""

import copy

import numpy as np
import xarray as xr

from cwaves.symmetrize_xspectra import symmetrize_xspectrum


# copy from xsarslc commit https://github.com/umr-lops/xsar_slc/commit/14f80b4663d82efe40d2243b65f02a835500d3d7
def compute_cwave_parameters(
    xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs
):
    """
    Compute CWAVE parameters.
    Args:
        xs (xarray.Dataset) : complex ONE-SIDED xspectra
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
    xs = symmetrize_xspectrum(xs)
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
