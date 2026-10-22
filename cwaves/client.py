"""CLI client to compute CWAVE parameters from Sentinel-1 SARWave products.

Two input product types are supported (roadmap task 9). Both feed the
one-sided (positive range) / two-sided (azimuth) complex cross-spectrum to
:func:`cwaves.compute_cwave_parameters`, which symmetrizes internally. The
``kx = range``, ``ky = azimuth`` convention is the same as the latest
``xsar_slc`` commits.

* **L1B XSP cross-spectrum** products (``S1A_WV1_XSP__...nc``) : self-contained
  cross-spectra ``xspectra_{lag}tau_Re/Im`` with ``k_rg`` / ``k_az`` coordinates
  and no ``s1_int2.xml`` dependency. This is the primary input.
* **OCN WV measurement** products (``s1d-...-ocn-...nc``) : the polar spectrum
  ``oswPolSpec`` is resampled to a cartesian cross-spectrum via
  :func:`polar_to_cartesian` before the CWAVE computation.
"""

import argparse
import logging
import os
import sys
from typing import Any, cast

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from cwaves.compute_cwaves import compute_cwave_parameters
from cwaves.config import get_default_config

logger = logging.getLogger("cwaves.client")

DEFAULT_LAG = 2
DEFAULT_N_RANGE = 32
DEFAULT_N_AZIMUTH = 64


def _setup_logging() -> None:
    """Configure the ``cwaves.client`` logger to write INFO to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def polar_to_cartesian(
    ds: xr.Dataset,
    n_range: int = DEFAULT_N_RANGE,
    n_azimuth: int = DEFAULT_N_AZIMUTH,
) -> xr.DataArray:
    """Convert the OCN WV polar spectrum to a cartesian cross-spectrum.

    The OCN WV measurement stores a real polar spectrum ``oswPolSpec`` on the
    grid ``(oswK [rad/m], oswPhi [deg])``. It is resampled onto a cartesian
    wavenumber grid matching the convention expected by
    :func:`cwaves.compute_cwave_parameters` : one-sided in range
    (``k_rg >= 0``, mapped to ``freq_sample``) and two-sided in azimuth
    (``k_az``, mapped to ``freq_line``). The result is cast to complex.

    Args:
        ds: opened OCN WV measurement dataset (must contain ``oswPolSpec``,
            ``oswK`` and ``oswPhi``).
        n_range: number of cartesian range wavenumbers (``freq_sample``).
        n_azimuth: number of cartesian azimuth wavenumbers (``freq_line``).

    Returns:
        The cartesian complex cross-spectrum ``xarray.DataArray`` with
        ``k_rg`` / ``k_az`` coordinates.
    """
    pol = ds["oswPolSpec"].isel(oswAzSize=0, oswRaSize=0).values  # (phi, K)
    k = ds["oswK"].values  # (K,) rad/m
    phi_rad = np.deg2rad(ds["oswPhi"].values)  # (phi,)
    interpolator = RegularGridInterpolator(
        (k, phi_rad), pol.T, bounds_error=False, fill_value=0.0
    )

    kmax = float(k.max())
    k_rg = np.linspace(0.0, kmax, n_range)
    k_az = np.linspace(-kmax, kmax, n_azimuth)
    grid_k_rg, grid_k_az = np.meshgrid(k_rg, k_az, indexing="ij")

    k_query = np.sqrt(grid_k_rg**2 + grid_k_az**2)
    phi_query = (np.arctan2(grid_k_az, grid_k_rg) + 2 * np.pi) % (2 * np.pi)
    points = np.column_stack([k_query.ravel(), phi_query.ravel()])
    values = interpolator(points).reshape(n_range, n_azimuth)

    return xr.DataArray(
        values.astype(np.complex128),
        dims=("freq_sample", "freq_line"),
        coords={"k_rg": ("freq_sample", k_rg), "k_az": ("freq_line", k_az)},
        name="xspectra_osw",
    )


def compute_cwaves_from_osw(
    input_nc: str,
    n_range: int = DEFAULT_N_RANGE,
    n_azimuth: int = DEFAULT_N_AZIMUTH,
    **cwave: Any,
) -> xr.DataArray:
    """Compute the CWAVE parameters from an OCN WV measurement file.

    Args:
        input_nc: path to the OCN WV measurement NetCDF file.
        n_range: number of cartesian range wavenumbers.
        n_azimuth: number of cartesian azimuth wavenumbers.
        **cwave: ``cwave_*`` parameters forwarded to
            :func:`cwaves.compute_cwave_parameters` (defaults from the packaged
            ``default_config.yml`` if not given).

    Returns:
        The CWAVE parameters ``xarray.DataArray`` (``k_gp`` x ``phi_hf``).
    """
    if not os.path.exists(input_nc):
        raise FileNotFoundError(input_nc)
    if not cwave:
        cwave = get_default_config()["cwave"]
    with xr.open_dataset(input_nc) as ds:
        if "oswPolSpec" not in ds:
            raise ValueError(f"oswPolSpec not found in {input_nc}")
        logger.info("loaded OCN WV measurement %s", input_nc)
        xs = polar_to_cartesian(ds, n_range=n_range, n_azimuth=n_azimuth)
        cp = compute_cwave_parameters(xs, **cwave)
    logger.info(
        "CWAVE parameters computed: %d kernels (k_gp) x %d (phi_hf)",
        cp.sizes["k_gp"],
        cp.sizes["phi_hf"],
    )
    return cast("xr.DataArray", cp)


def _l1b_cross_spectrum(
    ds: xr.Dataset, lag: int, time_idx: int, pol_idx: int
) -> xr.DataArray:
    """Build the complex one-sided cross-spectrum for one L1B time slice."""
    re = (
        ds[f"xspectra_{lag}tau_Re"]
        .isel(time=time_idx, pol=pol_idx, drop=True)
        .mean(dim=f"{lag}tau")
    )
    im = (
        ds[f"xspectra_{lag}tau_Im"]
        .isel(time=time_idx, pol=pol_idx, drop=True)
        .mean(dim=f"{lag}tau")
    )
    return (
        (re + 1j * im)
        .assign_coords(
            k_rg=ds["k_rg"].isel(time=time_idx), k_az=ds["k_az"].isel(time=time_idx)
        )
        .rename(f"xspectra_{lag}tau")
    )


def compute_cwaves_from_l1b(
    input_nc: str,
    lag: int = DEFAULT_LAG,
    pol: int = 0,
    **cwave: Any,
) -> xr.DataArray:
    """Compute the CWAVE parameters for every time slice of an L1B XSP product.

    The L1B XSP product stores the cross-spectra as ``xspectra_{lag}tau_Re`` /
    ``xspectra_{lag}tau_Im`` (one-sided in range ``k_rg``, two-sided in azimuth
    ``k_az``) with no ``s1_int2.xml`` dependency. The highest-lag
    cross-spectrum is used by default (``--lag 2``, the 2-tau cross-spectrum).

    Args:
        input_nc: path to the L1B XSP cross-spectrum NetCDF file.
        lag: cross-spectrum lag (1 or 2). Default 2 (the 2-tau cross-spectrum).
        pol: polarization index. Default 0.
        **cwave: ``cwave_*`` parameters forwarded to
            :func:`cwaves.compute_cwave_parameters` (defaults from the packaged
            ``default_config.yml`` if not given).

    Returns:
        The CWAVE parameters ``xarray.DataArray`` with a ``time`` dimension.
    """
    if not os.path.exists(input_nc):
        raise FileNotFoundError(input_nc)
    if not cwave:
        cwave = get_default_config()["cwave"]
    re_var = f"xspectra_{lag}tau_Re"
    with xr.open_dataset(input_nc) as ds:
        if re_var not in ds:
            raise ValueError(f"{re_var} not found in {input_nc}")
        n_time = ds.sizes["time"]
        times = ds["sensing_time"].values
        logger.info(
            "loaded L1B XSP %s (%d time slices, lag=%dtau)", input_nc, n_time, lag
        )
        results = []
        for t in range(n_time):
            xs = _l1b_cross_spectrum(ds, lag, t, pol)
            cp = compute_cwave_parameters(xs, **cwave)
            results.append(cp)
            logger.info("time slice %d/%d: CWAVE computed", t + 1, n_time)
    out = xr.concat(results, dim="time")
    out = out.assign_coords(time=times)
    out.name = "cwave_params"
    logger.info(
        "CWAVE parameters computed: %d kernels (k_gp) x %d (phi_hf)",
        out.sizes["k_gp"],
        out.sizes["phi_hf"],
    )
    return cast("xr.DataArray", out)


def _cwave_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the ``cwave_*`` parameters from the parsed CLI options."""
    cwave: dict[str, Any] = {}
    if args.cwave_wlmax is not None:
        cwave["cwave_wlmax"] = args.cwave_wlmax
    if args.cwave_wlmin is not None:
        cwave["cwave_wlmin"] = args.cwave_wlmin
    if args.cwave_nk is not None:
        cwave["cwave_Nk"] = args.cwave_nk
    if args.cwave_nphi is not None:
        cwave["cwave_Nphi"] = args.cwave_nphi
    return cwave


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute CWAVE parameters from a Sentinel-1 SARWave product."
    )
    parser.add_argument(
        "--input-nc-osw",
        default=None,
        help="path to the OCN WV measurement NetCDF file",
    )
    parser.add_argument(
        "--input-nc-l1b",
        default=None,
        help="path to the L1B XSP cross-spectrum NetCDF file",
    )
    parser.add_argument(
        "--n-range",
        type=int,
        default=DEFAULT_N_RANGE,
        help="OCN WV: number of cartesian range wavenumbers",
    )
    parser.add_argument(
        "--n-azimuth",
        type=int,
        default=DEFAULT_N_AZIMUTH,
        help="OCN WV: number of cartesian azimuth wavenumbers",
    )
    parser.add_argument(
        "--lag",
        type=int,
        choices=[1, 2],
        default=DEFAULT_LAG,
        help="L1B XSP: cross-spectrum lag (default 2 = 2tau)",
    )
    parser.add_argument(
        "--pol", type=int, default=0, help="L1B XSP: polarization index (default 0)"
    )
    parser.add_argument("--output-nc", default=None, help="optional output NetCDF path")
    parser.add_argument(
        "--cwave-wlmax", type=float, default=None, help="maximum wavelength [m]"
    )
    parser.add_argument(
        "--cwave-wlmin", type=float, default=None, help="minimum wavelength [m]"
    )
    parser.add_argument(
        "--cwave-nk",
        type=int,
        default=None,
        help="number of Gegenbauer basis functions",
    )
    parser.add_argument(
        "--cwave-nphi",
        type=int,
        default=None,
        help="number of harmonic basis functions",
    )
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    """Command-line entry point for the CWAVE client."""
    _setup_logging()
    args = _build_parser().parse_args(argv)

    if bool(args.input_nc_osw) == bool(args.input_nc_l1b):
        logger.error("exactly one of --input-nc-osw / --input-nc-l1b must be provided")
        return 1

    cwave = _cwave_kwargs(args)
    try:
        if args.input_nc_l1b:
            out = compute_cwaves_from_l1b(
                args.input_nc_l1b, lag=args.lag, pol=args.pol, **cwave
            )
        else:
            out = compute_cwaves_from_osw(
                args.input_nc_osw,
                n_range=args.n_range,
                n_azimuth=args.n_azimuth,
                **cwave,
            )
    except (KeyError, ValueError, OSError) as exc:
        logger.error("failed to compute CWAVE parameters: %s", exc)
        return 1

    if args.output_nc is not None:
        out.to_netcdf(args.output_nc)
        logger.info("saved CWAVE parameters to %s", args.output_nc)
    logger.info("CWAVE parameters: %s", dict(out.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
