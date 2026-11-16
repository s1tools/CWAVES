"""Command-line client: CWAVE parameters of a spectrum stored in a NetCDF file.

The client has **one** entry point, :func:`compute_cwaves_from_file` (CLI
``cwaves --input-nc <file>``). It opens the file, **detects its schema**, extracts
the spectrum with its native wavenumber grid and hands it to the library entry
point :func:`cwaves.compute_cwaves`. No resampling is done: the library
integrates on the native grid (cartesian or polar, uniform or not).

Recognised schemas (detection is based on the variables present, never on the
file name):

* ``"xsp"`` -- a cross-spectrum file with ``xspectra_{lag}tau_Re`` /
  ``xspectra_{lag}tau_Im`` and cartesian ``k_rg`` / ``k_az`` coordinates in
  rad/m, one spectrum per ``time`` slice (e.g. SARWave L1B XSP WV).
* ``"polar"`` -- a polar spectrum ``oswPolSpec`` on ``oswK`` [rad/m] x
  ``oswPhi`` [degrees] (e.g. ESA OCN WV), with ``oswPhi = 0`` along the range
  axis (``k_rg = k cos(phi)``).

Files whose spectra live in NetCDF **groups** (tiled IW products, ``intraburst``
/ ``interburst``) hold one spectrum per tile: they are rejected with an explicit
message pointing to the tile-wise script :mod:`cwaves.scripts.iw_cwaves_map`.
"""

import argparse
import logging
import os
import sys
from typing import Any, cast

import h5py
import numpy as np
import xarray as xr

from cwaves.compute_cwaves import PARTS, compute_cwaves
from cwaves.config import get_default_config

logger = logging.getLogger("cwaves.client")

DEFAULT_LAG = 2

#: schema name -> human readable description (for messages and the CLI help)
SCHEMAS = {
    "xsp": "cross-spectrum xspectra_{lag}tau_Re/Im on (k_rg, k_az) [rad/m]",
    "polar": "polar spectrum oswPolSpec on (oswK [rad/m], oswPhi [deg])",
}


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


def _root_content(input_nc: str | os.PathLike[str]) -> tuple[set[str], list[str]]:
    """Return the root variable names and group names of a NetCDF file.

    NetCDF-4 (HDF5) files are inspected with ``h5py`` (no data read); other
    formats (NetCDF-3) fall back to ``xarray`` and have no groups.
    """
    try:
        with h5py.File(input_nc, "r") as h5:
            variables = {k for k, v in h5.items() if isinstance(v, h5py.Dataset)}
            groups = [k for k, v in h5.items() if isinstance(v, h5py.Group)]
        return variables, groups
    except OSError:
        with xr.open_dataset(input_nc) as ds:
            return set(ds.variables), []


def detect_schema(input_nc: str | os.PathLike[str]) -> str:
    """Return the schema of a NetCDF file (``"xsp"`` or ``"polar"``).

    Args:
        input_nc: path to the NetCDF file.

    Returns:
        The schema name (a key of :data:`SCHEMAS`).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the spectra are stored in groups (tiled product) or no
            supported spectrum variable is found.
    """
    if not os.path.exists(input_nc):
        raise FileNotFoundError(input_nc)
    variables, groups = _root_content(input_nc)
    if "oswPolSpec" in variables:
        return "polar"
    if any(v.startswith("xspectra_") and v.endswith("_Re") for v in variables):
        return "xsp"
    if groups:
        raise ValueError(
            f"{input_nc} stores its spectra in groups {groups} (tiled product, one "
            "spectrum per tile): use the tile-wise script "
            "`python -m cwaves.scripts.iw_cwaves_map`, or open the group yourself "
            "and call cwaves.compute_cwaves on each tile spectrum."
        )
    raise ValueError(
        f"no supported spectrum found in {input_nc}; expected one of: "
        + "; ".join(SCHEMAS.values())
    )


def _xsp_spectrum(
    ds: xr.Dataset, lag: int, time_idx: int, pol_idx: int
) -> xr.DataArray:
    """Build the complex cross-spectrum of one ``time`` slice of an XSP file."""
    real = ds[f"xspectra_{lag}tau_Re"].isel(time=time_idx, pol=pol_idx, drop=True)
    imag = ds[f"xspectra_{lag}tau_Im"].isel(time=time_idx, pol=pol_idx, drop=True)
    if f"{lag}tau" in real.dims:  # average the lag pairs
        real = real.mean(dim=f"{lag}tau")
        imag = imag.mean(dim=f"{lag}tau")
    spec = real + 1j * imag
    return spec.assign_coords(
        k_rg=ds["k_rg"].isel(time=time_idx), k_az=ds["k_az"].isel(time=time_idx)
    ).rename(f"xspectra_{lag}tau")


#: backward-compatible private alias (used by the report / figure scripts)
_l1b_cross_spectrum = _xsp_spectrum


def polar_spectrum(ds: xr.Dataset, iaz: int = 0, ira: int = 0) -> xr.DataArray:
    """Extract the polar spectrum ``oswPolSpec`` with ``k`` / ``phi`` coordinates.

    Args:
        ds: an opened dataset with ``oswPolSpec``, ``oswK`` [rad/m] and
            ``oswPhi`` [degrees] (``oswPhi = 0`` along the range axis).
        iaz: ``oswAzSize`` index of the tile.
        ira: ``oswRaSize`` index of the tile.

    Returns:
        The polar spectrum, dims ``(phi, k)``, ``phi`` in radians.
    """
    pol = ds["oswPolSpec"].isel(oswAzSize=iaz, oswRaSize=ira).values  # (phi, K)
    k = np.asarray(ds["oswK"].values, dtype=float)
    phi = np.deg2rad(np.asarray(ds["oswPhi"].values, dtype=float))
    return xr.DataArray(
        pol, dims=("phi", "k"), coords={"phi": phi, "k": k}, name="oswPolSpec"
    )


def compute_cwaves_from_file(
    input_nc: str | os.PathLike[str],
    lag: int = DEFAULT_LAG,
    pol: int = 0,
    **cwave: Any,
) -> xr.DataArray:
    """Compute the CWAVE parameters of the spectrum (spectra) stored in a file.

    The schema is detected with :func:`detect_schema`; the spectrum is passed on
    its native grid to :func:`cwaves.compute_cwaves`.

    Args:
        input_nc: path to the NetCDF file.
        lag: cross-spectrum lag for the ``"xsp"`` schema (default 2 = 2-tau).
        pol: polarization index for the ``"xsp"`` schema (default 0).
        **cwave: ``cwave_*`` parameters (defaults: packaged
            ``default_config.yml``).

    Returns:
        The CWAVE parameters (``k_gp`` x ``phi_hf``, plus ``time`` for
        ``"xsp"`` files). The ``schema`` attribute records the detected schema.
    """
    schema = detect_schema(input_nc)
    if not cwave:
        cwave = get_default_config()["cwave"]
    with xr.open_dataset(input_nc) as ds:
        logger.info("loaded %s (schema %r: %s)", input_nc, schema, SCHEMAS[schema])
        if schema == "polar":
            out = compute_cwaves(polar_spectrum(ds), **cwave)
        else:
            re_var = f"xspectra_{lag}tau_Re"
            if re_var not in ds:
                raise ValueError(f"{re_var} not found in {input_nc}")
            results = [
                compute_cwaves(_xsp_spectrum(ds, lag, t, pol), **cwave)
                for t in range(ds.sizes["time"])
            ]
            out = xr.concat(results, dim="time").assign_coords(
                time=ds["sensing_time"].values
            )
            out.attrs.update(results[0].attrs)
            out.name = "cwave_params"
    out.attrs["schema"] = schema
    logger.info(
        "CWAVE parameters computed on the native %s grid (%s plane): %d (k_gp) x "
        "%d (phi_hf)",
        out.attrs.get("grid"),
        out.attrs.get("coverage"),
        out.sizes["k_gp"],
        out.sizes["phi_hf"],
    )
    return cast("xr.DataArray", out)


def to_netcdf(cwave_params: xr.DataArray, path: str | os.PathLike[str]) -> None:
    """Write CWAVE parameters to NetCDF (complex values split into re / im).

    NetCDF has no complex type: complex parameters (``part="complex"``) are
    written as two variables ``cwave_params_re`` and ``cwave_params_im``.
    """
    if np.iscomplexobj(cwave_params.values):
        ds = xr.Dataset(
            {
                "cwave_params_re": np.real(cwave_params),
                "cwave_params_im": np.imag(cwave_params),
            },
            attrs=cwave_params.attrs,
        )
        ds.to_netcdf(path)
    else:
        cwave_params.to_netcdf(path)


def _cwave_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the ``cwave_*`` parameters (config defaults + CLI overrides)."""
    cwave: dict[str, Any] = dict(get_default_config()["cwave"])
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
        description="Compute the CWAVE parameters of the spectrum stored in a "
        "NetCDF file (schema auto-detected: "
        + "; ".join(f"{k} = {v}" for k, v in SCHEMAS.items())
        + ")."
    )
    parser.add_argument("--input-nc", required=True, help="input NetCDF file")
    parser.add_argument(
        "--lag",
        type=int,
        choices=[1, 2],
        default=DEFAULT_LAG,
        help="xsp schema: cross-spectrum lag (default 2 = 2tau)",
    )
    parser.add_argument(
        "--pol", type=int, default=0, help="xsp schema: polarization index (default 0)"
    )
    parser.add_argument(
        "--part",
        choices=list(PARTS),
        default="modulus",
        help="projected part: modulus (default, paper), real, imag (odd harmonics) "
        "or complex (written as cwave_params_re / cwave_params_im)",
    )
    parser.add_argument("--output-nc", default=None, help="optional output NetCDF path")
    parser.add_argument(
        "--cwave-wlmax", type=float, default=None, help="maximum wavelength [m]"
    )
    parser.add_argument(
        "--cwave-wlmin", type=float, default=None, help="minimum wavelength [m]"
    )
    parser.add_argument(
        "--cwave-nk", type=int, default=None, help="number of Gegenbauer functions"
    )
    parser.add_argument(
        "--cwave-nphi", type=int, default=None, help="number of harmonic functions"
    )
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    """Command-line entry point (``cwaves``)."""
    _setup_logging()
    args = _build_parser().parse_args(argv)
    try:
        out = compute_cwaves_from_file(
            args.input_nc,
            lag=args.lag,
            pol=args.pol,
            part=args.part,
            **_cwave_kwargs(args),
        )
    except (KeyError, ValueError, OSError) as exc:
        logger.error("failed to compute CWAVE parameters: %s", exc)
        return 1
    if args.output_nc is not None:
        to_netcdf(out, args.output_nc)
        logger.info("saved CWAVE parameters to %s", args.output_nc)
    logger.info("CWAVE parameters: %s", dict(out.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
