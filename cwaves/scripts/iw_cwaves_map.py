"""Compute CWAVE parameters on the IW subswaths of an L2 OCN product.

This script computes the CWAVE parameters, tile by tile, on the
cross-spectra of the IW subswaths (``iw1``/``iw2``/``iw3``) of a Sentinel-1 IW
ocean wave product and renders a single georeferenced map with the three
subswaths collocated in geographic space.

Two product families are supported (auto-detected from the ``.SAFE`` content):

* **OCN IW measurement** (``s1d_iw_ocn__2s``) : per-subswath ``s1?-iw?-osw-*.nc``
  files. The cross-spectra ``oswCartSpecRe/Im`` are on a **sinh (non-uniform)
  grid** in rad/m (checked against a colocated L1B XSP, issue #12). They are
  passed **on their native sinh grid** to :func:`cwaves.compute_cwaves`, which
  integrates with the exact non-uniform cell areas (no resampling; the grid
  size may vary from tile to tile).
* **SARWAVE L1B XSP** (``slc/iw/l1b``) : per-subswath ``l1b-s1?-iw?-xsp-*.nc``
  files. The cross-spectra ``xspectra_{lag}tau_Re/Im`` are already on a uniform
  grid in rad/m and are used as-is.

Both products hold ``intraburst`` and ``interburst`` sub-groups of cross-spectra.
The CWAVE parameters are computed by the library entry point
:func:`cwaves.compute_cwaves` (real modulus of the half-plane cross-spectrum,
see ``repository_specifications.md`` section 6).

Usage::

    python -m cwaves.scripts.iw_cwaves_map --input-safe <.SAFE> \
        [--group intraburst|interburst] [--lag 2] [--pol VV|HH] \
        [--k-gp 1] [--phi-hf 1] [--output-png <out.png>]
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from cwaves.compute_cwaves import compute_cwaves  # noqa: E402
from cwaves.config import get_default_config  # noqa: E402

logger = logging.getLogger("cwaves.iw_cwaves_map")

#: IW subswath identifiers
SUBSWATHS = ("iw1", "iw2", "iw3")
#: default polarization per S1 acquisition mode
DEFAULT_POL_BY_MODE = {"SDV": "VV", "SSV": "VV", "SDH": "HH", "SSH": "HH"}

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)
logger.propagate = False


def _setup_logging(verbose: bool) -> None:
    """Configure the logger level."""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def detect_product(safe_dir: str) -> str:
    """Auto-detect the product family of a ``.SAFE`` directory.

    Args:
        safe_dir: path to the ``.SAFE`` directory.

    Returns:
        ``"ocn"`` for an OCN IW measurement product (``s1?-iw?-osw-*.nc``) or
        ``"sarwave"`` for a SARWAVE L1B XSP product (``l1b-s1?-iw?-xsp-*.nc``).

    Raises:
        ValueError: if no per-subswath cross-spectrum file is found.
    """
    safe = Path(safe_dir)
    if not safe.is_dir():
        raise ValueError(f"not a directory: {safe_dir}")
    ocn = list(safe.rglob("s1?-iw?-osw-*.nc"))
    sarwave = list(safe.rglob("l1b-s1?-iw?-xsp-*.nc"))
    if ocn:
        return "ocn"
    if sarwave:
        return "sarwave"
    raise ValueError(f"no per-subswath cross-spectrum file found in {safe_dir}")


def _mode_from_safe(safe_dir: str) -> str:
    """Extract the acquisition mode (``SDV``/``SSV``/``SDH``/``SSH``) from a SAFE name."""
    m = re.search(r"__(2S)([DVH])", os.path.basename(safe_dir))
    return m.group(0)[2:] if m else "SDV"


def default_pol(mode: str) -> str:
    """Return the default polarization for an acquisition mode (VV for SDV/SSV, HH for SDH/SSH)."""
    return DEFAULT_POL_BY_MODE.get(mode.upper(), "VV")


def _subswath_files(safe_dir: str, kind: str, pol: str) -> dict[str, Path]:
    """Return the per-subswath cross-spectrum file paths (``{iw1: path, ...}``).

    Args:
        safe_dir: path to the ``.SAFE`` directory.
        kind: ``"ocn"`` or ``"sarwave"``.
        pol: requested polarization (``"VV"``/``"HH"`), used to select the OCN
            per-pol file.
    """
    safe = Path(safe_dir)
    if kind == "ocn":
        pattern = f"s1?-iw?-osw-{pol.lower()}-*.nc"
    else:
        pattern = "l1b-s1?-iw?-xsp-*.nc"
    files: dict[str, Path] = {}
    for nc in safe.rglob(pattern):
        m = re.search(r"-iw(\d)-", nc.name)
        if m:
            files[f"iw{m.group(1)}"] = nc
    return files


def _cwave_shape(cwave: dict[str, Any]) -> tuple[int, int]:
    """Return the ``(Nk, Nphi)`` basis size of a ``cwave_*`` parameter dict."""
    return int(cwave["cwave_Nk"]), int(cwave["cwave_Nphi"])


def ocn_tile_spectrum(ds: xr.Dataset, iaz: int, ira: int, lag: int) -> xr.DataArray:
    """Complex cross-spectrum of one OCN osw tile on its native sinh grid.

    The ``oswKx`` / ``oswKy`` wavenumbers are physical rad/m (a colocated L1B
    XSP check rejects cycles/m, issue #12) and are used as is; grid lines
    with NaN wavenumbers (trailing bins of some interburst tiles) are
    dropped. The result carries the cartesian ``k_rg`` / ``k_az`` coordinates
    expected by :func:`cwaves.compute_cwaves`.

    Args:
        ds: an opened OCN osw group (``intraburst`` / ``interburst``).
        iaz: ``oswAzSize`` index of the tile.
        ira: ``oswRaSize`` index of the tile.
        lag: ``oswLag`` index (ignored if the product has no lag dimension).

    Returns:
        The complex cross-spectrum ``real + 1j * imag``, dims
        ``(ky_bin, kx_bin)`` of the product.
    """
    tile = {"oswAzSize": iaz, "oswRaSize": ira}
    real = ds["oswCartSpecRe"].isel(tile)
    imag = ds["oswCartSpecIm"].isel(tile)
    if "oswLag" in real.dims:
        real = real.isel(oswLag=lag)
        imag = imag.isel(oswLag=lag)
    spec = real + 1j * imag
    k_rg = np.asarray(ds["oswKx"].isel(tile).values, dtype=float)
    k_az = np.asarray(ds["oswKy"].isel(tile).values, dtype=float)
    dim_az, dim_rg = spec.dims  # (ky_bin, kx_bin)
    spec = spec.assign_coords(k_rg=(dim_rg, k_rg), k_az=(dim_az, k_az))
    return spec.isel({dim_rg: np.isfinite(k_rg), dim_az: np.isfinite(k_az)})


def compute_ocn_subswath(
    nc_path: str,
    group: str,
    lag: int,
    **cwave: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the CWAVE parameters of every tile of an OCN osw per-subswath file.

    Each tile cross-spectrum is used on its native sinh grid (no resampling).

    Args:
        nc_path: path to the ``s1?-iw?-osw-*.nc`` file.
        group: ``"intraburst"`` or ``"interburst"``.
        lag: the ``oswLag`` index (0/1/2, default 2 = 2-tau).
        **cwave: ``cwave_*`` parameters for :func:`cwaves.compute_cwaves`
            (defaults: packaged ``default_config.yml``).

    Returns:
        A tuple ``(lons, lats, params)`` where ``lons``/``lats`` have shape
        ``(oswAzSize, oswRaSize)`` (tile centers) and ``params`` has shape
        ``(oswAzSize, oswRaSize, cwave_Nk, cwave_Nphi)``.
    """
    cwave = {**get_default_config()["cwave"], **cwave}
    with xr.open_dataset(nc_path, group=group) as ds:
        n_az, n_ra = ds.sizes["oswAzSize"], ds.sizes["oswRaSize"]
        lons = ds["oswLon"].values
        lats = ds["oswLat"].values
        params = np.full((n_az, n_ra, *_cwave_shape(cwave)), np.nan)
        for iaz in range(n_az):
            for ira in range(n_ra):
                spec = ocn_tile_spectrum(ds, iaz, ira, lag)
                if not np.isfinite(spec.values).any():
                    continue
                params[iaz, ira] = compute_cwaves(spec, **cwave).values
        logger.info(
            "OCN %s: %d x %d tiles computed (lag=%d)",
            os.path.basename(nc_path),
            n_az,
            n_ra,
            lag,
        )
    return lons, lats, params


def _sarwave_pol_index(ds: xr.Dataset, pol: str) -> int:
    """Return the index of ``pol`` in the SARWAVE ``pol`` coordinate."""
    pols = [str(p).upper() for p in ds["pol"].values]
    if pol.upper() not in pols:
        raise ValueError(f"polarization {pol} not in {pols}")
    return pols.index(pol.upper())


def compute_sarwave_subswath(
    nc_path: str,
    group: str,
    lag: int,
    pol: str,
    **cwave: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the CWAVE parameters of every tile of a SARWAVE L1B XSP file.

    The cross-spectra are already on a uniform grid in rad/m and are used as-is.

    Args:
        nc_path: path to the ``l1b-s1?-iw?-xsp-*.nc`` file.
        group: ``"intraburst"`` or ``"interburst"``.
        lag: the cross-spectrum lag (0/1/2, default 2 = 2-tau).
        pol: polarization (``"VV"``/``"HH"``).
        **cwave: ``cwave_*`` parameters for :func:`cwaves.compute_cwaves`
            (defaults: packaged ``default_config.yml``).

    Returns:
        A tuple ``(lons, lats, params)`` where ``lons``/``lats`` have shape
        ``(tile_line, tile_sample)`` (tile centers) and ``params`` has shape
        ``(tile_line, tile_sample, cwave_Nk, cwave_Nphi)``.
    """
    cwave = {**get_default_config()["cwave"], **cwave}
    with xr.open_dataset(nc_path, group=group) as ds:
        pol_idx = _sarwave_pol_index(ds, pol)
        n_tl, n_ts = ds.sizes["tile_line"], ds.sizes["tile_sample"]
        lons = ds["longitude"].values
        lats = ds["latitude"].values
        # intraburst cross-spectra are lag-indexed (xspectra_{lag}tau_Re/Im with a
        # '{lag}tau' dimension); interburst cross-spectra are plain
        # 'xspectra_Re/Im' with no lag dimension.
        re_var = (
            f"xspectra_{lag}tau_Re" if f"xspectra_{lag}tau_Re" in ds else "xspectra_Re"
        )
        im_var = (
            f"xspectra_{lag}tau_Im" if f"xspectra_{lag}tau_Im" in ds else "xspectra_Im"
        )
        tau_dim = f"{lag}tau"
        real = ds[re_var].isel(pol=pol_idx)
        imag = ds[im_var].isel(pol=pol_idx)
        if tau_dim in real.dims:
            real = real.mean(dim=tau_dim)
            imag = imag.mean(dim=tau_dim)
        krg = ds["k_rg"]
        kaz = ds["k_az"]
        params = np.full((n_tl, n_ts, *_cwave_shape(cwave)), np.nan)
        for itl in range(n_tl):
            for its in range(n_ts):
                xs = (
                    (
                        real.isel(tile_line=itl, tile_sample=its, drop=True)
                        + 1j * imag.isel(tile_line=itl, tile_sample=its, drop=True)
                    )
                    .assign_coords(
                        k_rg=krg.isel(tile_line=itl, tile_sample=its),
                        k_az=kaz,
                    )
                    .rename(f"xspectra_{lag}tau")
                )
                if not np.isfinite(xs.values).any():
                    continue
                params[itl, its] = compute_cwaves(xs, **cwave).values
        logger.info(
            "SARWAVE %s: %d x %d tiles computed (lag=%d, pol=%s)",
            os.path.basename(nc_path),
            n_tl,
            n_ts,
            lag,
            pol,
        )
    return lons, lats, params


def plot_map(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_gp: int,
    phi_hf: int,
    output_png: str,
    title: str,
) -> None:
    """Render the CWAVE map of all subswaths on a single georeferenced figure.

    Args:
        results: ``{subswath: (lons, lats, params)}`` with ``params`` shaped
            ``(n1, n2, k_gp, phi_hf)``.
        k_gp: the Gegenbauer index to map (1-based).
        phi_hf: the harmonic index to map (1-based).
        output_png: path of the output PNG file.
        title: figure title.

    Raises:
        ImportError: if ``cartopy`` is not installed (install the ``scripts``
            extra : ``pip install "cwaves[scripts]"``).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)

    all_values: list[np.ndarray] = []
    for sub, (lons, lats, params) in results.items():
        vals = params[:, :, k_gp - 1, phi_hf - 1]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        lons_f = lons[finite]
        lats_f = lats[finite]
        vals_f = vals[finite]
        all_values.append(vals_f)
        ax.scatter(
            lons_f,
            lats_f,
            c=vals_f,
            s=12,
            cmap="viridis",
            edgecolors="none",
            label=sub,
            transform=ccrs.PlateCarree(),
        )
    if all_values:
        all_values_cat = np.concatenate(all_values)
        norm = Normalize(
            vmin=float(np.nanmin(all_values_cat)), vmax=float(np.nanmax(all_values_cat))
        )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"CWAVE  k_gp={k_gp}, phi_hf={phi_hf}")
    ax.set_title(title)
    if len(results) > 0:
        ax.legend(loc="upper left", fontsize=8)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1.0,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": "black", "fontsize": 8}
    gl.ylabel_style = {"color": "black", "fontsize": 8}
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    logger.info("map saved to %s", output_png)


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
        description="Compute CWAVE parameters on the IW subswaths and render a map."
    )
    parser.add_argument(
        "--input-safe", required=True, help="path to the .SAFE directory"
    )
    parser.add_argument(
        "--group",
        choices=["intraburst", "interburst"],
        default="intraburst",
        help="cross-spectra group (default intraburst)",
    )
    parser.add_argument(
        "--lag", type=int, default=2, help="cross-spectrum lag (default 2 = 2tau)"
    )
    parser.add_argument(
        "--pol",
        default=None,
        help="polarization (default: VV for SDV/SSV, HH for SDH/SSH)",
    )
    parser.add_argument(
        "--k-gp", type=int, default=1, help="Gegenbauer index to map (default 1)"
    )
    parser.add_argument(
        "--phi-hf", type=int, default=1, help="harmonic index to map (default 1)"
    )
    parser.add_argument(
        "--output-png", default=None, help="output PNG path (default: derived)"
    )
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
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    """Command-line entry point for the IW subswath CWAVE map script."""
    _setup_logging(False)
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    try:
        kind = detect_product(args.input_safe)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("product family: %s", kind)
    mode = _mode_from_safe(args.input_safe)
    pol = (args.pol or default_pol(mode)).upper()
    cwave = _cwave_kwargs(args)

    files = _subswath_files(args.input_safe, kind, pol)
    if not files:
        logger.error("no per-subswath file found (kind=%s, pol=%s)", kind, pol)
        return 1
    logger.info("subswaths found: %s", sorted(files))

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sub in SUBSWATHS:
        if sub not in files:
            logger.warning("subswath %s not present, skipped", sub)
            continue
        try:
            if kind == "ocn":
                results[sub] = compute_ocn_subswath(
                    str(files[sub]), args.group, args.lag, **cwave
                )
            else:
                results[sub] = compute_sarwave_subswath(
                    str(files[sub]), args.group, args.lag, pol, **cwave
                )
        except (KeyError, ValueError, OSError) as exc:
            logger.error("failed on subswath %s: %s", sub, exc)
            return 1

    if not results:
        logger.error("no subswath could be processed")
        return 1

    safe_stem = os.path.basename(os.path.normpath(args.input_safe))
    out_png = args.output_png or os.path.join(
        os.getcwd(),
        f"{safe_stem}_cwave_{args.group}-{args.lag}tau-{pol.lower()}_k{args.k_gp}_phi{args.phi_hf}.png",
    )
    title = (
        f"CWAVE (k_gp={args.k_gp}, phi_hf={args.phi_hf}) - {safe_stem}\n"
        f"mode={mode} group={args.group} lag={args.lag}tau pol={pol} product={kind}"
    )
    plot_map(results, args.k_gp, args.phi_hf, out_png, title)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
