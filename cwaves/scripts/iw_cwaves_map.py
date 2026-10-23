"""Compute CWAVE parameters on the IW subswaths of an L2 OCN product.

This script (roadmap task 10) computes the CWAVE parameters, tile by tile, on the
cross-spectra of the IW subswaths (``iw1``/``iw2``/``iw3``) of a Sentinel-1 IW
ocean wave product and renders a single georeferenced map with the three
subswaths collocated in geographic space.

Two product families are supported (auto-detected from the ``.SAFE`` content):

* **OCN IW measurement** (``s1d_iw_ocn__2s``) : per-subswath ``s1?-iw?-osw-*.nc``
  files. The cross-spectra ``oswCartSpecRe/Im`` are on a **sinh (non-uniform)
  grid** in units ``m/m`` (cycles/m). They are converted to rad/m and resampled
  onto a regular grid (see :func:`resample_sinh_to_regular`) before the CWAVE
  projection, as required by the uniform-grid contract (see
  ``repository_specifications.md`` section 6.6).
* **SARWAVE L1B XSP** (``slc/iw/l1b``) : per-subswath ``l1b-s1?-iw?-xsp-*.nc``
  files. The cross-spectra ``xspectra_{lag}tau_Re/Im`` are already on a uniform
  grid in rad/m and are used as-is.

Both products hold ``intraburst`` and ``interburst`` sub-groups of cross-spectra.
The computation is fully paper-compliant: one-sided (half) + real modulus
``|xs|``, no symmetrization (see ``repository_specifications.md`` section 6).

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
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from cwaves.compute_cwaves import compute_cwave_parameters  # noqa: E402
from cwaves.config import get_default_config  # noqa: E402

logger = logging.getLogger("cwaves.iw_cwaves_map")

#: IW subswath identifiers
SUBSWATHS = ("iw1", "iw2", "iw3")
#: default polarization per S1 acquisition mode
DEFAULT_POL_BY_MODE = {"SDV": "VV", "SSV": "VV", "SDH": "HH", "SSH": "HH"}
#: default number of points of the regular grid used to resample the OCN sinh grid
DEFAULT_N_GRID = 128

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


def resample_sinh_to_regular(
    spec: xr.DataArray,
    kx_sinh: np.ndarray,
    ky_sinh: np.ndarray,
    kmax: float,
    n_grid: int = DEFAULT_N_GRID,
) -> xr.DataArray:
    """Resample a cartesian cross-spectrum from a sinh (non-uniform) grid to a regular one.

    The OCN ``oswCartSpec`` cross-spectra are sampled on a sinh grid
    (``k = sinh(i * r)``). The CWAVE projection requires a uniform wavenumber
    grid (``repository_specifications.md`` section 6.6), so the real modulus
    ``|spec|`` is resampled (scipy ``RegularGridInterpolator``) onto a regular
    grid spanning ``[0, kmax]`` in range and ``[-kmax, kmax]`` in azimuth.

    Args:
        spec: the (complex) cross-spectrum on the sinh grid, dims
            ``(ky_bin, kx_bin)``.
        kx_sinh: the (one-sided) sinh range wavenumbers in rad/m.
        ky_sinh: the (two-sided) sinh azimuth wavenumbers in rad/m.
        kmax: the maximum wavenumber (rad/m) of the CWAVE domain.
        n_grid: number of points of the regular grid along each axis.

    Returns:
        The resampled real-modulus cross-spectrum on the regular grid, dims
        ``(freq_sample, freq_line)`` with ``k_rg``/``k_az`` coordinates.
    """
    spec_values = spec.values
    # drop NaN wavenumber bins (some OCN interburst tiles carry trailing NaN
    # oswKy bins): the interpolator needs strictly ascending finite coordinates.
    kx_finite = np.isfinite(kx_sinh)
    ky_finite = np.isfinite(ky_sinh)
    kx_sinh = kx_sinh[kx_finite]
    ky_sinh = ky_sinh[ky_finite]
    spec_values = spec_values[np.ix_(ky_finite, kx_finite)]
    # values are (ky, kx); the interpolator grid is (kx, ky)
    values = np.abs(spec_values).T
    kx_lin = np.linspace(0.0, kmax, n_grid)
    ky_lin = np.linspace(-kmax, kmax, n_grid)
    grid_kx, grid_ky = np.meshgrid(kx_lin, ky_lin, indexing="ij")
    interpolator = RegularGridInterpolator(
        (kx_sinh, ky_sinh), values, bounds_error=False, fill_value=0.0
    )
    points = np.column_stack([grid_kx.ravel(), grid_ky.ravel()])
    resampled = interpolator(points).reshape(n_grid, n_grid)
    return xr.DataArray(
        resampled.astype(np.complex128),
        dims=("freq_sample", "freq_line"),
        coords={"k_rg": ("freq_sample", kx_lin), "k_az": ("freq_line", ky_lin)},
        name="xspectra_resampled",
    )


def compute_ocn_subswath(
    nc_path: str,
    group: str,
    lag: int,
    kmax: float,
    n_grid: int = DEFAULT_N_GRID,
    **cwave: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the CWAVE parameters of every tile of an OCN osw per-subswath file.

    Args:
        nc_path: path to the ``s1?-iw?-osw-*.nc`` file.
        group: ``"intraburst"`` or ``"interburst"``.
        lag: the ``oswLag`` index (0/1/2, default 2 = 2-tau).
        kmax: maximum wavenumber (rad/m) of the CWAVE domain.
        n_grid: number of points of the regular resampling grid.
        **cwave: ``cwave_*`` parameters for :func:`compute_cwave_parameters`.

    Returns:
        A tuple ``(lons, lats, params)`` where ``lons``/``lats`` have shape
        ``(oswAzSize, oswRaSize)`` (tile centers) and ``params`` has shape
        ``(oswAzSize, oswRaSize, k_gp, phi_hf)``.
    """
    if not cwave:
        cwave = get_default_config()["cwave"]
    with xr.open_dataset(nc_path, group=group) as ds:
        n_az, n_ra = ds.sizes["oswAzSize"], ds.sizes["oswRaSize"]
        lons = ds["oswLon"].values
        lats = ds["oswLat"].values
        has_lag = "oswLag" in ds["oswCartSpecRe"].dims
        params = np.full((n_az, n_ra, 4, 5), np.nan)
        for iaz in range(n_az):
            for ira in range(n_ra):
                kx = 2.0 * np.pi * ds["oswKx"].isel(oswAzSize=iaz, oswRaSize=ira).values
                ky = 2.0 * np.pi * ds["oswKy"].isel(oswAzSize=iaz, oswRaSize=ira).values
                spec = ds["oswCartSpecRe"].isel(oswAzSize=iaz, oswRaSize=ira)
                im = ds["oswCartSpecIm"].isel(oswAzSize=iaz, oswRaSize=ira)
                if has_lag:
                    spec = spec.isel(oswLag=lag)
                    im = im.isel(oswLag=lag)
                spec = spec + 1j * im
                if not (np.isfinite(spec).any()):
                    continue
                xs = resample_sinh_to_regular(spec, kx, ky, kmax, n_grid)
                cp = compute_cwave_parameters(xs, **cwave)
                params[iaz, ira] = cp.values
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
        **cwave: ``cwave_*`` parameters for :func:`compute_cwave_parameters`.

    Returns:
        A tuple ``(lons, lats, params)`` where ``lons``/``lats`` have shape
        ``(tile_line, tile_sample)`` (tile centers) and ``params`` has shape
        ``(tile_line, tile_sample, k_gp, phi_hf)``.
    """
    if not cwave:
        cwave = get_default_config()["cwave"]
    with xr.open_dataset(nc_path, group=group) as ds:
        pol_idx = _sarwave_pol_index(ds, pol)
        n_tl, n_ts = ds.sizes["tile_line"], ds.sizes["tile_sample"]
        lons = ds["longitude"].values
        lats = ds["latitude"].values
        tau_dim = f"{lag}tau"
        re = ds[f"xspectra_{lag}tau_Re"].isel(pol=pol_idx).mean(dim=tau_dim)
        im = ds[f"xspectra_{lag}tau_Im"].isel(pol=pol_idx).mean(dim=tau_dim)
        krg = ds["k_rg"]
        kaz = ds["k_az"]
        params = np.full((n_tl, n_ts, 4, 5), np.nan)
        for itl in range(n_tl):
            for its in range(n_ts):
                xs = (
                    (
                        re.isel(tile_line=itl, tile_sample=its, drop=True)
                        + 1j * im.isel(tile_line=itl, tile_sample=its, drop=True)
                    )
                    .assign_coords(
                        k_rg=krg.isel(tile_line=itl, tile_sample=its),
                        k_az=kaz,
                    )
                    .rename(f"xspectra_{lag}tau")
                )
                if not np.isfinite(xs.values).any():
                    continue
                cp = compute_cwave_parameters(xs, **cwave)
                params[itl, its] = cp.values
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
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    logger.info("map saved to %s", output_png)


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
        "--n-grid",
        type=int,
        default=DEFAULT_N_GRID,
        help="regular grid size for the OCN sinh resampling (default 128)",
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
    kmax = (
        2.0
        * np.pi
        / (cwave.get("cwave_wlmin") or get_default_config()["cwave"]["cwave_wlmin"])
    )

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
                    str(files[sub]), args.group, args.lag, kmax, args.n_grid, **cwave
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
