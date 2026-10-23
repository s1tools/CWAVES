"""Plot the CWAVE kernels in the (kx, ky) wavenumber domain.

This script (roadmap task 12) renders the orthonormal CWAVE kernel basis of
Schulz-Stellenfleth et al. (2007) as 2D fields over the elliptic wavenumber
domain ``A`` (paper equations B1-B13), in the style of the right-hand subplot
of Fig. 2 of Quach et al. (2020).

Axis convention (same as the latest ``xsar_slc`` commits and
``repository_specifications.md``) :

* ``kx`` = range wavenumber (``k_rg``), horizontal axis,
* ``ky`` = azimuth wavenumber (``k_az``), vertical axis,
* the major axis of the kernel ellipses lies along the range direction.

The kernels are masked outside the domain ``A`` (where the wavenumber mapping
is not defined), so each panel shows the characteristic elliptical support.

Usage::

    python -m cwaves.scripts.plot_kernels --output-png kernels.png
    python -m cwaves.scripts.plot_kernels --k-gp 1 --phi-hf 1 \
        --output-png k1_phi1.png
    python -m cwaves.scripts.plot_kernels --k-gp 2 --output-png row_k2.png
"""

import argparse
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from cwaves.compute_cwaves import compute_kernel  # noqa: E402
from cwaves.config import get_default_config  # noqa: E402

logger = logging.getLogger("cwaves.plot_kernels")

#: default number of points along each wavenumber axis of the plot grid
DEFAULT_N_GRID = 200
#: margin factor applied to ``kmax`` so the ellipse is not clipped at the edges
GRID_MARGIN = 1.1
#: elliptic-domain parameter (paper equation B1)
GAMMA = 2.0


def _setup_logging(verbose: bool) -> None:
    """Configure the logger level."""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def build_kernel_grid(
    wlmax: float, wlmin: float, n_grid: int
) -> tuple[xr.DataArray, xr.DataArray, float, float, np.ndarray, np.ndarray]:
    """Build the two-sided (k_rg, k_az) meshgrid and the wavenumber bounds.

    The grid spans ``[-kmax*GRID_MARGIN, kmax*GRID_MARGIN]`` in both range and
    azimuth so that the full elliptic domain ``A`` (elongated in range) is
    visible with a small margin.

    Args:
        wlmax: maximum wavelength [m] (sets ``kmin = 2*pi/wlmax``).
        wlmin: minimum wavelength [m] (sets ``kmax = 2*pi/wlmin``).
        n_grid: number of points along each axis.

    Returns:
        A tuple ``(krg, kaz, kmin, kmax, k_rg_1d, k_az_1d)`` where ``krg``/
        ``kaz`` are 2D meshgrid DataArrays with dims ``("i", "j")``
        (``i`` = range, ``j`` = azimuth) and ``k_rg_1d``/``k_az_1d`` are the
        1D axis vectors.
    """
    kmin = 2.0 * np.pi / wlmax
    kmax = 2.0 * np.pi / wlmin
    kmax_m = kmax * GRID_MARGIN
    k_rg_1d = np.linspace(-kmax_m, kmax_m, n_grid)
    k_az_1d = np.linspace(-kmax_m, kmax_m, n_grid)
    krg_grid, kaz_grid = np.meshgrid(k_rg_1d, k_az_1d, indexing="ij")
    krg = xr.DataArray(krg_grid, dims=("i", "j"), coords={"k_rg": ("i", k_rg_1d)})
    kaz = xr.DataArray(kaz_grid, dims=("i", "j"), coords={"k_az": ("j", k_az_1d)})
    return krg, kaz, kmin, kmax, k_rg_1d, k_az_1d


def domain_mask(
    krg: xr.DataArray, kaz: xr.DataArray, kmin: float, kmax: float
) -> xr.DataArray:
    """Return the boolean mask of the elliptic wavenumber domain ``A``.

    The domain is defined by ``kmin**2 <= u <= kmax**2`` with
    ``u = a1*kaz**4 + a2*kaz**2 + krg**2`` (paper equation B1, ``gamma=2``).

    Args:
        krg: 2D range-wavenumber grid.
        kaz: 2D azimuth-wavenumber grid.
        kmin: minimum wavenumber (rad/m).
        kmax: maximum wavenumber (rad/m).

    Returns:
        A boolean DataArray (True inside the domain ``A``).
    """
    a1 = (GAMMA**2 - GAMMA**4) / (GAMMA**2 * kmin**2 - kmax**2)
    a2 = (kmax**2 - GAMMA**4 * kmin**2) / (kmax**2 - GAMMA**2 * kmin**2)
    u = a1 * kaz**4 + a2 * kaz**2 + krg**2
    return (u >= kmin**2) & (u <= kmax**2)


def select_kernels(
    n_k: int, n_phi: int, k_gp: int | None, phi_hf: int | None
) -> tuple[list[tuple[int, int]], int, int]:
    """Return the (k_gp, phi_hf) pairs to plot and the subplot grid shape.

    Args:
        n_k: number of Gegenbauer basis functions (``cwave_Nk``).
        n_phi: number of harmonic basis functions (``cwave_Nphi``).
        k_gp: single Gegenbauer index to plot (1-based) or None for all.
        phi_hf: single harmonic index to plot (1-based) or None for all.

    Returns:
        A tuple ``(pairs, nrows, ncols)`` where ``pairs`` is the ordered list
        of ``(k_gp, phi_hf)`` (k_gp as rows, phi_hf as columns).

    Raises:
        ValueError: if ``k_gp`` or ``phi_hf`` is out of range.
    """
    if k_gp is not None and not (1 <= k_gp <= n_k):
        raise ValueError(f"k_gp={k_gp} out of range [1, {n_k}]")
    if phi_hf is not None and not (1 <= phi_hf <= n_phi):
        raise ValueError(f"phi_hf={phi_hf} out of range [1, {n_phi}]")
    kgs = [k_gp] if k_gp is not None else list(range(1, n_k + 1))
    phs = [phi_hf] if phi_hf is not None else list(range(1, n_phi + 1))
    pairs = [(kg, ph) for kg in kgs for ph in phs]
    return pairs, len(kgs), len(phs)


def plot_kernels(
    kernel: xr.Dataset,
    k_rg_1d: np.ndarray,
    k_az_1d: np.ndarray,
    mask: xr.DataArray,
    pairs: list[tuple[int, int]],
    output_png: str,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    """Render the selected CWAVE kernels as a grid of 2D panels.

    Args:
        kernel: the kernel dataset from :func:`compute_kernel`
            (variable ``cwave_kernel``, dims ``k_gp/phi_hf/i/j``).
        k_rg_1d: 1D range-wavenumber axis (rad/m).
        k_az_1d: 1D azimuth-wavenumber axis (rad/m).
        mask: boolean domain-``A`` mask (dims ``i/j``).
        pairs: ordered list of ``(k_gp, phi_hf)`` to plot.
        output_png: path of the output PNG file.
        title: figure super-title.
        cmap: matplotlib colormap name (a diverging map is recommended).

    Raises:
        ValueError: if ``pairs`` is empty.
    """
    if not pairs:
        raise ValueError("no kernel selected to plot")
    nrows = len({kg for kg, _ in pairs})
    ncols = len({ph for _, ph in pairs})

    fields: list[np.ndarray] = []
    for kg, ph in pairs:
        field = kernel["cwave_kernel"].sel(k_gp=kg, phi_hf=ph).values
        fields.append(np.where(mask.values, field, np.nan))

    finite = np.concatenate([f[np.isfinite(f)] for f in fields])
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.1 * nrows), dpi=150, squeeze=False
    )
    pcm = None
    for ax, (kg, ph), field in zip(axes.ravel(), pairs, fields, strict=True):
        # field is (i=k_rg, j=k_az); pcolormesh wants (rows=k_az, cols=k_rg)
        pcm = ax.pcolormesh(
            k_rg_1d, k_az_1d, field.T, cmap=cmap, norm=norm, shading="auto"
        )
        ax.set_title(f"k_gp={kg}, phi_hf={ph}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(float(k_rg_1d.min()), float(k_rg_1d.max()))
        ax.set_ylim(float(k_az_1d.min()), float(k_az_1d.max()))
    for ax in axes[-1, :]:
        ax.set_xlabel("kx (range) [rad/m]", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("ky (azimuth) [rad/m]", fontsize=9)
    if pcm is None:  # unreachable : pairs is non-empty (checked above)
        raise ValueError("no kernel was rendered")
    fig.suptitle(title, fontsize=11)
    fig.colorbar(pcm, ax=axes.ravel().tolist(), label="kernel amplitude", pad=0.02)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    logger.info("kernels saved to %s", output_png)


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
        description="Plot the CWAVE kernels in the (kx, ky) wavenumber domain."
    )
    parser.add_argument(
        "--k-gp",
        type=int,
        default=None,
        help="Gegenbauer index to plot (1-based); default: all",
    )
    parser.add_argument(
        "--phi-hf",
        type=int,
        default=None,
        help="harmonic index to plot (1-based); default: all",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=DEFAULT_N_GRID,
        help=f"plot grid size along each axis (default {DEFAULT_N_GRID})",
    )
    parser.add_argument(
        "--output-png", default=None, help="output PNG path (default: derived)"
    )
    parser.add_argument(
        "--cmap",
        default="RdBu_r",
        help="matplotlib colormap (default RdBu_r)",
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
    """Command-line entry point for the CWAVE kernel plotting script."""
    _setup_logging(False)
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    cwave = _cwave_kwargs(args)
    default = get_default_config()["cwave"]
    wlmax = cwave.get("cwave_wlmax", default["cwave_wlmax"])
    wlmin = cwave.get("cwave_wlmin", default["cwave_wlmin"])
    n_k = cwave.get("cwave_Nk", default["cwave_Nk"])
    n_phi = cwave.get("cwave_Nphi", default["cwave_Nphi"])

    try:
        pairs, _, _ = select_kernels(n_k, n_phi, args.k_gp, args.phi_hf)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = build_kernel_grid(
        wlmax, wlmin, args.n_grid
    )
    kernel = compute_kernel(
        krg=krg,
        kaz=kaz,
        kmin=kmin,
        kmax=kmax,
        cwave_Nk=n_k,
        cwave_Nphi=n_phi,
    )
    mask = domain_mask(krg, kaz, kmin, kmax)

    out_png = args.output_png or (
        f"cwave_kernels_Nk{n_k}_Nphi{n_phi}_wl{int(wlmin)}-{int(wlmax)}m.png"
    )
    title = (
        f"CWAVE kernels (Schulz-Stellenfleth et al. 2007) - "
        f"Nk={n_k}, Nphi={n_phi}, wl {int(wlmin)}-{int(wlmax)} m"
    )
    plot_kernels(kernel, k_rg_1d, k_az_1d, mask, pairs, out_png, title, args.cmap)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
