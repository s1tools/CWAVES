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

The kernel is even in the range wavenumber ``k_rg``, but the integration domain
``A`` is **one-sided in range** (``k_rg >= 0``). The ``k_rg < 0`` half of each
panel is therefore shaded to make clear that it is not part of ``A`` and is not
integrated (see ``repository_specifications.md`` section 6.1).

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
from matplotlib.figure import Figure  # noqa: E402

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


def _domain_coeffs(kmin: float, kmax: float) -> tuple[float, float]:
    """Return the (a1, a2) coefficients of the elliptic domain ``A``.

    ``u = a1*kaz**4 + a2*kaz**2 + krg**2`` (paper equation B1, ``gamma=2``).
    """
    a1 = (GAMMA**2 - GAMMA**4) / (GAMMA**2 * kmin**2 - kmax**2)
    a2 = (kmax**2 - GAMMA**4 * kmin**2) / (kmax**2 - GAMMA**2 * kmin**2)
    return float(a1), float(a2)


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
    a1, a2 = _domain_coeffs(kmin, kmax)
    u = a1 * kaz**4 + a2 * kaz**2 + krg**2
    return (u >= kmin**2) & (u <= kmax**2)


def _apply_style() -> None:
    """Apply publication-quality matplotlib settings.

    Serif font : Times New Roman when available on the system, otherwise
    ``STIXGeneral`` (metric-compatible with Times New Roman, bundled with
    matplotlib). Mathtext uses the matching STIX fontset so that the symbols
    in the axis labels and titles blend with the roman text.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


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


#: axis labels (mathtext, library convention kx = range, ky = azimuth)
X_LABEL = r"$k_x$ (range)  [rad m$^{-1}$]"
Y_LABEL = r"$k_y$ (azimuth)  [rad m$^{-1}$]"
#: panel title template (mathtext)
PANEL_TITLE = r"$k_{{gp}}$ = {kg},  $\varphi_{{hf}}$ = {ph}"


def _build_figure(
    kernel: xr.Dataset,
    k_rg_1d: np.ndarray,
    k_az_1d: np.ndarray,
    mask: xr.DataArray,
    kmin: float,
    kmax: float,
    pairs: list[tuple[int, int]],
    title: str,
    cmap: str,
) -> Figure:
    """Build (but neither save nor close) the kernel figure.

    See :func:`plot_kernels` for the argument documentation and the
    publication-style / domain-``A`` shading description.

    Raises:
        ValueError: if ``pairs`` is empty.
    """
    if not pairs:
        raise ValueError("no kernel selected to plot")
    _apply_style()
    nrows = len({kg for kg, _ in pairs})
    ncols = len({ph for _, ph in pairs})
    single = nrows == 1 and ncols == 1

    fields: list[np.ndarray] = []
    for kg, ph in pairs:
        field = kernel["cwave_kernel"].sel(k_gp=kg, phi_hf=ph).values
        fields.append(np.where(mask.values, field, np.nan))

    finite = np.concatenate([f[np.isfinite(f)] for f in fields])
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    # domain-A boundary (dashed contour), computed once on the full grid
    a1, a2 = _domain_coeffs(kmin, kmax)
    krg_grid, kaz_grid = np.meshgrid(k_rg_1d, k_az_1d, indexing="ij")
    u_grid = a1 * kaz_grid**4 + a2 * kaz_grid**2 + krg_grid**2

    figsize = (12.0, 10.0) if single else (2.8 * ncols + 0.8, 2.85 * nrows + 0.8)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=300,
        squeeze=False,
        constrained_layout=True,
    )
    pcm = None
    for ip, (ax, (kg, ph), field) in enumerate(
        zip(axes.ravel(), pairs, fields, strict=True)
    ):
        # field is (i=k_rg, j=k_az); pcolormesh wants (rows=k_az, cols=k_rg)
        pcm = ax.pcolormesh(
            k_rg_1d, k_az_1d, field.T, cmap=cmap, norm=norm, shading="auto"
        )
        # shade the k_rg < 0 half : the kernel is even in k_rg so this half is a
        # mirror of the integrated one, but it is OUTSIDE the integration domain A
        # (one-sided in range, see repository_specifications.md section 6.1)
        ax.axvspan(
            float(k_rg_1d.min()),
            0.0,
            facecolor="0.55",
            alpha=0.13,
            hatch="////",
            edgecolor="0.55",
            linewidth=0.0,
            zorder=1,
        )
        ax.contour(
            k_rg_1d,
            k_az_1d,
            u_grid.T,
            levels=[kmin**2, kmax**2],
            colors="0.35",
            linewidths=0.9,
            linestyles="--",
            zorder=2,
        )
        if not single:
            ax.set_title(PANEL_TITLE.format(kg=kg, ph=ph))
        if ip == 0:
            ax.text(
                0.04,
                0.95,
                r"$k_x<0$ : outside $A$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9 if not single else 11,
                color="0.25",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": "0.7",
                    "lw": 0.5,
                    "alpha": 0.8,
                },
            )
        ax.set_aspect("equal")
        ax.set_xlim(float(k_rg_1d.min()), float(k_rg_1d.max()))
        ax.set_ylim(float(k_az_1d.min()), float(k_az_1d.max()))
    for ax in axes[-1, :]:
        ax.set_xlabel(X_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL)
    if pcm is None:  # unreachable : pairs is non-empty (checked above)
        raise ValueError("no kernel was rendered")
    cbar = fig.colorbar(
        pcm,
        ax=axes.ravel().tolist(),
        label="kernel amplitude",
        pad=0.10 if single else 0.08,
    )
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(title, fontsize=13)
    return fig


def plot_kernels(
    kernel: xr.Dataset,
    k_rg_1d: np.ndarray,
    k_az_1d: np.ndarray,
    mask: xr.DataArray,
    kmin: float,
    kmax: float,
    pairs: list[tuple[int, int]],
    output_png: str,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    """Render the selected CWAVE kernels to a PNG file.

    The kernel is a smooth function over the whole wavenumber plane and is even
    in the range wavenumber ``k_rg`` (the left half is a mirror of the right
    half). The **integration domain ``A`` is one-sided in range** (``k_rg >= 0``,
    see ``repository_specifications.md`` section 6.1), so the ``k_rg < 0`` half of
    each panel is shaded (hatched gray) to make clear that it is *not* part of
    ``A`` and is *not* integrated by :func:`compute_cwave_parameters`.

    Style : serif font (Times New Roman, STIX fallback), mathtext labels,
    thin spines with inward ticks, a shared symmetric colorbar, the elliptic
    domain-``A`` boundary drawn as a dashed contour, 300 dpi.

    Args:
        kernel: the kernel dataset from :func:`compute_kernel`
            (variable ``cwave_kernel``, dims ``k_gp/phi_hf/i/j``).
        k_rg_1d: 1D range-wavenumber axis (rad/m).
        k_az_1d: 1D azimuth-wavenumber axis (rad/m).
        mask: boolean domain-``A`` mask (dims ``i/j``).
        kmin: minimum wavenumber (rad/m), for the domain boundary.
        kmax: maximum wavenumber (rad/m), for the domain boundary.
        pairs: ordered list of ``(k_gp, phi_hf)`` to plot.
        output_png: path of the output PNG file.
        title: figure super-title.
        cmap: matplotlib colormap name (a diverging map is recommended).

    Raises:
        ValueError: if ``pairs`` is empty.
    """
    fig = _build_figure(kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, pairs, title, cmap)
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
    plot_kernels(
        kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, pairs, out_png, title, args.cmap
    )
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
