"""Regenerate the PNG figures used by the Sphinx documentation.

Run from the repository root with the development environment :

    python docs/generate_figures.py

This produces, in ``docs/figures/`` :

* ``kernels_k1_phi1.png``            -- a single CWAVE kernel (k_gp=1, phi_hf=1),
  showing the elliptic domain ``A`` and the ``kx = range`` / ``ky = azimuth``
  convention (see :mod:`cwaves.scripts.plot_kernels`).
* ``kernels_grid.png``               -- the full kernel basis (Nk x Nphi panels).
* ``certification_ocn_polar.png`` / ``certification_ocn_sinh.png`` /
  ``certification_xsp.png`` -- one figure per decode method, for the same S1A
  WV1 acquisition (2025-05-01, orbit 058996_075117) : the modulus ``|X_S|``,
  the same spectrum with the CWAVE kernel contours overlaid, and the pointwise
  integrand ``kernel x |X_S|_norm x dk`` (the intermediate step summed to
  produce the CWAVE parameter). All panels share the CWAVE display domain
  ``k_rg in [0, kmax]``, ``k_az in [-kmax, kmax]`` (black = zero spectrum).
* the SAR ``sigma0`` roughness image (``...-roughness.png``) is provided
  separately (not generated here) and is documented on the certification page.
* ``cwave_map_k1_phi1.png``          -- the georeferenced CWAVE map (k_gp=1,
  phi_hf=1) produced by the script :mod:`cwaves.scripts.iw_cwaves_map`.

The figures only need the core dependencies (numpy, xarray, scipy, matplotlib);
the map figure additionally needs ``cartopy`` (the ``scripts`` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cwaves.client import _l1b_cross_spectrum, polar_to_cartesian  # noqa: E402
from cwaves.compute_cwaves import compute_kernel  # noqa: E402
from cwaves.config import get_default_config  # noqa: E402
from cwaves.ocn_sinh import compute_ocn_sinh_cartesian  # noqa: E402
from cwaves.scripts.plot_kernels import (  # noqa: E402
    _apply_style,
    build_kernel_grid,
    domain_mask,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

#: white -> red signal colormap for the certification cross-spectra / integrands
CWMAP = mcolors.LinearSegmentedColormap.from_list(
    "cwave",
    ["white", "violet", "mediumpurple", "cyan", "springgreen", "yellow", "red"],
)

#: azimuth half-width (rad/m) of the certification display window (beyond it,
#: there is no signal and no kernel)
K_Y_BAND = 0.15

#: real OCN WV + SARWAVE XSP L1B fixtures, from the SAME S1A WV1 acquisition
#: (2025-05-01, orbit 058996_075117) so the certification rows share a scene.
OCN_WV_NC = REPO_ROOT / "assets" / "s1a-wv1-ocn-vv-20250501t060101-058996-075117-001.nc"
XSP_NC = REPO_ROOT / "assets" / "s1a_wv1_xsp_20250501t060101_058996_075117_dd24_a24.nc"


def kmin(wlmax: float, wlmin: float) -> float:
    """Minimum wavenumber (rad/m) from the maximum wavelength."""
    return 2.0 * np.pi / wlmax


def _gen_kernels() -> None:
    """Generate the single-kernel and full-grid kernel figures."""
    cfg = get_default_config()["cwave"]
    wlmax, wlmin = cfg["cwave_wlmax"], cfg["cwave_wlmin"]
    n_k, n_phi = cfg["cwave_Nk"], cfg["cwave_Nphi"]
    n_grid = 200

    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = build_kernel_grid(wlmax, wlmin, n_grid)
    kernel = compute_kernel(
        krg=krg, kaz=kaz, kmin=kmin, kmax=kmax, cwave_Nk=n_k, cwave_Nphi=n_phi
    )
    mask = domain_mask(krg, kaz, kmin, kmax)

    from cwaves.scripts.plot_kernels import plot_kernels  # local import (uses plt)

    plot_kernels(
        kernel,
        k_rg_1d,
        k_az_1d,
        mask,
        kmin,
        kmax,
        [(1, 1)],
        str(FIG_DIR / "kernels_k1_phi1.png"),
        "CWAVE kernel (k_gp = 1, phi_hf = 1)",
        "RdBu_r",
    )
    pairs = [(kg, ph) for kg in range(1, n_k + 1) for ph in range(1, n_phi + 1)]
    plot_kernels(
        kernel,
        k_rg_1d,
        k_az_1d,
        mask,
        kmin,
        kmax,
        pairs,
        str(FIG_DIR / "kernels_grid.png"),
        f"CWAVE kernel basis (Nk = {n_k}, Nphi = {n_phi})",
        "RdBu_r",
    )


def _row_arrays(
    xs: xr.DataArray, kmin_: float, kmax: float, n_k: int, n_phi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the two data panels (spec, integrand) and the CWAVE value.

    The integrand is exactly the pointwise product summed by
    :func:`cwaves.compute_cwave_parameters` :
    ``kernel * |xs|_normalized * dk_rg * dk_az``.

    Args:
        xs: the one-sided (``k_rg >= 0``) complex cross-spectrum, dims
            ``(freq_sample, freq_line)``.
        kmin_: minimum wavenumber (rad/m).
        kmax: maximum wavenumber (rad/m).
        n_k: number of Gegenbauer basis functions.
        n_phi: number of harmonic basis functions.

    Returns:
        A tuple ``(k_rg_1d, k_az_1d, spec_2d, integrand_2d, cwave_val)`` with
        the 2-D arrays oriented for plotting (rows = azimuth, cols = range).
    """
    xs = xs.transpose("freq_sample", "freq_line")
    k_rg_1d = np.asarray(xs.k_rg, dtype=float)
    k_az_1d = np.asarray(xs.k_az, dtype=float)
    spec = (xs.real**2 + xs.imag**2) ** 0.5
    kernel = compute_kernel(
        krg=xs.k_rg,
        kaz=xs.k_az,
        kmin=kmin_,
        kmax=kmax,
        cwave_Nk=n_k,
        cwave_Nphi=n_phi,
    )
    k1 = kernel["cwave_kernel"].sel(k_gp=1, phi_hf=1)
    kk = (xs.k_rg**2 + xs.k_az**2) ** 0.5
    xxs = xs.where((kk > kmin_) & (kk < kmax))
    xxsm = (xxs.real**2 + xxs.imag**2) ** 0.5
    dkrg = float(xxs.k_rg.diff(dim="freq_sample").mean(dim="freq_sample"))
    dkaz = float(xxs.k_az.diff(dim="freq_line").mean(dim="freq_line"))
    xxsmn = xxsm / (
        xxsm.sum(dim=["freq_line", "freq_sample"], skipna=True) * dkrg * dkaz
    )
    integrand = (k1 * xxsmn) * dkrg * dkaz
    cwave_val = float(
        integrand.sum(dim=["freq_sample", "freq_line"], skipna=True, min_count=1)
    )
    return (
        k_rg_1d,
        k_az_1d,
        spec.transpose("freq_line", "freq_sample").values,
        integrand.transpose("freq_line", "freq_sample").values,
        cwave_val,
    )


def _common_kernel_field(
    kmin_: float,
    kmax_display: float,
    k_y_band: float,
    n_k: int,
    n_phi: int,
    k_gp: int,
    phi_hf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (k_gp, phi_hf) kernel on a common regular display grid.

    The display grid spans ``[0, kmax_display]`` in range and
    ``[-k_y_band, k_y_band]`` in azimuth and is **independent of the
    cross-spectrum's own azimuth extent** (which is narrow for the SARWAVE XSP
    but wide for the OCN decodes). Computing the kernel here, rather than on the
    data grid, gives every decode method the identical, un-truncated kernel
    contours. The field is masked outside the elliptic domain ``A``.

    Returns:
        ``(k_rg_2d, k_az_2d, kernel_2d)`` oriented for ``contour`` (rows =
        azimuth, cols = range), ``kernel_2d`` NaN outside domain ``A``.
    """
    k_rg_1d = np.linspace(0.0, kmax_display, 200)
    k_az_1d = np.linspace(-k_y_band, k_y_band, 200)
    krg, kaz = np.meshgrid(k_rg_1d, k_az_1d, indexing="ij")
    kernel = compute_kernel(
        krg=xr.DataArray(krg, dims=("i", "j")),
        kaz=xr.DataArray(kaz, dims=("i", "j")),
        kmin=kmin_,
        kmax=kmax_display,
        cwave_Nk=n_k,
        cwave_Nphi=n_phi,
    )
    k1 = kernel["cwave_kernel"].sel(k_gp=k_gp, phi_hf=phi_hf).values
    mask = domain_mask(
        xr.DataArray(krg, dims=("i", "j")),
        xr.DataArray(kaz, dims=("i", "j")),
        kmin_,
        kmax_display,
    ).values
    return (
        krg.T,
        kaz.T,
        np.where(mask.T, k1.T, np.nan),
    )


def _plot_method(
    fig,
    axes,
    k_rg_1d,
    k_az_1d,
    spec_2d,
    integrand_2d,
    cwave_val,
    kmax_display,
    k_y_band,
    title,
    k_rg_disp,
    k_az_disp,
    kernel_disp,
):
    """Render the three panels of one decode method on a common display domain.

    All panels share the display domain ``[0, kmax_display] x
    [-k_y_band, k_y_band]`` (cropped in azimuth, where there is no signal or
    kernel beyond it) and the white->red :data:`CWMAP` colormap (white = zero),
    so the panels are directly comparable and the empty corners read as zero.
    The kernel contours use a common display grid (:func:`_common_kernel_field`)
    so they are not truncated by the data's azimuth extent (narrow for XSP).
    """
    pcm = axes[0].pcolormesh(k_rg_1d, k_az_1d, spec_2d, cmap=CWMAP, shading="auto")
    axes[0].set_title(r"cross-spectrum  $|X_S|$")
    fig.colorbar(pcm, ax=axes[0], label=r"$|X_S|$  (a.u.)")

    pcm = axes[1].pcolormesh(
        k_rg_1d, k_az_1d, spec_2d, cmap=CWMAP, shading="auto", alpha=0.85
    )
    axes[1].contour(
        k_rg_disp, k_az_disp, kernel_disp, levels=16, colors="0.25", linewidths=1.0
    )
    axes[1].set_title(r"$|X_S|$  + CWAVE kernel  $(k_{gp}=1,\ \varphi_{hf}=1)$")
    fig.colorbar(pcm, ax=axes[1], label=r"$|X_S|$  (a.u.)")

    pcm = axes[2].pcolormesh(k_rg_1d, k_az_1d, integrand_2d, cmap=CWMAP, shading="auto")
    axes[2].set_title(
        r"integrand  kernel $\times\, |X_S|_{\rm norm}$"
        r"   $\ \sum \Rightarrow$  CWAVE $= $" + f"{cwave_val:.2f}"
    )
    fig.colorbar(pcm, ax=axes[2], label=r"kernel $\times\, |X_S|_{\rm norm}$  (a.u.)")

    for ax in axes:
        ax.set_xlim(0.0, kmax_display)
        ax.set_ylim(-k_y_band, k_y_band)
        ax.set_xlabel(r"$k_x$ (range)  [rad m$^{-1}$]")
        ax.set_ylabel(r"$k_y$ (azimuth)  [rad m$^{-1}$]")
    fig.suptitle(title, fontsize=14)


def _gen_certification() -> None:
    """Certify the CWAVE kernels on real, co-registered Sentinel-1 WV data.

    Generates one figure per decode method, each with three panels : the modulus
    ``|X_S|``, the same spectrum with the (k_gp=1, phi_hf=1) kernel contours
    overlaid, and the pointwise integrand ``kernel x |X_S|_norm x dk`` that is
    summed to produce the CWAVE parameter. All methods come from the same S1A
    WV1 acquisition (2025-05-01, orbit 058996_075117) and are plotted on the
    same display domain (see :func:`_plot_method`). The SAR ``sigma0``
    roughness image is documented separately (not generated here).
    """
    cfg = get_default_config()["cwave"]
    wlmax, wlmin = cfg["cwave_wlmax"], cfg["cwave_wlmin"]
    n_k, n_phi = cfg["cwave_Nk"], cfg["cwave_Nphi"]
    kmin_ = kmin(wlmax, wlmin)
    kmax_display = 2.0 * np.pi / wlmin
    n_grid = 128

    with xr.open_dataset(OCN_WV_NC) as ds:
        xs_polar = polar_to_cartesian(ds, n_range=n_grid, n_azimuth=n_grid)
        xs_sinh = compute_ocn_sinh_cartesian(ds, lag=2, m_range=256, m_azimuth=256)
    with xr.open_dataset(XSP_NC) as ds:
        xs_xsp = _l1b_cross_spectrum(ds, lag=2, time_idx=0, pol_idx=0)

    methods = [
        ("OCN WV  polar $\\to$ cartesian", "ocn_polar", xs_polar),
        ("OCN WV  cart-sinh  $\\to$ regular", "ocn_sinh", xs_sinh),
        ("SARWAVE L1B  XSP  (2tau)", "xsp", xs_xsp),
    ]
    # common (un-truncated) kernel field, shared by all three decode methods
    k_rg_disp, k_az_disp, kernel_disp = _common_kernel_field(
        kmin_, kmax_display, K_Y_BAND, n_k, n_phi, 1, 1
    )
    _apply_style()
    for title, fname, xs in methods:
        (
            k_rg_1d,
            k_az_1d,
            spec_2d,
            integrand_2d,
            cwave_val,
        ) = _row_arrays(xs, kmin_, kmax_display, n_k, n_phi)
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), constrained_layout=True)
        _plot_method(
            fig,
            axes,
            k_rg_1d,
            k_az_1d,
            spec_2d,
            integrand_2d,
            cwave_val,
            kmax_display,
            K_Y_BAND,
            title,
            k_rg_disp,
            k_az_disp,
            kernel_disp,
        )
        fig.savefig(FIG_DIR / f"certification_{fname}.png", bbox_inches="tight")
        plt.close(fig)


def _all_param_panels(
    xs: xr.DataArray, kmin_: float, kmax: float, n_k: int, n_phi: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, float]]]:
    """Compute the integrand panel and the CWAVE value for every (k_gp, phi_hf).

    Generalises :func:`_row_arrays` to the full default basis : for each of the
    ``n_k x n_phi`` kernels it returns the pointwise integrand
    ``kernel_{kg,ph} x |X_S|_norm x dk`` (the step summed to yield the CWAVE
    parameter) and the resulting scalar value.

    Args:
        xs: the one-sided (``k_rg >= 0``) complex cross-spectrum.
        kmin_: minimum wavenumber (rad/m).
        kmax: maximum wavenumber (rad/m).
        n_k: number of Gegenbauer basis functions.
        n_phi: number of harmonic basis functions.

    Returns:
        A tuple ``(k_rg_1d, k_az_1d, panels)`` where ``panels`` is a list of
        ``(integrand_2d, cwave_val)`` ordered row-major (``k_gp`` outer,
        ``phi_hf`` inner), each 2-D array oriented for plotting.
    """
    xs = xs.transpose("freq_sample", "freq_line")
    k_rg_1d = np.asarray(xs.k_rg, dtype=float)
    k_az_1d = np.asarray(xs.k_az, dtype=float)
    kernel = compute_kernel(
        krg=xs.k_rg,
        kaz=xs.k_az,
        kmin=kmin_,
        kmax=kmax,
        cwave_Nk=n_k,
        cwave_Nphi=n_phi,
    )
    kk = (xs.k_rg**2 + xs.k_az**2) ** 0.5
    xxs = xs.where((kk > kmin_) & (kk < kmax))
    xxsm = (xxs.real**2 + xxs.imag**2) ** 0.5
    dkrg = float(xxs.k_rg.diff(dim="freq_sample").mean(dim="freq_sample"))
    dkaz = float(xxs.k_az.diff(dim="freq_line").mean(dim="freq_line"))
    xxsmn = xxsm / (
        xxsm.sum(dim=["freq_line", "freq_sample"], skipna=True) * dkrg * dkaz
    )
    panels: list[tuple[np.ndarray, float]] = []
    for kg in range(1, n_k + 1):
        for ph in range(1, n_phi + 1):
            k = kernel["cwave_kernel"].sel(k_gp=kg, phi_hf=ph)
            integrand = (k * xxsmn) * dkrg * dkaz
            val = float(
                integrand.sum(
                    dim=["freq_sample", "freq_line"], skipna=True, min_count=1
                )
            )
            panels.append((integrand.transpose("freq_line", "freq_sample").values, val))
    return k_rg_1d, k_az_1d, panels


def _gen_certification_all() -> None:
    """Render the full default CWAVE basis (20 params) for each decode method.

    For each of the three decode methods (same S1A WV1 acquisition as
    :func:`_gen_certification`) this produces one figure with a ``Nk x Nphi``
    grid of panels (default 4 x 5 = 20 CWAVE parameters). Each panel shows the
    pointwise integrand ``kernel_{kg,ph} x |X_S|_norm`` for that parameter, with
    the resulting CWAVE value in the title, so the three methods' full
    20-parameter output can be compared at a glance.
    """
    cfg = get_default_config()["cwave"]
    wlmax, wlmin = cfg["cwave_wlmax"], cfg["cwave_wlmin"]
    n_k, n_phi = cfg["cwave_Nk"], cfg["cwave_Nphi"]
    kmin_ = kmin(wlmax, wlmin)
    kmax_display = 2.0 * np.pi / wlmin
    n_grid = 128

    with xr.open_dataset(OCN_WV_NC) as ds:
        xs_polar = polar_to_cartesian(ds, n_range=n_grid, n_azimuth=n_grid)
        xs_sinh = compute_ocn_sinh_cartesian(ds, lag=2, m_range=256, m_azimuth=256)
    with xr.open_dataset(XSP_NC) as ds:
        xs_xsp = _l1b_cross_spectrum(ds, lag=2, time_idx=0, pol_idx=0)

    methods = [
        ("OCN WV  polar $\\to$ cartesian", "ocn_polar", xs_polar),
        ("OCN WV  cart-sinh  $\\to$ regular", "ocn_sinh", xs_sinh),
        ("SARWAVE L1B  XSP  (2tau)", "xsp", xs_xsp),
    ]
    _apply_style()
    for title, fname, xs in methods:
        k_rg_1d, k_az_1d, panels = _all_param_panels(
            xs, kmin_, kmax_display, n_k, n_phi
        )
        fig, axes = plt.subplots(
            n_k, n_phi, figsize=(4.2 * n_phi, 3.2 * n_k), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        # one shared, symmetric color scale across all panels of this method,
        # bounded by the 90 % central interval (5th-95th percentile) of the
        # absolute integrand so that isolated kernel peaks do not wash out
        # the bulk of the data
        all_abs = np.concatenate([np.abs(p).ravel() for p, _ in panels])
        vmax = float(np.nanpercentile(all_abs, 95))
        for i, ax in enumerate(axes.ravel()):
            kg, ph = divmod(i, n_phi)
            integrand, val = panels[i]
            ax.pcolormesh(
                k_rg_1d,
                k_az_1d,
                integrand,
                cmap="bwr",
                shading="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_xlim(0.0, kmax_display)
            ax.set_ylim(-K_Y_BAND, K_Y_BAND)
            ax.set_title(
                rf"$k_{{gp}}$={kg + 1}, $\varphi_{{hf}}$={ph + 1}"
                f"\nCWAVE = {val:.3f}",
                fontsize=9,
            )
            if kg == n_k - 1:
                ax.set_xlabel(r"$k_x$ (range) [rad m$^{-1}$]", fontsize=8)
            if ph == 0:
                ax.set_ylabel(r"$k_y$ (azimuth) [rad m$^{-1}$]", fontsize=8)
        fig.colorbar(
            plt.cm.ScalarMappable(
                cmap="bwr",
                norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            ),
            ax=axes.ravel().tolist(),
            label=r"kernel $\times\, |X_S|_{\rm norm}$  (a.u.)",
        )
        fig.suptitle(
            f"{title}  --  CWAVE basis (Nk = {n_k}, Nphi = {n_phi}, "
            f"{n_k * n_phi} parameters)",
            fontsize=14,
        )
        fig.savefig(FIG_DIR / f"certification_all_{fname}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _gen_kernels()
    _gen_certification()
    _gen_certification_all()
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
