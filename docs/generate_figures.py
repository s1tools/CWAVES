"""Regenerate the PNG figures used by the Sphinx documentation.

Run from the repository root with the development environment :

    python docs/generate_figures.py

This produces, in ``docs/figures/`` :

* ``kernels_k1_phi1.png``            -- a single CWAVE kernel (k_gp=1, phi_hf=1),
  showing the elliptic domain ``A`` and the ``kx = range`` / ``ky = azimuth``
  convention (see :mod:`cwaves.scripts.plot_kernels`).
* ``kernels_grid.png``               -- the full kernel basis (Nk x Nphi panels).
* ``certification_crossspec_kernels.png`` -- a real ESA OCN WV Sentinel-1 cartesian
  cross-spectrum (``oswCartSpecRe``, one tile) with the CWAVE kernel contours
  overlaid, to certify that the kernel basis matches the spectral structure.
* ``cwave_map_k1_phi1.png``          -- the georeferenced CWAVE map (k_gp=1,
  phi_hf=1) produced by the task-10 script :mod:`cwaves.scripts.iw_cwaves_map`.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cwaves.compute_cwaves import compute_kernel  # noqa: E402
from cwaves.config import get_default_config  # noqa: E402
from cwaves.scripts.iw_cwaves_map import (  # noqa: E402
    compute_ocn_subswath,
    plot_map,
    resample_sinh_to_regular,
)
from cwaves.scripts.plot_kernels import (  # noqa: E402
    _apply_style,
    build_kernel_grid,
    domain_mask,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

#: path of the ESA OCN WV IW osw fixture shipped in ``assets/``
OCN_IW1_NC = REPO_ROOT / "assets" / "s1d-iw1-osw-vv-fixture.nc"


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


def _gen_certification() -> None:
    """Overlay the CWAVE kernels on a real OCN cartesian cross-spectrum."""
    cfg = get_default_config()["cwave"]
    wlmax, wlmin = cfg["cwave_wlmax"], cfg["cwave_wlmin"]
    n_k, n_phi = cfg["cwave_Nk"], cfg["cwave_Nphi"]
    kmax = 2.0 * np.pi / wlmin
    n_grid = 128

    # one real tile of the OCN IW measurement (cartesian cross-spectrum, sinh grid)
    with xr.open_dataset(OCN_IW1_NC, group="intraburst") as ds:
        kx = 2.0 * np.pi * ds["oswKx"].isel(oswAzSize=0, oswRaSize=0).values
        ky = 2.0 * np.pi * ds["oswKy"].isel(oswAzSize=0, oswRaSize=0).values
        spec = ds["oswCartSpecRe"].isel(oswAzSize=0, oswRaSize=0, oswLag=2) + 1j * ds[
            "oswCartSpecIm"
        ].isel(oswAzSize=0, oswRaSize=0, oswLag=2)
    # resample |spec| onto the regular (one-sided range, two-sided azimuth) grid
    xs = resample_sinh_to_regular(spec, kx, ky, kmax, n_grid)

    # the kernel on the same regular grid
    krg_grid, kaz_grid = np.meshgrid(xs.k_rg.values, xs.k_az.values, indexing="ij")
    krg = xr.DataArray(krg_grid, dims=("freq_sample", "freq_line"))
    kaz = xr.DataArray(kaz_grid, dims=("freq_sample", "freq_line"))
    kernel = compute_kernel(
        krg=krg,
        kaz=kaz,
        kmin=kmin(wlmax, wlmin),
        kmax=kmax,
        cwave_Nk=n_k,
        cwave_Nphi=n_phi,
    )

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), constrained_layout=True)

    spec_2d = np.abs(xs.values).T  # (rows=azimuth, cols=range)
    k_rg_1d = xs.k_rg.values
    k_az_1d = xs.k_az.values

    # (a) the real cartesian cross-spectrum, masked to the domain A
    pcm = axes[0].pcolormesh(k_rg_1d, k_az_1d, spec_2d, cmap="inferno", shading="auto")
    axes[0].set_title("Cartesian cross-spectrum  $|X_S|$  (ESA OCN WV, iw1)")
    fig.colorbar(pcm, ax=axes[0], label=r"$|X_S|$  (a.u.)")

    # (b) the same cross-spectrum with the CWAVE kernel contours overlaid
    pcm = axes[1].pcolormesh(
        k_rg_1d, k_az_1d, spec_2d, cmap="inferno", shading="auto", alpha=0.85
    )
    k1 = kernel["cwave_kernel"].sel(k_gp=1, phi_hf=1).values  # (freq_sample, freq_line)
    cs = axes[1].contour(
        k_rg_1d, k_az_1d, k1.T, levels=24, colors="cyan", linewidths=1.0
    )
    axes[1].clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    axes[1].set_title(r"overlaid with the CWAVE kernel  $(k_{gp}=1,\ \varphi_{hf}=1)$")
    fig.colorbar(pcm, ax=axes[1], label=r"$|X_S|$  (a.u.)")

    for ax in axes:
        ax.axvspan(
            float(k_rg_1d.min()), 0.0, facecolor="0.55", alpha=0.10, hatch="////"
        )
        ax.set_aspect("equal")
        ax.set_xlim(float(k_rg_1d.min()), float(k_rg_1d.max()))
        ax.set_ylim(float(k_az_1d.min()), float(k_az_1d.max()))
    axes[0].text(
        0.03,
        0.96,
        r"$k_x<0$ : outside $A$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.2",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.85},
    )
    for ax in axes:
        ax.set_xlabel(r"$k_x$ (range)  [rad m$^{-1}$]")
        ax.set_ylabel(r"$k_y$ (azimuth)  [rad m$^{-1}$]")

    fig.savefig(FIG_DIR / "certification_crossspec_kernels.png", bbox_inches="tight")
    plt.close(fig)


def _gen_map() -> None:
    """Generate the georeferenced CWAVE map (k_gp=1, phi_hf=1) via task-10 functions.

    Combines the two product families supported by the task-10 script : the ESA
    OCN WV IW osw fixture (``iw1``) and the SARWAVE L1B XSP fixture (``iw2``).
    """
    from cwaves.scripts.iw_cwaves_map import compute_sarwave_subswath

    cfg = get_default_config()["cwave"]
    kmax = 2.0 * np.pi / cfg["cwave_wlmin"]
    ocn = compute_ocn_subswath(str(OCN_IW1_NC), "intraburst", 2, kmax, 96)
    sarwave = compute_sarwave_subswath(
        str(REPO_ROOT / "assets" / "l1b-s1a-iw2-xsp-fixture.nc"), "intraburst", 2, "VV"
    )
    results = {"iw1 (OCN osw)": ocn, "iw2 (SARWAVE xsp)": sarwave}
    title = "CWAVE (k_gp = 1, phi_hf = 1) - Sentinel-1 IW, intraburst, 2tau"
    plot_map(results, 1, 1, str(FIG_DIR / "cwave_map_k1_phi1.png"), title)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _gen_kernels()
    _gen_certification()
    _gen_map()
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
