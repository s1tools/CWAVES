"""Compare CWAVE parameters from a listing of SARWAVE L1B XSP products.

This script (case-study tool) takes a **listing** of SARWAVE L1B IW XSP
``.SAFE`` directories and, for every per-subswath cross-spectrum file
(``l1b-s1?-iw?-xsp-*.nc``), computes the CWAVE parameters with **this library**
(:func:`cwaves.scripts.iw_cwaves_map.compute_sarwave_subswath`). The product
itself already carries CWAVE parameters computed by the **historical**
``xsar_slc`` pipeline (variable ``cwave_params``, in a different way :
symmetrized, different normalisation), so the two are compared parameter by
parameter.

For a listing it produces, into ``--output-dir`` :

* **one georeferenced map per CWAVE parameter** (default 4 ``k_gp`` x 5
  ``phi_hf`` = 20 maps) of the *library* values, all subswaths colocated in
  geographic space (colour = subswath).
* **one scatter per CWAVE parameter** (20 scatters) of the *library* value
  (y) against the *xsar_slc* value (x) for every tile of every subswath of the
  listing, with the 1:1 line, Pearson r and mean bias annotated.

Usage::

    python -m cwaves.scripts.cwave_listing_compare \
        --input-listing <listing.txt> --output-dir <out_dir> \
        [--subswath iw1 iw2 iw3] [--group intraburst|interburst] [--lag 2] \
        [--pol VV|HH] [--cwave-nk 4] [--cwave-nphi 5]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from cwaves.scripts.iw_cwaves_map import (  # noqa: E402
    _sarwave_pol_index,
    _setup_logging,
    compute_sarwave_subswath,
)

logger = logging.getLogger("cwaves.cwave_listing_compare")

#: the three IW subswath identifiers
SUBSWATHS = ("iw1", "iw2", "iw3")
DEFAULT_NK, DEFAULT_NPHI = 4, 5


def read_listing(path: str) -> list[str]:
    """Return the list of ``.SAFE`` paths from a listing file (one per line)."""
    paths = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [p for p in paths if p and not p.startswith("#")]


def _xsp_files(safe_dir: str) -> dict[str, Path]:
    """Map ``{subswath: path}`` for the ``l1b-s1?-iw?-xsp-*.nc`` of a SAFE."""
    files: dict[str, Path] = {}
    for nc in Path(safe_dir).rglob("l1b-s1?-iw?-xsp-*.nc"):
        m = re.search(r"-iw(\d)-", nc.name)
        if m:
            files[f"iw{m.group(1)}"] = nc
    return files


def _read_old_cwave(
    nc_path: str, group: str, pol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the historical (``xsar_slc``) ``cwave_params`` of one XSP file.

    Returns:
        A tuple ``(lons, lats, params)`` where ``lons``/``lats`` have shape
        ``(tile_line, tile_sample)`` and ``params`` has shape
        ``(tile_line, tile_sample, k_gp, phi_hf)``.
    """
    with xr.open_dataset(nc_path, group=group) as ds:
        pol_idx = _sarwave_pol_index(ds, pol)
        cp = ds["cwave_params"].isel(pol=pol_idx)
        lons = ds["longitude"].values
        lats = ds["latitude"].values
        params = cp.values
    return lons, lats, params


def _plot_cwave_map(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_gp: int,
    phi_hf: int,
    output_png: str,
    title: str,
) -> None:
    """Render one georeferenced CWAVE map (all subswaths) for one parameter."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(11, 7), dpi=150)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)
    all_values: list[np.ndarray] = []
    for sub, (lons, lats, params) in results.items():
        vals = params[:, k_gp - 1, phi_hf - 1]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        lons_f, lats_f, vals_f = lons[finite], lats[finite], vals[finite]
        all_values.append(vals_f)
        ax.scatter(
            lons_f,
            lats_f,
            c=vals_f,
            s=14,
            cmap="viridis",
            edgecolors="none",
            label=sub,
            transform=ccrs.PlateCarree(),
        )
    if all_values:
        cat = np.concatenate(all_values)
        norm = Normalize(vmin=float(np.nanmin(cat)), vmax=float(np.nanmax(cat)))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"CWAVE  k_gp={k_gp}, phi_hf={phi_hf} (library)")
    ax.set_title(title)
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


def _plot_scatter(
    old: np.ndarray,
    new: np.ndarray,
    k_gp: int,
    phi_hf: int,
    output_png: str,
) -> dict[str, float]:
    """Render one old-vs-new scatter for one CWAVE parameter.

    Args:
        old: the historical (``xsar_slc``) values, 1-D.
        new: the library values, 1-D (same tile order).
        k_gp: the Gegenbauer index (1-based).
        phi_hf: the harmonic index (1-based).
        output_png: output PNG path.

    Returns:
        A dict with the comparison statistics (``n``, ``r``, ``bias``, ``std``).
    """
    finite = np.isfinite(old) & np.isfinite(new)
    x, y = old[finite], new[finite]
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan")
    bias = float(np.mean(y - x)) if x.size else float("nan")
    std = float(np.std(y - x)) if x.size else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.scatter(x, y, s=12, edgecolors="none", alpha=0.7, cmap="viridis", c=y)
    lo = min(float(x.min()), float(y.min())) if x.size else 0.0
    hi = max(float(x.max()), float(y.max())) if x.size else 1.0
    one_to_one = np.array([lo, hi])
    ax.plot(one_to_one, one_to_one, "k--", linewidth=1.0, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(rf"CWAVE $k_{{gp}}={k_gp},\ \varphi_{{hf}}={phi_hf}$  (xsar_slc)")
    ax.set_ylabel(rf"CWAVE $k_{{gp}}={k_gp},\ \varphi_{{hf}}={phi_hf}$  (cwaves lib)")
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="cwaves lib value")
    ax.set_title(
        f"xsar_slc vs cwaves lib  (k_gp={k_gp}, phi_hf={phi_hf})\n"
        f"n={x.size}  r={r:.3f}  mean(new-old)={bias:+.3f}  std={std:.3f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return {"n": int(x.size), "r": r, "bias": bias, "std": std}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map + compare CWAVE parameters over a listing of SARWAVE XSP products."
    )
    parser.add_argument(
        "--input-listing", required=True, help="path to the listing .txt"
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory for the output PNGs"
    )
    parser.add_argument(
        "--subswath",
        nargs="+",
        default=list(SUBSWATHS),
        choices=list(SUBSWATHS),
        help="subswaths to process (default iw1 iw2 iw3)",
    )
    parser.add_argument(
        "--group",
        choices=["intraburst", "interburst"],
        default="intraburst",
        help="cross-spectra group (default intraburst)",
    )
    parser.add_argument(
        "--lag", type=int, default=2, help="cross-spectrum lag (default 2)"
    )
    parser.add_argument("--pol", default="VV", help="polarization (default VV)")
    parser.add_argument(
        "--max-safe", type=int, default=None, help="limit the number of SAFEs"
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
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "comma-separated (k_gp,phi_hf) cases to render, e.g. '1,1;2,2;4,5'. "
            "Default : the full Nk x Nphi grid."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    _setup_logging(False)
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    cwave: dict[str, float] = {}
    for src, dst in (
        ("cwave_wlmax", "cwave_wlmax"),
        ("cwave_wlmin", "cwave_wlmin"),
        ("cwave_nk", "cwave_Nk"),
        ("cwave_nphi", "cwave_Nphi"),
    ):
        if getattr(args, src) is not None:
            cwave[dst] = getattr(args, src)
    n_k = cast(int, cwave.get("cwave_Nk", DEFAULT_NK))
    n_phi = cast(int, cwave.get("cwave_Nphi", DEFAULT_NPHI))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safes = read_listing(args.input_listing)
    if args.max_safe is not None:
        safes = safes[: args.max_safe]
    logger.info("processing %d SAFEs", len(safes))

    # results[sub] = (lons 1d, lats 1d, params (N, k_gp, phi_hf))  -- library
    map_results: dict[str, list[np.ndarray]] = {s: [] for s in args.subswath}
    # compare[(kg, ph)] = [old 1d, new 1d]  -- pooled over all tiles/subswaths
    old_acc: dict[tuple[int, int], list[np.ndarray]] = {}
    new_acc: dict[tuple[int, int], list[np.ndarray]] = {}

    for i, safe in enumerate(safes):
        if not os.path.isdir(safe):
            logger.warning("missing SAFE, skipped: %s", safe)
            continue
        files = _xsp_files(safe)
        for sub in args.subswath:
            if sub not in files:
                continue
            nc = str(files[sub])
            try:
                lons, lats, params = compute_sarwave_subswath(
                    nc, args.group, args.lag, args.pol, **cwave
                )
            except (KeyError, ValueError, OSError) as exc:
                logger.error("library compute failed on %s: %s", nc, exc)
                continue
            try:
                old_lons, old_lats, old_params = _read_old_cwave(
                    nc, args.group, args.pol
                )
            except (KeyError, ValueError, OSError) as exc:
                logger.error("old cwave_params failed on %s: %s", nc, exc)
                continue
            if old_params.shape[:2] != lons.shape:
                logger.error(
                    "tile-grid mismatch on %s: lib %s vs old %s, skipped",
                    nc,
                    lons.shape,
                    old_params.shape,
                )
                continue
            flat = params.reshape(-1, n_k, n_phi)
            flat_old = old_params.reshape(-1, n_k, n_phi)
            map_results[sub].append(
                np.column_stack([lons.ravel(), lats.ravel()]).astype(float),
            )
            map_results[sub].append(flat)
            for kg in range(1, n_k + 1):
                for ph in range(1, n_phi + 1):
                    old_acc.setdefault((kg, ph), []).append(flat_old[:, kg - 1, ph - 1])
                    new_acc.setdefault((kg, ph), []).append(flat[:, kg - 1, ph - 1])
        logger.info("processed SAFE %d/%d", i + 1, len(safes))

    # build the combined (lons, lats, params) per subswath for the maps
    combined: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sub in args.subswath:
        if not map_results[sub]:
            continue
        coords = np.concatenate(map_results[sub][0::2], axis=0)
        params = np.concatenate(map_results[sub][1::2], axis=0)
        combined[sub] = (coords[:, 0], coords[:, 1], params)
    if not combined:
        logger.error("no subswath could be processed")
        return 1

    stem = os.path.basename(os.path.normpath(args.output_dir))
    title_base = (
        f"CWAVE (library) - listing {stem}, {args.group}, {args.lag}tau, {args.pol}"
    )
    # (k_gp, phi_hf) cases to render : the full grid unless --cases is given
    if args.cases:
        cases = [
            (int(a), int(b)) for a, b in (c.split(",") for c in args.cases.split(";"))
        ]
    else:
        cases = [(kg, ph) for kg in range(1, n_k + 1) for ph in range(1, n_phi + 1)]

    stats: list[dict[str, float]] = []
    for kg, ph in cases:
        # 1. georeferenced map of the library value
        map_png = out_dir / f"cwave_map_k{kg}_phi{ph}.png"
        _plot_cwave_map(
            combined, kg, ph, str(map_png), f"{title_base}\nk_gp={kg}, phi_hf={ph}"
        )
        # 2. old-vs-new scatter
        old = np.concatenate(old_acc.get((kg, ph), []))
        new = np.concatenate(new_acc.get((kg, ph), []))
        scatter_png = out_dir / f"compare_cwave_k{kg}_phi{ph}.png"
        st = _plot_scatter(old, new, kg, ph, str(scatter_png))
        st.update({"k_gp": float(kg), "phi_hf": float(ph)})
        stats.append(st)
        logger.info(
            "k_gp=%d phi_hf=%d: n=%d r=%.3f bias=%+.3f std=%.3f",
            kg,
            ph,
            st["n"],
            st["r"],
            st["bias"],
            st["std"],
        )

    # write a small summary table
    summary = out_dir / "compare_summary.txt"
    with open(summary, "w") as fh:
        fh.write("k_gp  phi_hf  n      r        bias      std\n")
        for st in stats:
            fh.write(
                f"{st['k_gp']:<5.0f} {st['phi_hf']:<7.0f} {st['n']:<6.0f} "
                f"{st['r']:9.4f} {st['bias']:9.4f} {st['std']:9.4f}\n"
            )
    logger.info("wrote %d maps + %d scatters to %s", len(cases), len(cases), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
