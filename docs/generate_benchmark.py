"""Benchmark the CWAVE compute time as a function of the number of kernels.

Run from the repository root :

    python docs/generate_benchmark.py

It times :func:`cwaves.compute_cwave_parameters` on a realistic one-sided
cross-spectrum (a real ESA OCN WV tile resampled onto a regular grid) for a grid
of ``cwave_Nk`` (Gegenbauer) and ``cwave_Nphi`` (harmonic) values, and prints a
Markdown table that can be pasted into ``docs/performance.rst``.

The cost is dominated by the kernel build (``O(Nk * Nphi)``) and the projection
(``O(Nk * Nphi * n_grid^2)``); the table shows the resulting wall-clock time.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cwaves.compute_cwaves import compute_cwave_parameters  # noqa: E402
from cwaves.scripts.iw_cwaves_map import resample_sinh_to_regular  # noqa: E402

OCN_IW1_NC = REPO_ROOT / "assets" / "s1d-iw1-osw-vv-fixture.nc"
WLMAX, WLMIN = 600.0, 25.0
N_GRID = 128
N_REPEAT = 3

#: (cwave_Nk, cwave_Nphi) combinations to benchmark
GRID = [(nk, nphi) for nk in (2, 3, 4, 5, 6, 8) for nphi in (3, 5, 7, 9, 11)]


def _load_spectrum() -> xr.DataArray:
    """Load one real OCN tile resampled onto the regular one-sided grid."""
    kmax = 2.0 * np.pi / WLMIN
    with xr.open_dataset(OCN_IW1_NC, group="intraburst") as ds:
        kx = 2.0 * np.pi * ds["oswKx"].isel(oswAzSize=0, oswRaSize=0).values
        ky = 2.0 * np.pi * ds["oswKy"].isel(oswAzSize=0, oswRaSize=0).values
        spec = ds["oswCartSpecRe"].isel(oswAzSize=0, oswRaSize=0, oswLag=2) + 1j * ds[
            "oswCartSpecIm"
        ].isel(oswAzSize=0, oswRaSize=0, oswLag=2)
    return resample_sinh_to_regular(spec, kx, ky, kmax, N_GRID)


def _time_once(xs: xr.DataArray, nk: int, nphi: int) -> float:
    t0 = time.perf_counter()
    compute_cwave_parameters(
        xs, cwave_wlmax=WLMAX, cwave_wlmin=WLMIN, cwave_Nk=nk, cwave_Nphi=nphi
    )
    return time.perf_counter() - t0


def main() -> None:
    xs = _load_spectrum()
    print(
        f"input grid: {xs.sizes['freq_sample']} x {xs.sizes['freq_line']} "
        f"(one-sided range x two-sided azimuth), wl {WLMIN:.0f}-{WLMAX:.0f} m\n"
    )

    rows: list[tuple[int, int, float]] = []
    for nk, nphi in GRID:
        times = [_time_once(xs, nk, nphi) for _ in range(N_REPEAT)]
        med = statistics.median(times)
        rows.append((nk, nphi, med))
        print(f"Nk={nk} Nphi={nphi}: {med * 1000:8.1f} ms")

    # Markdown table (rows = Nk, columns = Nphi)
    nphis = sorted({nphi for _, nphi, _ in rows})
    nks = sorted({nk for nk, _, _ in rows})
    print("\n```markdown")
    print("| `cwave_Nk` \\ `cwave_Nphi` | " + " | ".join(f"{p}" for p in nphis) + " |")
    print("|" + "---|" * (len(nphis) + 1))
    for nk in nks:
        cells = []
        for nphi in nphis:
            med = next(t for k, p, t in rows if k == nk and p == nphi)
            cells.append(f"{med * 1000:.0f} ms")
        print(f"| {nk} | " + " | ".join(cells) + " |")
    print("```")


if __name__ == "__main__":
    main()
