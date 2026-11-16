"""Benchmark the CWAVE compute time as a function of the number of kernels.

Run from the repository root :

    python docs/generate_benchmark.py

It times :func:`cwaves.compute_cwaves` on a realistic one-sided
cross-spectrum (a real ESA OCN IW tile on its native sinh grid) for a grid
of ``cwave_Nk`` (Gegenbauer) and ``cwave_Nphi`` (harmonic) values, and prints a
Markdown table that can be pasted into ``docs/performance.rst``.

The cost is dominated by the kernel build (``O(Nk * Nphi)``) and the projection
(``O(Nk * Nphi * n_samples)``); the table shows the resulting wall-clock time.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cwaves.compute_cwaves import compute_cwaves  # noqa: E402
from cwaves.scripts.iw_cwaves_map import ocn_tile_spectrum  # noqa: E402

OCN_IW1_NC = REPO_ROOT / "assets" / "s1a-iw1-osw-vv-fixture.nc"
WLMAX, WLMIN = 600.0, 25.0
N_REPEAT = 3

#: (cwave_Nk, cwave_Nphi) combinations to benchmark
GRID = [(nk, nphi) for nk in (2, 3, 4, 5, 6, 8) for nphi in (3, 5, 7, 9, 11)]


def _load_spectrum() -> xr.DataArray:
    """Load one real OCN IW tile on its native (sinh) grid."""
    with xr.open_dataset(OCN_IW1_NC, group="intraburst") as ds:
        return ocn_tile_spectrum(ds, 0, 0, 2).load()


def _time_once(xs: xr.DataArray, nk: int, nphi: int) -> float:
    t0 = time.perf_counter()
    compute_cwaves(
        xs, cwave_wlmax=WLMAX, cwave_wlmin=WLMIN, cwave_Nk=nk, cwave_Nphi=nphi
    )
    return time.perf_counter() - t0


def main() -> None:
    xs = _load_spectrum()
    print(
        f"input grid: {xs.k_rg.size} x {xs.k_az.size} native sinh grid "
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
