"""Tests for the CWAVE kernel plotting script (``cwaves.scripts.plot_kernels``).

Builds the kernels on a synthetic two-sided meshgrid (no fixture needed) and
checks the grid/mask/select helpers, the figure rendering and the CLI.
"""

import struct

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cwaves.scripts.plot_kernels import (
    _build_figure,
    build_kernel_grid,
    domain_mask,
    entrypoint,
    plot_kernels,
    select_kernels,
)

WLMAX = 600.0
WLMIN = 25.0


def _grid(n=40):
    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = build_kernel_grid(WLMAX, WLMIN, n)
    return krg, kaz, kmin, kmax, k_rg_1d, k_az_1d


def _kernel(n_k=4, n_phi=5, n=40):
    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = _grid(n)
    from cwaves.compute_cwaves import compute_kernel

    kernel = compute_kernel(
        krg=krg, kaz=kaz, kmin=kmin, kmax=kmax, cwave_Nk=n_k, cwave_Nphi=n_phi
    )
    return kernel, k_rg_1d, k_az_1d


# ---------------------------------------------------------------------------
# build_kernel_grid
# ---------------------------------------------------------------------------


def test_build_kernel_grid_shape_and_symmetry():
    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = _grid()
    assert krg.dims == ("i", "j")
    assert kaz.dims == ("i", "j")
    # the grid must be two-sided (symmetric) in both range and azimuth
    assert k_rg_1d[0] < 0 < k_rg_1d[-1]
    assert k_az_1d[0] < 0 < k_az_1d[-1]
    assert np.isclose(k_rg_1d[0], -k_rg_1d[-1])
    # the grid must extend beyond kmax so the ellipse is not clipped
    assert k_rg_1d.max() > kmax
    # kmin / kmax derive from the wavelengths
    assert np.isclose(kmin, 2 * np.pi / WLMAX)
    assert np.isclose(kmax, 2 * np.pi / WLMIN)
    # the range axis must be attached as a coordinate (for the plotting labels)
    assert "k_rg" in krg.coords


# ---------------------------------------------------------------------------
# domain_mask
# ---------------------------------------------------------------------------


def test_domain_mask_elliptic_shape():
    krg, kaz, kmin, kmax, k_rg_1d, k_az_1d = _grid()
    mask = domain_mask(krg, kaz, kmin, kmax)
    # the origin (0, 0) is outside the domain A (bounded by kmin in range)
    i_mid = krg.sizes["i"] // 2
    j_mid = kaz.sizes["j"] // 2
    assert not bool(mask.values[i_mid, j_mid])
    # the range mid (k_az=0, |k_rg| ~ kmax/2) is inside the domain
    i_range = int(np.argmin(np.abs(k_rg_1d - kmax / 2.0)))
    assert bool(mask.values[i_range, j_mid])
    # the far corner (largest |k_rg| and |k_az|) is outside
    assert not bool(mask.values[0, 0])
    # the domain must be elongated along range (k_rg)
    extent_i = int(mask.any(dim="j").values.sum())
    extent_j = int(mask.any(dim="i").values.sum())
    assert extent_i > extent_j


# ---------------------------------------------------------------------------
# select_kernels
# ---------------------------------------------------------------------------


def test_select_kernels_all():
    pairs, nrows, ncols = select_kernels(4, 5, None, None)
    assert len(pairs) == 20
    assert (nrows, ncols) == (4, 5)
    assert pairs[0] == (1, 1)
    assert pairs[-1] == (4, 5)


def test_select_kernels_single():
    pairs, nrows, ncols = select_kernels(4, 5, 2, 3)
    assert pairs == [(2, 3)]
    assert (nrows, ncols) == (1, 1)


def test_select_kernels_row():
    pairs, nrows, ncols = select_kernels(4, 5, 1, None)
    assert len(pairs) == 5
    assert (nrows, ncols) == (1, 5)


def test_select_kernels_out_of_range():
    with pytest.raises(ValueError, match="k_gp"):
        select_kernels(4, 5, 5, None)
    with pytest.raises(ValueError, match="phi_hf"):
        select_kernels(4, 5, None, 0)


# ---------------------------------------------------------------------------
# plot_kernels / entrypoint
# ---------------------------------------------------------------------------


def test_plot_kernels_full_grid(tmp_path):
    kernel, k_rg_1d, k_az_1d = _kernel(n_k=4, n_phi=5, n=60)
    krg, kaz, kmin, kmax, _, _ = _grid(60)
    mask = domain_mask(krg, kaz, kmin, kmax)
    pairs, _, _ = select_kernels(4, 5, None, None)
    out = tmp_path / "kernels.png"
    plot_kernels(
        kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, pairs, str(out), "all kernels"
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_kernels_single(tmp_path):
    kernel, k_rg_1d, k_az_1d = _kernel(n_k=4, n_phi=5, n=60)
    krg, kaz, kmin, kmax, _, _ = _grid(60)
    mask = domain_mask(krg, kaz, kmin, kmax)
    out = tmp_path / "k1_phi1.png"
    plot_kernels(
        kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, [(1, 1)], str(out), "k1 phi1"
    )
    assert out.exists()
    assert out.stat().st_size > 0
    # the figure must be rendered at 300 dpi : a 12 in wide figure is 3600 px
    # (bbox_inches="tight" trims the outer margins, so allow some headroom)
    with open(out, "rb") as fh:
        raw = fh.read(33)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG
    width, height = struct.unpack(">II", raw[16:24])
    assert width >= 2400 and height >= 1800
    # the publication style (serif font) must have been applied
    # (font.family is a list in recent matplotlib versions)
    assert plt.rcParams["font.family"] in ("serif", ["serif"])
    assert plt.rcParams["mathtext.fontset"] == "stix"


def test_plot_kernels_shades_half_domain():
    # the k_rg < 0 half (outside the integration domain A, one-sided in range)
    # must be shaded : a hatched Rectangle spanning [k_rg.min(), 0] plus a
    # 'k_x<0 : outside A' annotation on the first panel
    kernel, k_rg_1d, k_az_1d = _kernel(n_k=2, n_phi=2, n=30)
    krg, kaz, kmin, kmax, _, _ = _grid(30)
    mask = domain_mask(krg, kaz, kmin, kmax)
    fig = _build_figure(
        kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, [(1, 1)], "t", "RdBu_r"
    )
    ax = fig.axes[0]
    hatched = [
        p
        for p in ax.patches
        if isinstance(p, mpatches.Rectangle)
        and p.get_hatch()
        and np.isclose(p.get_xy()[0] + p.get_width(), 0.0)
        and p.get_xy()[0] < 0
    ]
    assert len(hatched) == 1  # exactly one k_rg<0 shading rectangle
    # the 'outside A' annotation must be present on the (single) panel
    assert any("outside" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_plot_kernels_empty_pairs(tmp_path):
    kernel, k_rg_1d, k_az_1d = _kernel(n_k=2, n_phi=2, n=20)
    krg, kaz, kmin, kmax, _, _ = _grid(20)
    mask = domain_mask(krg, kaz, kmin, kmax)
    with pytest.raises(ValueError, match="no kernel selected"):
        plot_kernels(
            kernel, k_rg_1d, k_az_1d, mask, kmin, kmax, [], str(tmp_path / "x.png"), "t"
        )


def test_entrypoint_full(tmp_path):
    out = tmp_path / "kernels.png"
    rc = entrypoint(["--n-grid", "60", "--output-png", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_entrypoint_single_index(tmp_path):
    out = tmp_path / "k1_phi1.png"
    rc = entrypoint(
        ["--k-gp", "1", "--phi-hf", "1", "--n-grid", "48", "--output-png", str(out)]
    )
    assert rc == 0
    assert out.exists()


def test_entrypoint_out_of_range():
    rc = entrypoint(["--k-gp", "99", "--n-grid", "24"])
    assert rc == 1


def test_entrypoint_with_cwave_options(tmp_path):
    # exercise the --cwave-* options (custom wavelength range and number of
    # basis functions)
    out = tmp_path / "custom.png"
    rc = entrypoint(
        [
            "--k-gp",
            "1",
            "--n-grid",
            "24",
            "--cwave-wlmax",
            "300",
            "--cwave-wlmin",
            "50",
            "--cwave-nk",
            "2",
            "--cwave-nphi",
            "3",
            "--output-png",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_entrypoint_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = entrypoint(["--k-gp", "1", "--phi-hf", "1", "--n-grid", "24"])
    assert rc == 0
    produced = list(tmp_path.glob("cwave_kernels_*.png"))
    assert len(produced) == 1
    assert produced[0].stat().st_size > 0
