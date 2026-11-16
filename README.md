# CWAVES

[![PyPI version](https://img.shields.io/pypi/v/cwaves)](https://pypi.org/project/cwaves/)
[![Build / tests / linters](https://github.com/s1tools/CWAVES/actions/workflows/build.yml/badge.svg)](https://github.com/s1tools/CWAVES/actions/workflows/build.yml)
[![Test coverage](https://codecov.io/gh/s1tools/CWAVES/branch/main/graph/badge.svg)](https://codecov.io/gh/s1tools/CWAVES)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`cwaves` is a Python library to compute **CWAVE parameters** from ocean wave
and SAR spectra (cartesian or polar grids, half or full plane, real or
complex). It implements the orthonormal kernel decomposition described in :

> J. Schulz-Stellenfleth, T. Konig, and S. Lehner, "An empirical approach for the
> retrieval of integral ocean wave parameters from synthetic aperture radar data",
> **J. Geophys. Res., 112, C03019, 2007.**
> [doi:10.1029/2006JC003970](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003970)

The CWAVE method retrieves integral ocean wave parameters from SAR data without the
intermediate step of retrieving the 2D ocean wave spectrum : the spectrum is
projected onto a set of orthonormal kernels built as tensor products of
Gegenbauer polynomials (wavenumber direction, the orthonormalisation of
Gradshteyn & Ryzhik 7.313) and harmonic functions (angle direction), weighted by
the Jacobian of the wavenumber mapping. Read the
[limits and implementation choices](docs/limits.rst) before use.

## Installation

```bash
pip install cwaves
```

From source :

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import cwaves

# spectrum: xarray.DataArray with coordinates k_rg / k_az (rad/m, cartesian,
# uniform or not) or k (rad/m) / phi (radians, polar); half or full plane
params = cwaves.compute_cwaves(spectrum, cwave_Nk=4, cwave_Nphi=5)
params_c = cwaves.compute_cwaves(spectrum, part="complex")  # Re -> even, Im -> odd harmonics
```

Command line (schema detected from the file's variables):

```bash
cwaves --input-nc s1a_wv1_xsp.nc --output-nc cwave.nc
```

## Public API

| function | description |
|----------|-------------|
| `cwaves.compute_cwaves(spectrum, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, part="modulus")` | **single entry point**: CWAVE parameters of a spectrum on any supported grid |
| `cwaves.compute_cwave_parameters(xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs)` | `xsar_slc`-compatible signature, delegates to `compute_cwaves` |
| `cwaves.compute_cwaves_from_file(path, lag=2, pol=0, **cwave)` | CWAVE of the spectrum stored in a NetCDF file (CLI `cwaves`) |
| `cwaves.compute_kernel(krg, kaz, kmin, kmax, cwave_Nk=4, cwave_Nphi=5, save_cwave_kernel=False, parity="even")` | compute the orthonormal CWAVE kernels |
| `cwaves.symmetrize_xspectrum(xs, dim_range='freq_sample', dim_azimuth='freq_line')` | symmetrize a one-sided cross-spectrum |
| `cwaves.gegenbauer_polynoms(x, nk, lbda=1.5)` | Gegenbauer polynomials `C_n^(3/2)` |
| `cwaves.harmonic_functions(x, nphi, parity="even")` | pi-periodic (even) or pi-antiperiodic (odd) harmonic functions |

## Axis convention

`kx = range` (`k_rg`, dimension `freq_sample`), `ky = azimuth` (`k_az`, dimension
`freq_line`). The major axis of the kernel ellipses is in the **range** direction.

## Corrected kernel implementation

With respect to the historical `xsar_slc` reference, this library implements the
**corrected** orthonormal kernels : the Gegenbauer coefficient carries the missing
`sqrt()` and `eta` is the exact Jacobian of the wavenumber mapping with the natural
logarithm. See [`repository_specifications.md`](repository_specifications.md)
(section 5) for the full investigation.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
pytest -c pyproject.toml --cov=cwaves --cov-report=term-missing tests/
```

See [AGENTS.md](AGENTS.md) for the full development guide.

## License

MIT — see [LICENSE](LICENSE).
