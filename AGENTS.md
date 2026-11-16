# AGENTS.md

Python 3.11+ library that computes **CWAVE parameters** from SAR ocean
cross-spectra, implementing the orthonormal kernel decomposition of
Schulz-Stellenfleth et al. (2007). Primary target environment: IFREMER HPC
(PBS/SLURM, Apptainer).

## Reference

- Scientific reference : Schulz-Stellenfleth, Konig, Lehner, J. Geophys. Res.,
  112, C03019, 2007 — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003970
- Repository specifications (API conservativeness, conventions, the `sqrt()`/`eta`
  bug fix) : [`repository_specifications.md`](repository_specifications.md)

## Commands

```bash
pip install -e ".[dev]"          # editable install + dev tools (use a venv first)
pre-commit install               # then run pre-commit hooks
pre-commit run --all-files       # run all linters/formatters
pytest -c pyproject.toml tests/  # unit tests
pytest -c pyproject.toml --cov=cwaves --cov-report=term-missing tests/  # + coverage
```

- CI (`.github/workflows/build.yml`) runs pre-commit (ruff, mypy, codespell) and
  the unit tests with coverage (>= 80 %). The PyPI publish is triggered by CalVer
  tags (see `.github/workflows/publish.yml`).
- Coverage gate : `fail_under = 80` in `pyproject.toml` (`[tool.coverage.report]`).
- All code comments, docstrings and variable names are written in **English**.

## Layout

- `cwaves/compute_cwaves.py` — **single entry point** `compute_cwaves` (any
  supported grid, half or full plane, `part` = modulus/real/imag/complex),
  `compute_cwave_parameters` (`xsar_slc` signature, delegates to
  `compute_cwaves`), `compute_kernel`, `gegenbauer_polynoms`,
  `harmonic_functions` (`parity` even/odd).
- `cwaves/grid.py` — grid detection and quadrature (`build_quadrature`):
  cartesian uniform/non-uniform (`k_rg`/`k_az`), polar linear/log (`k`/`phi`
  in radians), exact cell areas, half-plane edge handling, explicit errors on
  malformed inputs. The spectrum is never resampled.
- `cwaves/symmetrize_xspectra.py` — `symmetrize_xspectrum`.
- `cwaves/client.py` — CLI client (`cwaves`) : one input `--input-nc`, schema
  detected from the variables (`xspectra_*tau_Re/Im` cross-spectrum, or OCN WV
  `oswCartSpec` on its native sinh grid by default / `oswPolSpec` on its native
  polar grid with `--ocn-spectrum polar`); `--part complex` writes
  `cwave_params_re/_im`. Tiled (grouped) IW products are rejected with a
  pointer to the IW map script.
- `cwaves/scripts/iw_cwaves_map.py` — compute CWAVE on the IW subswaths (iw1/iw2/
  iw3) of a Sentinel-1 IW product and render a georeferenced map. Handles both the
  OCN IW measurement (`s1?-iw?-osw-*.nc`, native sinh grid, cycles/m → rad/m) and
  the SARWAVE L1B XSP (`l1b-s1?-iw?-xsp-*.nc`, uniform rad/m). Output shape follows
  `cwave_Nk` x `cwave_Nphi`. Map rendering needs the `scripts` extra (cartopy).
- `cwaves/scripts/plot_kernels.py` — render the CWAVE kernel basis in the
  (kx, ky) wavenumber domain (elliptic domain A, masked outside), as a grid of
  k_gp x phi_hf panels (console script `cwaves-kernels`) (roadmap task 12).
- `cwaves/default_config.yml` — default `cwave_*` configuration (roadmap task 6).
- `tests/` — unit tests (synthetic `xarray` data + the `assets/` fixture, no network).
- `assets/` — an ESA OCN WV Sentinel-1 measurement NetCDF fixture.
- `docs/` — Sphinx documentation (roadmap task 12).

## Code conventions

- Docstrings are Google-style.
- Linters : `ruff` (lint + format), `mypy` (strict-ish, config in `pyproject.toml`),
  `codespell`.
- The kernel implementation is **corrected** with respect to the historical
  `xsar_slc` reference : the Gegenbauer coefficient carries the missing
  `sqrt()` and `eta` is the exact Jacobian with the natural logarithm
  (see `repository_specifications.md` section 5). Do not "simplify" the `eta`
  expression back to the original buggy form.
- Kernels are **even** (pi-periodic harmonics, as in the 2007 paper): on the
  half plane (`k_rg >= 0`) the basis is complete; on a full plane the result is
  exactly the CWAVE of the centrally-symmetric part (identical for a Hermitian
  cross-spectrum). Odd harmonics (`part="imag"`/`"complex"`) encode the
  anti-symmetric imaginary part. See `repository_specifications.md` sections
  6, 9, 10 and `docs/limits.rst`. Do not reintroduce resampling (polar ->
  cartesian, sinh -> regular) nor cell-integrated kernels (evaluated, rejected).

## Packaging / release

- Version is derived from git tags via hatch-vcs — there is no version field to
  edit anywhere. Release = `git tag <calver>` (e.g. `2026.10.22`), push the tag.
  `publish.yml` builds and uploads to PyPI (or TestPyPI for `-test` tags).
- `_version.py` is generated at build time.

## Git

- Default branch is `main`. Never push directly to `main`; work on feature
  branches and open PRs (see `repository_specifications.md` section 1).
- Tag development branches with the LLM name (e.g. `feat/ifremer-shultz07-implementation`).
