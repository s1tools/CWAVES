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

- `cwaves/compute_cwaves.py` — `compute_cwave_parameters`, `compute_kernel`,
  `gegenbauer_polynoms`, `harmonic_functions`.
- `cwaves/symmetrize_xspectra.py` — `symmetrize_xspectrum`.
- `cwaves/client.py` — CLI client (`cwaves-osw`) computing CWAVE on a one-sided
  (half) cross-spectrum (roadmap task 9). Two inputs : the SARWave L1B XSP
  cross-spectrum product (`--input-nc-l1b`, primary) and the ESA OCN WV
  Sentinel-1 measurement file (`--input-nc-osw`, polar `oswPolSpec` -> cartesian).
- `cwaves/scripts/iw_cwaves_map.py` — compute CWAVE on the IW subswaths (iw1/iw2/
  iw3) of a Sentinel-1 IW product and render a georeferenced map. Handles both the
  OCN IW measurement (`s1?-iw?-osw-*.nc`, sinh grid → resampled to regular, `m/m`
  → rad/m) and the SARWAVE L1B XSP (`l1b-s1?-iw?-xsp-*.nc`, uniform rad/m). Map
  rendering needs the `scripts` extra (cartopy) (roadmap task 10).
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
- `compute_cwave_parameters` is **fully compliant with the 2007 paper** : it
  requires a **one-sided (half, `k_rg >= 0`)** cross-spectrum, uses only the
  real modulus `|xs|`, and computes **without symmetrization** (on the one-sided
  range domain). A full/symmetrized spectrum is rejected with a `ValueError`
  (see `repository_specifications.md` section 6). Do not reintroduce the
  internal symmetrization.

## Packaging / release

- Version is derived from git tags via hatch-vcs — there is no version field to
  edit anywhere. Release = `git tag <calver>` (e.g. `2026.10.22`), push the tag.
  `publish.yml` builds and uploads to PyPI (or TestPyPI for `-test` tags).
- `_version.py` is generated at build time.

## Git

- Default branch is `main`. Never push directly to `main`; work on feature
  branches and open PRs (see `repository_specifications.md` section 1).
- Tag development branches with the LLM name (e.g. `feat/ifremer-shultz07-implementation`).
