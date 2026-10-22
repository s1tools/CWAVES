# CWAVES Repository Specifications

- **Repository** : https://github.com/s1tools/CWAVES
- **Author** : Antoine Grouazel
- **Created** : 22 oct 2026

## 1. Git Governance

### 1.1 Protection of the `main` branch

- The `main` branch is **protected** : no direct pushes are allowed.
- Any change to `main` goes exclusively through a merged pull request (PR).
- The protection rules are configured on the GitHub side (branch protection rules) :
  - `Require pull request before merging` enabled
  - `Require status checks to pass` enabled (linters, unit tests, coverage)
  - `Restrict who can push` : only maintainers can merge

### 1.2 Pull request review

- **Every PR must be reviewed before being merged.**
- At least **1 approval** from a reviewer other than the PR author is required.
- The CI checks (linters, unit tests, coverage, doc build) must be green before the merge.
- The review covers :
  - compliance with the specifications of this document (signatures, conventions)
  - code quality (linters, type hints, docstrings)
  - test coverage of new features

### 1.3 Commit strategy

- One development branch per feature (`feat/...`) or per fix (`fix/...`).
- One local commit per completed step, with a descriptive message.
- No push nor PR opening before the end of all tasks of the branch.

## 2. Reference document for the CWAVE computation

The scientific reference for the CWAVE computation is :

> **J. Schulz-Stellenfleth, T. König, and S. Lehner, "An empirical approach for the retrieval of integral ocean wave parameters from synthetic aperture radar data", J. Geophys. Res., 112, C03019, 2007.**
> DOI : [10.1029/2006JC003970](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003970)

The CWAVE method is an empirical approach to retrieve integral ocean wave parameters from SAR data without the intermediate step of retrieving the 2D ocean wave spectrum. This paper defines (see in particular Appendix B, equations B1 to B13) :

- an elliptic wavenumber domain `A` shaped by the bunching parameter `g` (default `g = 2`), which takes into account the velocity bunching effect leading to a low-pass filtering of the spectrum in the azimuth direction
- the mapping of `A` onto the rectangle `B = [-1, 1] x [-pi/2, pi/2]`, with a logarithmic mapping of the wavenumber magnitude (equation B1)
- orthonormal functions built as tensor products of Gegenbauer polynomials `C_n^(3/2)` (in the wavenumber direction, equations B7 to B10) and pi-periodic harmonic functions (in the angle direction, equation B11), weighted by a Jacobian-like factor `h` (equation B13)
- the CWAVE parameters obtained by integrating the normalized cross-spectrum over each kernel

Any modification of the computation equations must be justified against this paper and documented in the Sphinx documentation.

## 3. Method signatures (conservativeness with respect to `xsar_slc`)

The signatures of the public methods of the lib **must remain identical** to the reference implementation of `xsar_slc` (commit [`14f80b4`](https://github.com/umr-lops/xsar_slc/commit/14f80b4663d82efe40d2243b65f02a835500d3d7), file `xsarslc/processing/xspectra.py`). The goal is that any user code of `xsar_slc` can migrate to `cwaves` without modifying its calls.

### 3.1 Reference signatures

| Method | Signature |
|---------|-----------|
| `symmetrize_xspectrum` | `symmetrize_xspectrum(xs, dim_range='freq_sample', dim_azimuth='freq_line')` |
| `compute_cwave_parameters` | `compute_cwave_parameters(xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, **kwargs)` |
| `compute_kernel` | `compute_kernel(krg, kaz, kmin, kmax, cwave_Nk=4, cwave_Nphi=5, save_cwave_kernel=False)` |
| `gegenbauer_polynoms` | `gegenbauer_polynoms(x, nk, lbda=1.5)` |
| `harmonic_functions` | `harmonic_functions(x, nphi)` |

### 3.2 Input/output contracts

- `symmetrize_xspectrum(xs, ...) -> xarray.DataArray | xarray.Dataset`
  - `xs` : complex **one-sided** cross-spectrum (positive wavenumbers only in range)
- `compute_cwave_parameters(xs, ...) -> xarray.DataArray` (renamed `cwave_params`)
  - `xs` : complex one-sided cross-spectrum
  - `kwargs` : arguments passed to `compute_kernel` (e.g. `save_cwave_kernel`)
- `compute_kernel(krg, kaz, kmin, kmax, ...) -> xarray.Dataset`
  - contains `cwave_kernel`, `Gegenbauer_polynoms`, `Harmonic_functions`, `eta`
- `gegenbauer_polynoms(x, nk, lbda) -> xarray.DataArray`
- `harmonic_functions(x, nphi) -> np.ndarray | xarray.DataArray`

### 3.3 API evolution rules

- Any signature evolution is **additive** (new optional parameter) or goes through a deprecation period (`DeprecationWarning`) of at least one release.
- Renaming or removing an existing parameter is forbidden without a discussion in the PR and an update of this document.
- Any divergence with respect to the reference signatures must be documented here with its justification.

## 4. Conventions

- **Axis convention** : `kx = range` (`k_rg`, dimension `freq_sample`), `ky = azimuth` (`k_az`, dimension `freq_line`). The major axis of the kernel ellipses is in the range direction. This convention is identical to the one of the last commits of `xsar_slc`.
- Note : in the paper (equation B1), the 4th-order bunching term is in the azimuth direction ; in the code it is applied to `k_az`.
- The units of the wavelengths (`cwave_wlmax`, `cwave_wlmin`) are in meters ; the wavenumbers `kmin`/`kmax` are in rad/m (`k = 2*pi/wavelength`).
- All code comments, docstrings and variable names in the lib are written in **English**.
