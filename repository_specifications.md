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

- **Axis convention** : in this lib `kx = range` (`k_rg`, dimension `freq_sample`), `ky = azimuth` (`k_az`, dimension `freq_line`). The major axis of the kernel ellipses is in the range direction. This convention is identical to the one of the last commits of `xsar_slc`.
- Note : in the paper (equation B1), the 4th-order bunching term is on the azimuth wavenumber. With the lib axis convention this means the bunching term must be applied to `k_az`. (In the paper's own notation the bunched axis is `kx` and the other is `ky`, i.e. paper `kx` = lib azimuth `k_az`, paper `ky` = lib range `k_rg`.)
- The units of the wavelengths (`cwave_wlmax`, `cwave_wlmin`) are in meters ; the wavenumbers `kmin`/`kmax` are in rad/m (`k = 2*pi/wavelength`).
- All code comments, docstrings and variable names in the lib are written in **English**.

## 5. Investigation : the missing `sqrt()` in the `xsar_slc` reference (task 5)

This section records the findings of the investigation on the `sqrt()` missing in the original `xsar_slc` implementation of `compute_cwaves.py` (reference commit [`14f80b4`](https://github.com/umr-lops/xsar_slc/commit/14f80b4663d82efe40d2243b65f02a835500d3d7), first introduced in commit [`064d3da`](https://github.com/umr-lops/xsar_slc/commit/064d3da) "adding cwave and imacs", never corrected since). Three related issues were found ; all three are corrected in the reviewed Ifremer reference (`cwaves_review_cls.py`), which confirms the analysis.

### 5.1 The missing `sqrt()` : the Gegenbauer coefficient

The orthonormal functions of the paper (equations B7 to B9) are built from the Gegenbauer polynomials `C_n^(3/2)` normalized to **unit norm** on `[-1, 1]` with weight `(1 - x^2)`. The normalization coefficient is :

```
coef_n = sqrt( (n + 3/2) / ((n + 2) (n + 1)) )
```

The `xsar_slc` code uses the coefficient **without the square root** :

```python
# xsar_slc / original CWAVES compute_cwaves.py  (BUG : missing sqrt)
coef = lambda nk : (nk + 3 / 2.) / ((nk + 2.) * (nk + 1.))
Gnk  = gegenbauer_polynoms(alpha_k, ik - 1, lbda=3 / 2.) * coef(ik - 1) * nu(alpha_k)
```

Numerically, `coef_n = 1 / h_n` (where `h_n = ∫ C_n^2 (1-x^2) dx`), so the code's `g_n` have norm `1/sqrt(h_n)` instead of 1. The corrected coefficient is `sqrt(coef_n) = 1/sqrt(h_n)`, which gives unit-norm `g_n` and matches the paper (B7-B9) and the reviewed reference exactly (e.g. `g1 = 0.5 sqrt(3) sqrt(1-ak^2)`, `g2 = 0.5 sqrt(15) ak sqrt(1-ak^2)`).

> This is the literal **missing `sqrt()`** of task 5. It does **not** break orthogonality (the `g_n` stay mutually orthogonal) ; it only scales each kernel, so the absolute CWAVE values are affected.

### 5.2 The `eta` (Jacobian) factor

The factor `eta` weights the kernels and must be the **Jacobian of the wavenumber mapping** of the paper (equation B13) so that the kernels are orthonormal on the domain `A`.

Original (buggy) :

```python
# xsar_slc / original CWAVES compute_cwaves.py
tmp = a1 * kaz**4 + a2 * kaz**2 + krg**2
eta = np.sqrt((2. * tmp) / ((kaz**2 + krg**2) * tmp * np.log10(kmax / kmin)))
#      the `tmp` cancels, leaving:
eta = np.sqrt(2.0 / ((kaz**2 + krg**2) * np.log10(kmax / kmin)))   # <-- BUG
```

Because the `tmp` in the numerator cancels the `tmp` in the denominator, the original code reduces to `sqrt(2 / (k^2 log10(kmax/kmin)))`, which is **not** the Jacobian of the mapping. The correct Jacobian (derived from the mapping of paper equation B1, with `S = 2 a1 k_az^4 + a2 k_az^2 + k_rg^2` and `u = a1 k_az^4 + a2 k_az^2 + k_rg^2`) is :

```python
tmp = a1 * kaz**4 + a2 * kaz**2 + krg**2
eta = np.sqrt((2. * (2*a1*kaz**4 + a2*kaz**2 + krg**2)) /
              ((kaz**2 + krg**2) * tmp * np.log(kmax / kmin)))
```

i.e. the numerator is `2 * S` (not `2 * tmp`) and the logarithm is the **natural** logarithm. The identity `J = 2 S / (k^2 u ln(kmax/kmin))` was verified numerically to be exact (ratio = 1.0000 everywhere on the domain).

### 5.3 `log10` vs natural logarithm (task 7)

- For `alpha_k` (equation B1) the log base **cancels** in the ratio `(log sqrt(u) - log kmin)/(log kmax - log kmin)` : `log10` and `ln` give the **same** `alpha_k`. No impact.
- For `eta` (equation B13) the log base does **not** cancel : using `log10` instead of `ln` rescales `eta` by the constant `sqrt(ln 10) ≈ 1.517` (i.e. `eta^2` by `ln 10 ≈ 2.30`). This does **not** break orthogonality but rescales **all** CWAVE parameters by the same constant. The paper uses the natural logarithm ; the `CWAVES` lib uses the natural logarithm (exact Jacobian).

### 5.4 Orthonormality validation (numerical)

The kernels were integrated on the domain `A` (paper parameters `g=2`, `kmin=2 pi/624 m`, `kmax=2 pi/60 m`, `Nk=4`, `Nphi=5`) and the orthonormality matrix was measured :

| implementation | off-diagonal (max) | orthonormal ? |
|----------------|:------------------:|:-------------:|
| original `xsar_slc` (`log10`, `tmp` cancelled, `coef` no sqrt) | **0.39** | **no** |
| corrected `eta` only | 2.0e-3 | yes (orthogonal) |
| corrected `eta` + `sqrt(coef)` | 8.0e-3 | yes (orthogonal) |

Conclusion : the original `xsar_slc` implementation produces **non-orthogonal** kernels (off-diagonal contamination of ~24-39 %) because the `tmp` term cancels in `eta`. The missing `sqrt()` in the Gegenbauer coefficient and the `log10` choice affect only the normalization (absolute values), not the orthogonality.

**Decision : the `CWAVES` lib implements the corrected version (section 5.1 + 5.2), non-configurable (no legacy option).** The method signatures remain identical to `xsar_slc` (section 3) ; only the internal computation of the `Gegenbauer` coefficient and of `eta` changes, which is a **bug fix** and not an API change. Consequently the CWAVE parameters differ from the uncorrected `xsar_slc` output.
