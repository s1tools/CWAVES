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

## 6. Input contract : one-sided (half) + real, no symmetrization (task 9)

The CWAVE parameters must be computed **fully compliant with the 2007 paper**, not
"conservatively" the way `xsar_slc` does it. This section fixes the input contract
of `compute_cwave_parameters` and, in particular, **why a symmetrized (full)
cross-spectrum is not an acceptable input**.

### 6.1 What the paper does

The 20 CWAVE parameters are the **projections of the SAR image variance spectrum
`P` onto the orthonormal functions `h_i`**, integrated over the elliptic wavenumber
domain `A` (equation 25):

```
S_i = ∫_A P(kx, ky) h_i(kx, ky) dkx dk y        (i = 1, ..., n_phi · n_k)
```

with two essential properties :

1. **`P` is a real, non-negative spectrum.** `P` is the **periodogram** of the
   (normalized) SAR imagette — "The periodogram of the wave mode imagette computed
   using an FFT algorithm is used as an estimate of P" (equation 32) — normalized to
   unit integral over `A` (equation 26). A periodogram is `|FFT|²` : **real and
   non-negative**. The paper **never** uses a complex cross-spectrum nor its phase.

2. **The integration domain `A` is a half wavenumber plane.** Equation B1 maps `A`
   onto the rectangle `B = [-1, 1] × [-π/2, π/2]`, with the angular coordinate
   `α_φ(kx, ky) = arctan(ky, kx) ∈ [-π/2, π/2]`. This range is only valid for
   `kx ≥ 0`, so `A` is **one-sided in the range wavenumber (`kx`) and two-sided in
   the azimuth wavenumber (`ky`)**. The orthonormality of the functions `h_i`
   (equation 24) is defined over this half domain `A`, **not** over the full plane.

### 6.2 The input contract of `compute_cwave_parameters`

`compute_cwave_parameters(xs, ...)` therefore requires :

- **`xs` : a one-sided (half) cross-spectrum** — positive wavenumbers only in range
  (`k_rg >= 0`), two-sided in azimuth (`k_az`). This is exactly the domain `A` of
  the paper (with the lib axis convention of section 4 : `kx = k_rg = range`,
  `ky = k_az = azimuth`).
- **Only the real, non-negative modulus `|xs|` is used** : the complex phase is
  discarded, matching the paper's real image variance spectrum. `xs` may be passed
  as a complex cross-spectrum (Re/Im) or directly as its real modulus.
- **`xs` must be sampled on a uniform (regular) wavenumber grid** in both `k_rg`
  (range) and `k_az` (azimuth) — see section 6.6.

The computation is performed **directly on the one-sided range domain, without any
symmetrization** (the `symmetrize_xspectrum` step that the `xsar_slc` reference
inserted before the projection has been **removed**).

### 6.3 Why a symmetrized (full) spectrum is not a good input

`xsar_slc` symmetrizes the one-sided input to a full two-sided spectrum before the
projection. This is a deviation from the paper and is rejected by this lib for the
following reasons :

1. **Wrong integration domain.** The paper integrates over the half domain `A`
   (equations 24, 25). Symmetrizing turns the integration domain into the full 2D
   plane, on which the kernels `h_i` are **not** orthonormal (their norm is doubled,
   since they are antipodally even). The resulting coefficients are therefore **not**
   the paper's `S_i`.

2. **Distorted first (DC) range-bin weighting.** A one-sided FFT spectrum carries a
   single DC bin (`k_rg = 0`) and the standard one-sided weighting. Symmetrization
   rebuilds a two-sided grid (`2n − 2` points) and re-weights the DC bin and the grid
   edges. Measured on a real SARWave L1B XSP product, computing on the symmetrized
   full grid instead of the one-sided half grid shifts the CWAVE parameters by
   **≈ 2.4 %** (relative). This is exactly the "specific weighting on the first bin
   in range" that symmetrization was introduced to avoid — i.e. it changes the result
   instead of preserving the paper's definition.

3. **Unnecessary.** The integrand is the real modulus `|xs|` (antipodally even for a
   real image) multiplied by kernels that are antipodally even (the angular basis is
   made of π-periodic harmonics, so `h_i(k_rg, k_az) = h_i(−k_rg, −k_az)`). The
   one-sided range domain `A` is therefore **complete** : the π-periodic harmonics
   already span the whole wave-direction range on `k_rg >= 0`. Symmetrization only
   appends the redundant mirrored half and buys nothing except the artifacts of
   points 1 and 2.

4. **Obscures the paper's structure.** The clean "orthonormal basis on `A`,
   coefficients = projections" formulation (equations 24–26) only holds when the
   integration is performed over `A`. Symmetrizing breaks that structure.

### 6.4 Hardening in the code

`compute_cwave_parameters` guards the input with `_check_one_sided(xs)` : if the
spectrum carries negative range wavenumbers (`k_rg.min() < 0`, i.e. a full /
symmetrized spectrum was passed) it raises a `ValueError` instead of silently
computing on the wrong domain. `symmetrize_xspectrum` is kept as a **standalone
utility** (its signature is part of section 3), but its output is **not** an
accepted input of `compute_cwave_parameters`.

### 6.5 Consequences

- The CWAVE parameters of this lib differ from the `xsar_slc` reference values by
  the ≈ 2.4 % of section 6.3 (the `xsar_slc` values were produced on the
  symmetrized full grid). This is **intentional** : the goal is compliance with the
  2007 paper, not agreement with the `xsar_slc` reference.
- The public method signatures are unchanged (section 3). The change is internal to
  `compute_cwave_parameters` (removal of the symmetrization step + the one-sided
  guard) and is a **paper-compliance fix**, not an API change.

### 6.6 The wavenumber grid must be uniform (regular) in `k_rg` and `k_az`

The projection integral (equation 25) is approximated by a **rectangle rule with a
single constant cell area**. In `compute_cwave_parameters` the cell area is computed
once as the **mean spacing** along each axis :

```python
dkrg = xxs.k_rg.diff(dim="freq_sample").mean(dim="freq_sample")
dkaz = xxs.k_az.diff(dim="freq_line").mean(dim="freq_line")
cwave = ((kernel * xxsmn) * dkrg * dkaz).sum(dim=["freq_sample", "freq_line"], ...)
```

The kernel `h_i` is **evaluated at the true grid coordinates** (`alpha_k` and
`alpha_phi` are functions of the actual `k_rg` / `k_az` values), so a non-uniform
grid is not rejected — but the **integration weight `dkrg · dkaz` is only exact when
the grid is uniform** in both axes :

- **Uniform grid** → `mean(diff)` equals the constant spacing → `dkrg · dkaz` is the
  exact cell area → the projection is correct.
- **Irregular grid** → the kernel is still at the right points, but the constant
  weight is only the *average* spacing → the CWAVE integral is **biased** (small for
  mild irregularity, large for a strongly non-linear grid, e.g. log-spaced). The
  computation does not fail; the result is simply wrong.

Therefore the input `xs` **must be sampled on a uniform wavenumber grid in both
`k_rg` (range) and `k_az` (azimuth)**. This is a natural, not extra, assumption :
the paper's `P` is the periodogram of the imagette on a regular FFT grid (equation
32), so a uniform wavenumber grid is exactly the sampling the method is defined for.

In practice the supported inputs all satisfy this :

| input | grid | uniform? |
|-------|------|:--------:|
| SARWave L1B XSP (`--input-nc-l1b`) | `k_rg` / `k_az` (rel. std of spacing ≈ 1e-6) | ✓ |
| OCN WV measurement (`--input-nc-osw`) | `polar_to_cartesian` resamples onto `np.linspace` (rel. std 0) | ✓ |

Note : the `kmin/kmax` frequency mask plus `dropna` (see section 6.2) keeps a
contiguous sub-block of the grid, which preserves uniformity. A non-uniform input
(e.g. the OCN product's own `oswKx`/`oswKy`, rel. std of spacing ≈ 1) **must not** be
fed directly to `compute_cwave_parameters` ; it must first be resampled onto a
uniform grid (as `polar_to_cartesian` does for the polar `oswPolSpec`).
