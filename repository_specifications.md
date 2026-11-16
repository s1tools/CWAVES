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
| `compute_kernel` | `compute_kernel(krg, kaz, kmin, kmax, cwave_Nk=4, cwave_Nphi=5, save_cwave_kernel=False)` (+ additive `parity="even"`, section 9) |
| `gegenbauer_polynoms` | `gegenbauer_polynoms(x, nk, lbda=1.5)` |
| `harmonic_functions` | `harmonic_functions(x, nphi)` (+ additive `parity="even"`, section 9) |

New in this library (no `xsar_slc` counterpart): the single entry point
`compute_cwaves(spectrum, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5, part="modulus", **kwargs)`
(section 8); `compute_cwave_parameters` delegates to it.

### 3.2 Input/output contracts

- `symmetrize_xspectrum(xs, ...) -> xarray.DataArray | xarray.Dataset`
  - `xs` : complex **one-sided** cross-spectrum (positive wavenumbers only in range)
- `compute_cwave_parameters(xs, ...) -> xarray.DataArray` (renamed `cwave_params`)
  - `xs` : real or complex spectrum with its wavenumber coordinates (any grid
    accepted by `compute_cwaves`, section 8; the historical input -- a
    one-sided cartesian cross-spectrum -- is one of them)
  - `kwargs` : `save_cwave_kernel`, `part`; unrelated `xsar_slc` keywords are ignored
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

## 6. Input contract : half or full plane, symmetry (revised 2026-11-16)

This section fixes what `compute_cwaves` / `compute_cwave_parameters` integrate
and why. It replaces the 2026-11-09 version, which **rejected** full-plane
(symmetrized) inputs; that rejection rested on a correct computation but an
over-strong interpretation, corrected below.

### 6.1 What the paper does

The CWAVE parameters are the projections of the normalised SAR image variance
spectrum `P` (a periodogram, equation 32; unit mass, equation 26) onto
orthonormal functions `h_i` over the elliptic domain `A` (equations 24-25).
The angular coordinate of the paper spans an interval of length `pi`, so `A`
is a **half wavenumber plane** (in the library: `k_rg >= 0`, section 4).

### 6.2 Why the half plane is enough -- and what the full plane means

- The angular functions `{1, sin 2p phi, cos 2p phi}` are pi-periodic: they are
  orthonormal and **complete on any interval of length pi**. The kernels are
  therefore **even**: `K(-k) = K(k)`.
- On the **half plane** the basis represents any function given there.
- On the **full plane** the same kernels have norm `sqrt(2)` (Gram -> 2I, the
  2026-11-09 measurement) and span only **centrally-symmetric** functions. This
  is a normalisation and a symmetry statement, not an inconsistency: integrated
  over the full plane and divided by the full-plane mass, the kernels return
  **exactly** the CWAVE parameters of the centrally-symmetric part
  `(f(k) + f(-k)) / 2` (verified: polar full circle vs even part on the half
  circle, 4e-16).
- A SAR image is real, so every cross-spectrum (any lag) is **Hermitian**,
  `X(-k) = conj X(k)`: `|X|` and `Re X` are centrally symmetric, `Im X` is
  anti-symmetric. For these inputs the half plane holds all the information
  and the half-plane and full-plane results coincide (L1B XSP asset: 0.23 %,
  the residual being the last `k_rg` column dropped by `symmetrize_xspectrum`).
- A **directional** spectrum (180 deg ambiguity removed: WW3, OCN
  `oswPolSpec`) is not centrally symmetric; its paper CWAVE parameters are
  those of its centrally-symmetric part. The former polar path, which kept only
  `cos(phi) >= 0`, discarded the waves travelling towards negative range;
  it has been removed.

### 6.3 The input contract

- a real or complex `xarray.DataArray` with **either** the cartesian
  coordinates `k_rg`, `k_az` (rad/m) **or** the polar coordinates `k` (rad/m),
  `phi` (radians, `k_rg = k cos(phi)`), each 1-D along its own dimension;
- covering the half plane (`k_rg >= 0`, or `phi` in `[-pi/2, pi/2]`) or the
  full plane (`k_rg` symmetric around 0 to one cell, or the full circle);
- by default (`part="modulus"`, the paper) only `|f|` is projected; see
  section 9 for the complex extension.

Explicit errors (`ValueError` / `TypeError`) are raised for: not a
`DataArray`; no supported coordinate pair (or both); NaN, non-monotonic or
single-sample axes; `phi` spanning `>= 2 pi` (degrees); a partial plane
coverage (asymmetric negative `k_rg`, a `phi` arc crossing both half planes
without being the full circle, or only `k_rg < 0` directions); zero mass in
the annulus. A wavenumber axis varying along an extra dimension (one `k_rg`
per tile, as in the `xsar_slc` mocks) is processed slice by slice.

### 6.4 Half-plane edge

On a half-plane grid the cells are fitted to the `k_rg = 0` edge: the cell of a
`k_rg = 0` sample keeps only its `k_rg >= 0` half, and a grid starting at
`k_rg = dk` (SARWave L1B XSP, whose `k_rg = 0` column is dropped) has its first
cell extended to 0. Without this, the `[0, dk/2]` strip was missing from the
half-plane integral: this was the origin of the ~2.4 % "symmetrization" gap
reported in the 2026-11-09 version. The library values on the L1B XSP asset
change by 2.3 % accordingly, and now equal the full-plane (symmetrized) values.

### 6.5 Quadrature on the native grid (no uniform-grid requirement)

The projection is evaluated with a **point rule with exact cell areas** on the
input grid (module `cwaves/grid.py`): cell boundaries at the mid-points between
samples; cartesian area `dk_rg * dk_az`; polar area `(k_+^2 - k_-^2)/2 * dphi`
(geometric boundaries for a log-uniform `k`, periodic `phi` cells on the full
circle). The former requirement of a uniform cartesian grid (constant mean
spacing as cell area) is lifted: sinh grids (OCN `oswCartSpec`, IW tiles of
varying size) and polar grids are integrated as they are, **without
resampling**. Verified: one analytic spectrum on a fine cartesian grid, a
70 x 139 sinh grid and an 80 x 144 log-polar grid -> same parameters to 2e-6;
the cartesian uniform half-plane path reproduces the previous implementation to
1e-7 apart from the edge fix of section 6.4.

## 7. Continuum orthonormality (verified 2026-11-09)

On 2026-11-09 the orthonormality of the implemented kernel basis was verified
end to end (no library change was made during the investigation). The full
evidence, the method of every check (X1-X9) and the reproducible scripts are in
the published investigation note :

<https://cerweb.ifremer.fr/datarmor/dev_reports/html/libs/agrouaze/cwave/orthonormality_investigation.html>

### 7.1 The kernel as a product of orthonormal 1-D factors

The kernel is `K_{n,p} = G_n(alpha_k) * F_p(phi) * eta` with

```
tmp    = a1 * k_az^4 + a2 * k_az^2 + k_rg^2,   a1 = (gamma^2 - gamma^4)/(gamma^2 kmin^2 - kmax^2), a2 = (kmax^2 - gamma^4 kmin^2)/(kmax^2 - gamma^2 kmin^2)   (gamma = 2)
alpha_k = 2 log10(sqrt(tmp)) / log10(kmax/kmin) - 1   in [-1, 1]
phi    = arctan2(k_rg, k_az)   in [0, pi]   (one-sided range domain, section 6)
G_n    = coef(n) * C_n^(3/2)(alpha_k) * sqrt(1 - alpha_k^2),   coef(n) = sqrt((n + 3/2)/((n + 2)(n + 1)))   (paper equation B9)
F      = { 1/sqrt(pi),  sqrt(2/pi) sin 2p phi,  sqrt(2/pi) cos 2p phi }
eta^2  = 2 S / ((k_rg^2 + k_az^2) * tmp * ln(kmax/kmin)),   S = 2 a1 k_az^4 + a2 k_az^2 + k_rg^2   (paper equation B13)
domain A = { (k_rg, k_az) : kmin^2 <= tmp <= kmax^2, k_rg >= 0 },   kmin = 2*pi/600, kmax = 2*pi/25
```

### 7.2 Radial factor : the Gradshteyn-Ryzhik orthonormality (exact)

With the orthonormality integral of Gradshteyn & Ryzhik (7th ed., 2000, section
7.313),

```
I_n = integral_{-1}^{1} (1 - x^2) C_n^(3/2)(x)^2 dx  =  2 (n + 1)(n + 2)/(2 n + 3)   for every n >= 0  (n = 0: 4/3)
```

(G&R 7.313 with lambda = 3/2: `pi 2^(1-2 lambda) Gamma(n + 2 lambda) / (n! (n + lambda) Gamma(lambda)^2)`.)
Equivalently `sqrt(1 - x^2) C_n^(3/2)(x)` is proportional to the associated
Legendre function `P^1_{n+1}(x)` (G&R 7.112, norm `2 l (l+1)/(2 l+1)` with
`l = n+1`): the radial functions `G_n` are exactly the normalised `P^1_{n+1}`
(checked to 2e-15, 2026-11-16).

the code's coefficient satisfies `coef(n)^2 * I_n = 1` **exactly for every n**.
Verified : the polynomials of the code's recurrence match
`scipy.special.gegenbauer` to 9.2e-14 (X1), the integrals match the analytic
values to machine precision (X2, 64-node Gauss-Legendre, exact for the
degree-2n polynomial integrands), and the radial block
`R_nm = integral G_n G_m W(alpha) dalpha` with `W = mean_phi(eta^2 J) ~ 1`
deviates from the identity by <= 2e-4 (12x12, X6).

### 7.3 Angular factor : pi-periodic harmonics

`F` is orthonormal and complete on **any interval of length pi** ; the
half-plane range domain gives `phi in [0, pi]` exactly. On a zero-centered grid
(`phi in [-pi, pi]`) the set remains orthogonal but its norm is doubled
(Gram -> 2I) and it spans only the centrally-symmetric spectra; integrating
over the full plane with the full-plane mass therefore yields the CWAVE
parameters of the centrally-symmetric part (section 6.2). The odd (pi-
antiperiodic) set of section 9 spans the anti-symmetric part.

### 7.4 The Jacobian identity eta^2 * J = 1 (exact)

The Jacobian `J = |d(k_rg, k_az) / d(alpha_k, phi)|` of the (alpha, phi) ->
(k_rg, k_az) mapping is analytic :

```
J = (k_rg^2 + k_az^2) * tmp * ln(kmax/kmin) / (2 S),   S = 2 a1 k_az^4 + a2 k_az^2 + k_rg^2
```

and therefore `eta^2 * J = 1` **identically** (verified without finite
differences on a 1201 x 2401 (alpha, phi) grid : max |eta^2 J - 1| =
8.9e-16, X9a ; independently confirmed by central finite differences at +/-1.2 %
accuracy, X5). `eta` (paper equation B13) is exactly the square root of the
inverse Jacobian : the 2-D measure on A becomes the plain d(alpha_k) d(phi).

### 7.5 Consequence : continuum orthonormality is an exact identity

With the change of variables (k_rg, k_az) -> (alpha_k, phi) the kernel
orthonormality factorizes over the orthonormalities of 7.2 and 7.3 :

```
integral_A K_i K_j dA  =  integral_B G_i G_j F_p F_q eta^2 J d(alpha_k) d(phi)  =  delta_ij
```

**exactly in the continuum** (`B = [-1, 1] x [0, pi]`). No implementation
defect can move this statement ; every discrete deviation measured so far is
pure quadrature :

- **uniform k-grid** : first-order quadrature, dominated by the elliptic domain
  boundary cutting through the cells — the idempotence error halves per grid
  doubling (in-span round trip 84 % -> 11 %, X8 ; refinement 3.5e-2 -> 1.7e-2
  -> 8.6e-3, X3/X9b), and the Gram matrix converges to I (1.5e-1 -> 2.4e-2
  diagonal deviation, X3).
- **(alpha, phi)-mapped grid** (the decomposition's native coordinates) : no
  domain boundary is cut through cells, the error shrinks ~4x per doubling
  (1.1e-2 -> 2.8e-3, X9b). The mapped integral of J reproduces the k-grid mask
  area to 0.2 % (0.053758 vs 0.053867, X9c).
- **product grids that cut the kernel domain** (k_rg max < kmax, i.e. the SAR
  XSP products) : the discrete Gram converges to a matrix that is not the
  identity (range-truncation floor) and the idempotence error plateaus instead
  of halving — a property of the product range, not of the implementation (see
  the "Part 2" section of the certification report).

Corollary : the former absolute `1e-7` coefficient-idempotence gate of the
certification report was **removed** — it is a continuum-limit property,
unreachable on any finite product grid (~10^6x too coarse for a first-order
quadrature). The report now gates the **proportionality of the reconstruction
error to the theoretical quadrature error** : the idempotence error must halve
(or better) per grid doubling on full-domain grids (ratio <= 0.75 per
doubling), which is the discrete signature of a consistent first-order
quadrature of the exact continuum basis verified in this section.

## 8. Single entry point and client (2026-11-16)

Answer to the PR #5 review (generic library, no product a priori, no
misleading entry points):

- **Library**: one entry point, `compute_cwaves(spectrum, ...)`, grid and
  coverage detected from the coordinates (section 6). `compute_cwave_parameters`
  keeps the `xsar_slc` signature and delegates to it.
- **Client**: one function, `compute_cwaves_from_file(path, lag=2, pol=0,
  **cwave)`, and one CLI, `cwaves --input-nc <file>` (`--lag`, `--pol`,
  `--part`, `--cwave-*`, `--output-nc`). The schema is detected from the
  variables of the file, never from its name: `xspectra_{lag}tau_Re/Im` on
  `k_rg`/`k_az` -> one parameter set per `time`; OCN WV files -> by default
  the cross-spectrum `oswCartSpec` on its native sinh grid
  (`ocn_cartesian_spectrum`, section 8.1), or `oswPolSpec` on `oswK`/`oswPhi`
  (degrees converted, native polar grid) with `ocn_spectrum="polar"` /
  `--ocn-spectrum polar`. Files whose spectra
  are stored in NetCDF groups (tiled IW products) are rejected with a pointer
  to `cwaves.scripts.iw_cwaves_map`, which processes them tile by tile on the
  native (sinh or uniform) grid; its output size follows `cwave_Nk x
  cwave_Nphi`.
- **Removed** (introduced earlier in this PR, never released, hence no
  deprecation period under section 3.3): `compute_cwave_parameters_polar`,
  `compute_cwaves_from_l1b`, `compute_cwaves_from_osw`,
  `compute_cwaves_from_osw_polar`, `polar_to_cartesian` (issue #11),
  `resample_sinh_to_regular`, the CLI `cwaves-osw` and its flags
  `--input-nc-osw / --input-nc-l1b / --polar-direct / --n-range /
  --n-azimuth`, and `--n-grid` of the IW script.
- The IFREMER sinh -> linear decompressor (`cwaves/scripts/ocn_sinh.py`) is
  **removed** (2026-11-16, issue #12): see section 8.1.

### 8.1 OCN WV cartesian grid: calibration (issue #12, 2026-11-16)

The OCN WV `oswCartSpecRe/Im` (half plane) is stored on a sinh grid whose axes
`oswKx` / `oswKy` (unit `m/m`) are normalised wavenumbers. Checked on 1800
S1A/S1C/S1D WV products (`tmp_data/cwave_s8/`):

- the axes follow exactly `K = a * 32 * sinh(kappa / 32)` (residual 3e-8;
  growth 1/32) and `oswJx` / `oswJy` are exactly `dK/dkappa`;
- `oswKx / oswGroundRngSize` and `oswKy / oswAziSize` give a central step equal
  to `2 pi / (1024 * ground pixel)` on both axes (ratio 1.00000), i.e. `m/m` =
  rad/m x estimation-area size (no 2 pi factor: rejected by the data).

The library reads the spectrum on that native grid
(`cwaves.client.ocn_cartesian_spectrum`) and integrates with exact cell areas.
The removed decompressor had four defects: hard-coded growth 1/30.1 (k too
large, up to ~+9 % at 2 pi/25 m), range scale from `owiIncidenceAngle[0]`
instead of `oswIncidenceAngle` (-2.8 %), azimuth scale averaged with the wrong
growth (-4.5 %), and a 256-point linear grid that cuts the CWAVE domain
(|k| <~ 0.21 / 0.19 rad/m < kmax). Once corrected (growth 1/32, product scale,
600 points) it reproduces the native-grid parameters to 0.1-0.7 % (~2 % in a
few cases, sinc interpolation), so decompression brings nothing.
OCN **IW** groups store physical axes (no area size); the IW script assumes
cycles/m (x 2 pi); a peak-wavelength check (median 133 m vs 838 m in rad/m)
supports it but is not conclusive (open point in issue #12).

## 9. Complex extension: odd harmonics (2026-11-16)

`harmonic_functions(x, nphi, parity="odd")` gives the pi-antiperiodic set
`{sqrt(2/pi) cos(m phi), sqrt(2/pi) sin(m phi)}, m = 1, 1, 3, 3, 5, ...` --
orthonormal and complete on `[0, pi]`, odd under `k -> -k`. The radial factor
`G_n` and `eta` are unchanged: the extension stays within the Gradshteyn &
Ryzhik construction (radial part) and the plain Fourier basis (angular part;
even and odd sets together = the full Fourier basis of the circle).

`compute_cwaves(..., part=...)`:

| `part` | projected | harmonics |
|--------|-----------|-----------|
| `"modulus"` (default, paper) | `abs(f)` | even |
| `"real"` | `Re f` | even |
| `"imag"` | `Im f` | odd |
| `"complex"` | `Re f` + 1j `Im f` | even + 1j odd |

All are normalised by the mass of `abs(f)`. Rationale: `Im X` of a Hermitian
cross-spectrum changes sign across the `k_rg = 0` edge, which the even set
cannot follow (Gibbs): on a smooth synthetic spectrum the imaginary part is
reconstructed to 4e-5 (33 odd harmonics) vs 0.26 (33 even). On the real L1B
XSP asset `Im X` is 8-15 % of `Re X` and noise / out-of-span dominated (both
bases reach ~0.5 relative L2), but the imaginary parameters scale with the lag
(2-tau / 1-tau slope 1.96-1.99, correlation >= 0.996; real parameters change by
2-3 %): they carry the propagation information of the cross-spectrum phase.
The CLI writes complex parameters as `cwave_params_re` / `cwave_params_im`
(NetCDF has no complex type).

## 10. Limits and implementation choices (2026-11-16)

The user-facing version is `docs/limits.rst`. Summary, with the evidence
(scripts and logs under `/scratch/agrouaze/opencode_sandbox/tmp_data/cwave_s{1,2,3,4}/`):

1. **Not a bijection**: a projection onto `Nk x Nphi` functions; the
   orthogonal complement is lost; the loss vanishes only in the continuum
   limit (complete basis).
2. **Even part only** (paper parameters): directional information requires the
   odd harmonics (`part="imag"/"complex"`).
3. **Finite elliptic domain**: energy outside `A` is in the mass but not
   represented; `G_n` vanish on the `kmin` / `kmax` ellipses; product grids
   that do not cover `A` (L1B WV XSP: `abs(k_az) <= 0.057` rad/m) cut it
   (discrete Gram deviation ~0.39 already at 4 x 5).
4. **Unit-mass normalisation**: energy level (Hs-like information) is not in
   the parameters.
5. **Quadrature error** grows with the kernel order (U-shaped round trip on a
   fixed grid, section 7).
6. **Cell-integrated kernels evaluated, not adopted**: integrating the kernels
   exactly over each cell (spectrum constant per cell) is less accurate on
   smooth spectra (max relative error 1.2e-3 to 2.3e-3 vs 7e-5 to 1.7e-3 for
   the point rule, 4 x 5 to 32 x 30), does not bring the discrete Gram closer to
   I, changes the real-product round trip by < 1.5 % up to 32 x 30 (0.160 ->
   0.145 at 32 x 30) and costs 50-100 x more. What mattered was the exact
   cell **areas** (section 6.5).
7. **Reconstruction** (`cwaves.reconstruct_spectrum`, `cwaves/reconstruct.py`,
   added 2026-11-16). Integrated **because it reduces the reconstruction
   loss**: on 62 real spectra (S1A L1B XSP 1/2-tau, S1A/S1D OCN WV polar, WW3
   polar, 55 Envisat L1C) x 4 bases (4x5 to 24x24), with the parameters of
   `compute_cwaves` and no other information (no mass), the maximum-entropy
   solution `g = m exp(sum l_i K_i)` (flat prior, Newton on the convex dual,
   parameters reproduced to 1e-8 relative) is closer to the spectrum than the
   linear `sum c_i K_i` clipped to >= 0 in 99.6 % of the 245 non-degenerate
   cases (median relative-L2 gain 8 %; OCN polar 0.19 -> 0.055 and L1B XSP
   0.29 -> 0.18 at 16 x 16; Envisat 0.49 -> 0.46). It adds a prior
   (positivity, maximum entropy), not information. With noisy parameters
   (`sigma`, Tikhonov at 10 x the noise, best of 1-10 x) the gain vanishes from
   ~16 x 16 on (5 % noise: 8 x 9 ratio 0.92, 24 x 24 ratio 0.99). Only
   `part="modulus"` parameters are reconstructed by MEM; `method="linear"`
   handles all parts. Evaluation scripts: `tmp_data/cwave_s6/`.
8. **Memory**: `compute_kernel` materialises `Nk * Nphi` kernels on the whole
   grid (1301^2 grid x 48 x 40 basis ~ 26 GB); product grids (~1e4 samples)
   are far from this limit.
