.. _limits:

Limits and implementation choices
=================================

This page states, in one place, **what the CWAVE decomposition can and cannot
do**, and **which choices the implementation makes**. Each point gives the
reason and, where relevant, the measured magnitude. The mathematical background
is in :ref:`theory`.

Limits of the decomposition
---------------------------

1. **It is a projection, not a bijection.** The map spectrum -> :math:`N_k
   \times N_\varphi` coefficients is linear and onto a finite-dimensional
   subspace: every component of the spectrum orthogonal to that subspace gives
   the same coefficients. No method can recover that component *from the
   coefficients alone*; the basis is complete, so the loss vanishes only when
   :math:`N_k, N_\varphi \to \infty` (continuum).

2. **Only the centrally-symmetric part is encoded by the paper's parameters.**
   The kernels are even. For a SAR cross-spectrum (Hermitian) the modulus and
   the real part are centrally symmetric, so nothing is lost by working on the
   half plane. A **directional** spectrum (180 degree ambiguity removed) is
   reduced to :math:`(f(\mathbf{k})+f(-\mathbf{k}))/2`: the propagation
   direction is not in the paper's parameters. The imaginary part of a
   cross-spectrum carries it: use ``part="imag"`` or ``"complex"`` (odd
   harmonics, :ref:`complex-cwave`).

3. **Finite elliptic domain.** The kernels live on the domain :math:`A`
   (:math:`k_{min} \le` elliptic wavenumber :math:`\le k_{max}`, compressed in
   azimuth). Spectral energy outside :math:`A` counts in the normalisation mass
   but is not represented. The radial functions vanish on the two boundary
   ellipses (:math:`G_n \propto \sqrt{1-\alpha_k^2}`), so energy close to
   :math:`k_{min}` (long swell, :math:`\lambda \approx 600` m by default) or
   :math:`k_{max}` converges slowly. A product whose wavenumber grid does not
   cover :math:`A` (e.g. SARWave L1B WV XSP, :math:`|k_{az}| \le 0.057`
   rad/m) cuts the domain: the discrete kernels are then far from orthonormal
   (Gram deviation ~0.39 at 4 x 5) whatever the quadrature.

4. **Normalisation to unit mass (paper equation 26).** The parameters do not
   depend on the spectrum's energy level: :math:`H_s`-like information is not
   in the coefficients and must be carried separately (the paper uses
   :math:`\sigma^0` and other auxiliary inputs in the regression). Reconstructing
   a spectrum in physical units requires that mass.

5. **Discrete grids add a quadrature error.** On a finite grid the discrete
   kernels are only approximately orthonormal (Gram = I + O(h)); the error
   grows with the order of the kernels, which oscillate faster than the grid
   resolves, especially near :math:`k_{min}` where the logarithmic radial
   coordinate is compressed onto few cartesian cells. Hence the U-shaped
   round-trip error vs. number of coefficients on a fixed grid (see
   :ref:`certification`): increasing :math:`N` beyond the optimum
   :math:`N^*(h)` does not help unless the grid is refined.

How far can a spectrum be reconstructed?
----------------------------------------

:func:`cwaves.reconstruct_spectrum` offers two reconstructions on any grid.
The plain **linear** one, :math:`\hat f = \sum_i c_i K_i` (on the full plane
:math:`\tfrac12 \sum_i c_i K_i`), has negative lobes and Gibbs oscillations.
The **maximum-entropy** one (``method="mem"``, the default) is the positive
density that reproduces the CWAVE parameters *exactly* and maximises
:math:`-\int \hat f(\log \hat f - 1)`: it respects the decomposition by
construction and needs no training and no information other than the
parameters. It was added to the library because it **reduces the
reconstruction loss**: on 62 real spectra x 4 bases (245 non-degenerate cases)
it is closer to the spectrum than the linear reconstruction (even clipped to
positive values) in 99.6 % of the cases. Median relative L2 error on the
kernel domain:

.. list-table::
   :header-rows: 1
   :align: center

   * - basis
     - OCN WV polar, linear
     - OCN WV polar, MEM
     - L1B XSP :math:`|X|`, linear
     - L1B XSP :math:`|X|`, MEM
     - Envisat L1C, linear
     - Envisat L1C, MEM
   * - 4 x 5
     - 0.64
     - 0.35
     - 0.57
     - 0.45
     - 0.58
     - 0.54
   * - 8 x 9
     - 0.37
     - 0.21
     - 0.41
     - 0.30
     - 0.53
     - 0.50
   * - 16 x 16
     - 0.19
     - 0.055
     - 0.29
     - 0.18
     - 0.49
     - 0.46
   * - 24 x 24
     - 0.064
     - 0.030
     - 0.19
     - 0.14
     - 0.50
     - 0.42

(S1A OCN WV polar spectrum: the truth is its centrally-symmetric part; L1B
XSP: 4 S1A cross-spectra; Envisat: 55 ASAR WV L1C cross-spectra.) The gain is
large on peaked, well-sampled spectra and small on noisy ones (Envisat).

With **noisy** parameters (e.g. predicted by a model) the constraints must be
relaxed (``sigma`` = noise standard deviation; Tikhonov at 10 x ``sigma``):
the gain then shrinks with the basis size and vanishes from about 16 x 16 on
(5 % noise: error ratio MEM / linear 0.92 at 8 x 9, 0.99 at 24 x 24); use few
coefficients or ``method="linear"`` there.

**Alternatives evaluated and not adopted** (same protocol, same 246 cases,
all reproducing the parameters exactly, no training):

.. list-table::
   :header-rows: 1
   :align: center

   * - method
     - result vs. maximum entropy (flat prior)
   * - power family :math:`\hat f = \max(\lambda\cdot K, 0)^p` (:math:`p=1`:
       minimum-norm positive; :math:`p = 2, 4`)
     - always worse; closer to MEM as :math:`p` grows (MEM is the limit
       :math:`p \to \infty`)
   * - MEM with a parametric prior (1-3 symmetrised Gaussian partitions fitted
       to the parameters)
     - worse (e.g. OCN polar 16 x 16: 0.10-0.13 vs 0.05)
   * - MEM with a prior outside the kernel span (:math:`\eta^2`,
       :math:`k^{-2}`, :math:`k^{-4}`)
     - < 1 % change (a prior inside the span gives the MEM result exactly)
   * - Rényi / Burg family :math:`\hat f = (-(\theta_0 + \lambda\cdot
       K))^{-r}` (peakier than MEM, which is :math:`r \to \infty`; domain
       mass taken from the MEM solution)
     - :math:`r = 4`: better in 76 % of the cases but by 0.4 % in median
       (up to -15 % on L1B XSP and WW3 at 8 x 9), worse on OCN polar spectra
       (+7 % median, up to +21 %); :math:`r = 2`: up to x2 on OCN (spikes
       at the domain boundary, where all kernels vanish)

No training-free method was found that is uniformly better than the flat-prior
maximum entropy, which is therefore kept as the default.

What the maximum-entropy step adds is a prior (positivity, maximum entropy),
not information: details orthogonal to the basis are plausible, not measured.
Learned priors (optimal transport on a library of spectra, conditional
generative models) follow the same rule -- they must reproduce the
coefficients exactly -- and trade a better prior for training data, compute
and a risk of hallucinated detail.

Implementation choices
----------------------

* **Single entry point**, :func:`cwaves.compute_cwaves`: no assumption on the
  product; the grid type (cartesian uniform / non-uniform, polar linear / log)
  and the plane coverage (half / full) are detected from the coordinates, and
  malformed inputs (degrees instead of radians, NaN coordinates, partial plane
  coverage...) raise explicit errors. :func:`cwaves.compute_cwave_parameters`
  keeps the ``xsar_slc`` signature and delegates to it.
* **No resampling**: point quadrature with exact cell areas on the native grid
  (:ref:`any-grid`). Resampling a spectrum (polar to cartesian, sinh to linear)
  adds an interpolation error and was removed.
* **Kernel sampled at the grid points** (point rule). Integrating the kernels
  exactly over each cell (spectrum constant per cell) was evaluated and **not
  adopted**: on a smooth, well-resolved spectrum it is *less* accurate (bias
  of order :math:`(h/\sigma)^2/24`, e.g. :math:`2\cdot10^{-3}` vs
  :math:`4\cdot10^{-4}` at 16 x 16), it does not bring the discrete Gram
  closer to the identity, it changes the real-product round trip by less than
  1.5 % up to 32 x 30, and it costs 50-100 x more.
* **Half-plane edge**: cells are clipped / extended to :math:`k_{rg} = 0`. This
  changes the parameters of grids without a :math:`k_{rg} = 0` column (SARWave
  L1B XSP) by ~2.3 % with respect to the previous version, and makes the
  half-plane result equal to the full-plane (symmetrized) one.
* **OCN WV spectra on their native sinh grid**: the cartesian
  ``oswCartSpecRe/Im`` of an OCN WV product is read by
  :func:`cwaves.client.ocn_cartesian_spectrum` with ``k_rg =
  oswKx / oswGroundRngSize`` and ``k_az = oswKy / oswAziSize`` (rad/m; the
  central step equals :math:`2\pi/(1024 \times` ground pixel)) and integrated
  with exact cell areas. Decompressing it to a linear grid was evaluated and
  removed: the previous decoder had four calibration defects (growth 1/30.1
  instead of 1/32, wrong incidence angle, biased azimuth scale, 256-point grid
  cutting the domain) and, once corrected, it reproduces the native-grid
  parameters to 0.1-0.7 %, so it brings nothing (issue #12).
* **OCN IW axes in rad/m**: the per-tile ``oswKx`` / ``oswKy`` of the OCN IW
  ``intraburst`` / ``interburst`` groups are used as is (no :math:`2\pi`
  factor). Against a colocated L1B IW XSP (175 tiles) the log-amplitude
  correlation is 0.68 with rad/m and -0.17 with cycles/m, confirmed on two
  more acquisitions (0.67 / -0.13 and 0.71 / 0.36). The best per-axis scale
  of a log-shape fit varies between acquisitions (azimuth 0.42, 0.71, 1.00):
  it reflects processing differences (azimuth cut-off, noise floor), not a
  calibration error, and the spectral peaks agree in azimuth (issue #12).
* **Modulus by default** (``part="modulus"``) as in the paper; the complex
  extension is opt-in.
* **Wavelength bounds** 25-600 m and basis 4 x 5 by default
  (:file:`cwaves/default_config.yml`), as in the ``xsar_slc`` production.
* **Memory**: the kernels are materialised on the whole grid
  (:math:`N_k N_\varphi \times n_{grid}` values): a 1301 x 1301 grid with a
  48 x 40 basis needs ~26 GB. Product grids (~10^4 samples) are far from this
  limit.
