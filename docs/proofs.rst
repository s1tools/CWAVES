.. _proofs:

Formal verification (Lean)
==========================

The angular and radial factors of the CWAVE kernel basis are **formally
verified in the Lean 4 theorem prover** (Mathlib). The source lives in the
:file:`lean/` directory of the repository; the build is reproducible with
``lake build`` (Lean toolchain ``v4.35.0``, Mathlib ``v4.35.0``) and currently
compiles with **0 errors, 0 warnings and no ``sorry``**; the theorems depend
only on the three standard axioms of Lean/Mathlib (``propext``,
``Classical.choice``, ``Quot.sound``, checked with ``#print axioms``).

What is proven
--------------

The angular harmonics :math:`F_m` — the angle-direction factor of every CWAVE
kernel, defined in ``CWave.Definitions`` exactly as the Python
:func:`cwaves.harmonic_functions` — form an **orthonormal basis** of
:math:`L^2([-\pi/2, \pi/2])` :

.. math::

   \int_{-\pi/2}^{\pi/2} F_m(t)\, F_n(t)\, \mathrm{d}t = 1_{m = n}
   \qquad (m, n \ge 1)

This is the statement ``CWave.angular_orthonormal`` in
``CWave.Harmonics``.

The radial functions :math:`G_n(x)\sqrt{1-x^2}` (paper B9, with
:math:`G_n = \sqrt{(n+3/2)/((n+1)(n+2))}\,C_n^{(3/2)}`, defined in
``CWave.Definitions`` as the Python :func:`cwaves.gegenbauer_polynoms`) are
**orthonormal on** :math:`[-1, 1]` :

.. math::

   \int_{-1}^{1} G_n(x)\sqrt{1-x^2}\; G_m(x)\sqrt{1-x^2}\,\mathrm{d}x
   = 1_{n = m} \qquad (n, m \ge 0)

This is ``CWave.radial_orthonormal_sqrt`` in ``CWave.Radial`` (with
``gegenbauer_orthogonal`` and ``gegenbauer_norm``:
:math:`\int (1-x^2)\,(C_n^{(3/2)})^2 = 2(n+1)(n+2)/(2n+3)`, Gradshteyn &
Ryzhik 7.313 with :math:`\lambda = 3/2`). It is the check of the
:math:`\sqrt{\cdot}` of the coefficient (issue #4): without it the
functions are not normalised.

Scope
-----

Both 1-D factors of the kernels are verified. The **change of variables**
from :math:`(k_{rg}, k_{az})` to the elliptic coordinates
:math:`(\alpha_k, \alpha_\varphi)` over the domain :math:`A` (paper
B1-B13, :math:`\eta^2 J = 1`) is **not** formalised: the 2-D orthonormality
on :math:`A` follows from the two 1-D results and this Jacobian identity,
which is checked numerically (to :math:`10^{-16}`) in the Python test suite,
not in Lean.

Structure of the proof
----------------------

The proof is a machine-checked reduction of the orthonormality to elementary
trigonometric integrals, in three layers :

1. **Frequency analysis.** Each harmonic :math:`F_m` carries a single
   frequency :math:`\nu(m)`, defined in ``CWave.Harmonics`` as ``freq`` :

   * :math:`\nu(1) = 0` (:math:`F_1 = 1/\sqrt{\pi}`),
   * :math:`\nu(m) = m` for even :math:`m \ge 2` (:math:`F_m =
     \sqrt{2/\pi}\,\sin(m t)`),
   * :math:`\nu(m) = m - 1` for odd :math:`m \ge 3` (:math:`F_m =
     \sqrt{2/\pi}\,\cos((m-1) t)`).

   All these frequencies are **even non-negative integers**, which is the
   key structural fact: they are mutually orthogonal on the symmetric interval
   :math:`[-\pi/2, \pi/2]`.

2. **Elementary integrals** on :math:`[-\pi/2, \pi/2]` :

   * :math:`\int 1 = \pi`,
   * :math:`\int \sin(k t)\,\mathrm{d}t = 0` for any non-zero integer
     :math:`k` (oddness of the integrand, via the substitution
     :math:`x = k t`),
   * :math:`\int \cos(k t)\,\mathrm{d}t = 0` for any non-zero **even**
     integer :math:`k` (because :math:`\sin(k\pi/2) = 0` for even
     :math:`k`).

3. **Product-to-sum lemmas.** The products of two harmonics are reduced with
   the identities :math:`2\cos a\cos b = \cos(a-b) + \cos(a+b)`,
   :math:`2\sin a\sin b = \cos(a-b) - \cos(a+b)` and
   :math:`2\sin a\cos b = \sin(a-b) + \sin(a+b)`, giving closed forms for
   :math:`\int \cos(a t)\cos(b t)`, :math:`\int \sin(a t)\sin(b t)` and
   :math:`\int \sin(a t)\cos(b t)` over the interval for even natural
   :math:`a, b` (diagonal :math:`\pi/2`, off-diagonal :math:`0`).

The main theorem ``angular_orthonormal`` then combines these lemmas over the
**eight parity cases** of :math:`(m, n)` (:math:`m = 1` / even / odd crossed
with :math:`n = 1` / even / odd, with the :math:`m = n` vs :math:`m \neq n`
sub-cases), using the normalization :math:`\sqrt{2/\pi} \cdot \sqrt{2/\pi} =
2/\pi` to turn each diagonal :math:`\pi/2` into :math:`1`.

Structure of the radial proof
-----------------------------

The proof is algebraic, on real polynomials (``P n : ℝ[X]`` follows the
three-term recurrence of ``Gegenbauer``; ``gegenbauer_eq_eval`` links the
two):

1. **Derivative relations** (joint induction on :math:`n`):
   :math:`(1-x^2) C_{n+1}' = -(n+1)\,x\,C_{n+1} + (n+3)\,C_n` and
   :math:`(1-x^2) C_n' = (n+3)\,x\,C_n - (n+1)\,C_{n+1}`.
2. **Sturm–Liouville form**:
   :math:`\big((1-x^2)^2 C_n'\big)' = -n(n+3)\,(1-x^2)\,C_n`.
3. **Orthogonality**: :math:`(\lambda_m-\lambda_n)(1-x^2)C_nC_m` is the
   derivative of :math:`(1-x^2)^2 (C_n'C_m - C_m'C_n)`, which vanishes at
   :math:`\pm 1`; the fundamental theorem of calculus gives 0 for
   :math:`n \ne m` (:math:`\lambda_n = n(n+3)` is strictly increasing).
4. **Norms**: multiplying the recurrence by :math:`(1-x^2)C_{n+2}` and by
   :math:`(1-x^2)C_{n+1}` and using the orthogonality gives
   :math:`(n+2)\,h_{n+2} = \frac{(2n+5)(n+4)}{2n+7}\,h_{n+1}`, with
   :math:`h_0 = 4/3`, :math:`h_1 = 12/5`.

Executable cross-check
----------------------

``CWave.Test`` mirrors the recurrence on exact rationals and checks at
compile time (``#guard``) the values :math:`C_n^{(3/2)}(1) = (n+1)(n+2)/2`,
and the exact weighted Gram matrix :math:`\int (1-x^2) C_n C_m` for
:math:`n, m < 8` (rational polynomial arithmetic); ``#eval`` prints its
diagonal :math:`[4/3, 12/5, 24/7, 40/9, \dots]`.

Reproducing the verification
----------------------------

.. code-block:: bash

   cd lean
   lake build

The build is incremental (the Mathlib precompile cache is stored in
:file:`.lake/`, which is gitignored; fetch it with ``lake exe cache get``).
The source files are :

* :file:`lean/CWave/Definitions.lean` — the kernel factors (Gegenbauer,
  :math:`G`, :math:`F_m`, :math:`\eta`) mirroring the Python implementation;
* :file:`lean/CWave/Harmonics.lean` — the lemmas and the
  ``angular_orthonormal`` theorem above;
* :file:`lean/CWave/Radial.lean` — the Gegenbauer recurrence, derivative
  relations, orthogonality, norms and ``radial_orthonormal_sqrt``;
* :file:`lean/CWave/Test.lean` — the ``#guard`` / ``#eval`` cross-checks.
