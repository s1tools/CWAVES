.. _proofs:

Formal verification (Lean)
==========================

The angular part of the CWAVE kernel basis is **formally verified in the Lean
4 theorem prover** (Mathlib). The source lives in the :file:`lean/` directory
of the repository; the build is reproducible with ``lake build`` (Lean
toolchain ``v4.35.0``, Mathlib ``v4.35.0``) and currently compiles with
**0 errors, 0 warnings, no ``sorry`` and no axioms**.

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

Scope
-----

The verification covers the **angular (harmonic) factor only**. The radial
factor — the orthonormal Gegenbauer polynomials :math:`C_n^{(3/2)}` and the
Jacobian :math:`\eta` over the elliptic domain :math:`A` (paper equations
B1-B13) — is **not** part of the formal statement; the full 2D orthonormality
of the kernel basis on :math:`A` is therefore not yet machine-checked.

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

Reproducing the verification
----------------------------

.. code-block:: bash

   cd lean
   lake build

The build is incremental (the Mathlib precompile cache is stored in
:file:`.lake/`, which is gitignored). The two source files are :

* :file:`lean/CWave/Definitions.lean` — the kernel factors (Gegenbauer,
  :math:`G`, :math:`F_m`, :math:`\eta`) mirroring the Python implementation;
* :file:`lean/CWave/Harmonics.lean` — the lemmas and the
  ``angular_orthonormal`` theorem above.
