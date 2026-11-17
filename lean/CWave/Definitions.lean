/-
Copyright (c) 2026 Antoine Grouazel. All rights reserved.
Released under the MIT license as described in the file LICENSE.
Authors: Antoine Grouazel
-/
module
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Int.Basic
public import Mathlib.Data.Nat.Basic
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# CWAVE kernel orthonormality — definitions
...
-/
namespace CWave
@[expose] public section

noncomputable def Gegenbauer (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | 1 => 3 * x
  | n + 2 =>
      ((2 * (n + 2) + 1) * x * Gegenbauer (n + 1) x - (n + 3) * Gegenbauer n x) /
        (n + 2)

noncomputable def G (n : ℕ) (t : ℝ) : ℝ :=
  Real.sqrt ((n + 3 / 2) / ((n + 2) * (n + 1))) * Gegenbauer n t

noncomputable def F (m : ℕ) (t : ℝ) : ℝ :=
  if m = 1 then 1 / Real.sqrt Real.pi
  else if Even m then Real.sqrt (2 / Real.pi) * Real.sin (m * t)
  else Real.sqrt (2 / Real.pi) * Real.cos ((m - 1 : ℕ) * t)

noncomputable def a1 (kmin kmax : ℝ) : ℝ := -12 / (4 * kmin^2 - kmax^2)
noncomputable def a2 (kmin kmax : ℝ) : ℝ := (kmax^2 - 16 * kmin^2) / (kmax^2 - 4 * kmin^2)
noncomputable def tmp (kmin kmax : ℝ) (kx ky : ℝ) : ℝ :=
  a1 kmin kmax * ky^4 + a2 kmin kmax * ky^2 + kx^2
noncomputable def eta (kmin kmax : ℝ) (kx ky : ℝ) : ℝ :=
  Real.sqrt (2 * (2 * a1 kmin kmax * ky^4 + a2 kmin kmax * ky^2 + kx^2) /
      ((kx^2 + ky^2) * tmp kmin kmax kx ky * Real.log (kmax / kmin)))
end
end CWave
