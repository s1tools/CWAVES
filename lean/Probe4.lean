import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

set_option linter.style.header false

noncomputable def F (m : ℕ) (t : ℝ) : ℝ :=
  if m = 1 then 1 / Real.sqrt Real.pi
  else if Even m then Real.sqrt (2 / Real.pi) * Real.sin (m * t)
  else Real.sqrt (2 / Real.pi) * Real.cos ((m - 1 : ℕ) * t)

variable (n : ℕ) (hn : 1 < n) (hodd : Odd n)

-- What does dsimp do to F n t (odd branch)?
#reduce F n 0

-- Is Nat.cast_sub a simp lemma (does dsimp rewrite ↑(n-1) -> ↑n - 1)?
example (h : 1 ≤ n) : (↑(n - 1) : ℝ) = (↑n - 1 : ℝ) := by
  dsimp

example (h : 1 ≤ n) : (↑(n - 1) : ℝ) = (↑n - 1 : ℝ) := by
  norm_num
