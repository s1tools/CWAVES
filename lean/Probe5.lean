import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

set_option linter.style.header false

variable (m n t : ℝ)

-- Goal form: ∫ √c * sin m * (√c * cos(n-1)) = ∫ √c*√c * (sin m * cos(n-1))
-- i.e. integrand: √c * sin m * (√c * cos n)  ?=  √c*√c * (sin m * cos n)
example : Real.sqrt (2 / Real.pi) * Real.sin m * (Real.sqrt (2 / Real.pi) * Real.cos n) =
    (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) * (Real.sin m * Real.cos n) := by ring

-- Goal form 2 (factored): ∫ √c * sin m * (√c * cos n) = ∫ (2/π) * (sin m * cos n)
example : Real.sqrt (2 / Real.pi) * Real.sin m * (Real.sqrt (2 / Real.pi) * Real.cos n) =
    (2 / Real.pi) * (Real.sin m * Real.cos n) := by
  have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
    rw [← pow_two, Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]
  ring_nf
  rw [hsc2]
