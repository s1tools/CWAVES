import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

set_option linter.style.header false

variable (m n t : ℝ)

let c : ℝ := Real.sqrt (2 / Real.pi)

-- The integrand rearrangement that ring fails on:
--   √c * sin m * (√c * cos n)  vs  √c * √c * (sin m * cos n)
example : c * Real.sin m * (c * Real.cos n) = c * c * (Real.sin m * Real.cos n) := by ring

-- with the c^2 = 2/pi fact (to factor out)
example : c * Real.sin m * (c * Real.cos n) = (2 / Real.pi) * (Real.sin m * Real.cos n) := by
  have hsc2 : c * c = 2 / Real.pi := by
    rw [← pow_two, Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]
  nlinarith [hsc2]
