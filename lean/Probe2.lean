import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

set_option linter.style.header false

namespace Probe2
open Real
variable (m n : ℕ)

def F (m : ℕ) (t : ℝ) : ℝ :=
  if m = 1 then 1 / Real.sqrt Real.pi
  else if Even m then Real.sqrt (2 / Real.pi) * Real.sin (m * t)
  else Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t)

lemma F_even_def (m : ℕ) (hm : Even m) (hm1 : ¬ m = 1) :
    F m = fun t => Real.sqrt (2 / Real.pi) * Real.sin (m * t) := by
  funext t
  simp [F, hm, hm1]

-- Test the hR form with apply integral_congr + ring
example (hmE : Even m) (hnE : Even n) (hm1 : ¬m = 1) (hn1 : ¬n = 1) :
    (∫ t in (-π / 2)..(π / 2), F m t * (Real.sqrt (2 / Real.pi) * Real.sin (n * t))) =
    (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
    (∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * Real.sin (n * t)) := by
  rw [F_even_def m hmE hm1]
  have hR : (∫ t in (-π / 2)..(π / 2),
      (Real.sqrt (2 / Real.pi) * Real.sin (m * t)) *
      (Real.sqrt (2 / Real.pi) * Real.sin (n * t))) =
      (∫ t in (-π / 2)..(π / 2),
        (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
        (Real.sin (m * t) * Real.sin (n * t))) := by
    apply intervalIntegral.integral_congr
    intro t _
    ring
  rw [hR]
  exact intervalIntegral.integral_const_mul _ _

-- Test rw [F_even] directly
example (hmE : Even m) (hm1 : ¬m = 1) (t : ℝ) : F m t = Real.sqrt (2 / Real.pi) * Real.sin (m * t) := by
  rw [F_even_def m hmE hm1]

-- nlinarith with hsc2 for the integrand (alternative)
example (t : ℝ) :
    (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) * (Real.sin (m * t) * Real.sin (n * t)) =
    (Real.sqrt (2 / Real.pi) * Real.sin (m * t)) * (Real.sqrt (2 / Real.pi) * Real.sin (n * t)) := by
  have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
    nlinarith [show Real.sqrt (2 / Real.pi) ^ 2 = 2 / Real.pi by
      field_simp [Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]]
  nlinarith [hsc2]

end
