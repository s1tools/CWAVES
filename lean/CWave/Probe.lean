import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
namespace CWave
open Real Set MeasureTheory
example : ∫ t in (-π / 2)..(π / 2), sin (2 * t) = 0 := by
  have hderiv : deriv (fun t : ℝ => -cos (2 * t) / 2) = (fun t => sin (2 * t)) := by
    ext t
    norm_num [deriv_neg, HasDerivAt.div_const]
    · apply (HasDerivAt.cos _).div_const
      · apply (HasDerivAt.sin (fun t => 2 * t)).const_mul
        · norm_num
        · exact deriv_id t
      · norm_num
    · norm_num
  haveI : IntervalIntegrable (fun t : ℝ => sin (2 * t)) volume (-π / 2) (π / 2) := by
    apply ContinuousOn.intervalIntegrable
    fun_prop
  rw [integral_deriv_eq_sub' (fun t : ℝ => -cos (2 * t) / 2)]
  · simpa using hderiv
  · norm_num
  · norm_num
  · fun_prop
end CWave
