import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

set_option linter.style.header false

noncomputable def F (m : ℕ) (t : ℝ) : ℝ :=
  if m = 1 then 1 / Real.sqrt Real.pi
  else if Even m then Real.sqrt (2 / Real.pi) * Real.sin (m * t)
  else Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t)

lemma F_odd (m : ℕ) (hm : Odd m) (hm1 : m ≠ 1) :
    F m = fun t => Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t) := by
  ext t
  have hneE : ¬ Even m := by
    intro he
    rcases he with ⟨a, ha⟩
    rcases hm with ⟨b, hb⟩
    omega
  unfold F
  split_ifs with h1 hE
  · contradiction
  · contradiction
  · rfl

variable (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) (hmE : Even m) (hodd : Odd n) (hn1 : n ≠ 1)

-- Test the m-even / n-odd branch with the hR structure
example : ∫ t in (-π / 2)..(π / 2), F m t * F n t =
    (2 / Real.pi) * (∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * Real.cos ((n - 1) * t)) := by
  rw [F_odd n hodd hn1]
  dsimp
  have hR : (∫ t in (-π / 2)..(π / 2),
      (Real.sqrt (2 / Real.pi) * Real.sin (m * t)) *
      (Real.sqrt (2 / Real.pi) * Real.cos ((n - 1) * t))) =
      (∫ t in (-π / 2)..(π / 2),
        (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
        (Real.sin (m * t) * Real.cos ((n - 1) * t))) := by
    apply intervalIntegral.integral_congr
    intro t _
    ring
  rw [hR, intervalIntegral.integral_const_mul]
  have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
    rw [← pow_two, Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]
  rw [hsc2]
