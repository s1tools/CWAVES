module
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
public import CWave.Definitions

/-!
# CWAVE trigonormal harmonics — orthonormality (roadmap task 11, pillar F2)

The angular basis `F_m` (defined in `CWave.Definitions`, exactly as the Python
`harmonic_functions`) is orthonormal on `[-pi/2, pi/2]` :

```
∫_{-pi/2}^{pi/2} F_m(t) F_n(t) dt = 1_{m = n}
```

The frequency carried by `F_m` is :
- `0` for `m = 1` (the constant `1/sqrt(pi)`),
- `m` for even `m >= 2` (`sqrt(2/pi) sin(m t)`),
- `m - 1` for odd `m >= 3` (`sqrt(2/pi) cos((m-1) t)`).

All these frequencies are even nonnegative integers, and they are mutually
orthogonal on the symmetric interval `[-pi/2, pi/2]` (every cross term is an
integral of `cos(k t)` or `sin(k t)` for a nonzero even integer `k`, which
vanishes there). -/

namespace CWave
open Real Set MeasureTheory
public section

/-! ## Frequencies -/

/-- The angular frequency carried by `F_m` : `0` for `m = 1`, `m` for even
    `m`, and `m-1` for odd `m`. In all cases it is an even nonnegative integer. -/
def freq (m : ℕ) : ℕ := if m = 1 then 0 else if Even m then m else m - 1

lemma freq_one : freq 1 = 0 := by simp [freq]
lemma freq_even (m : ℕ) (hm : Even m) : freq m = if m = 1 then 0 else m := by
  simp [freq, hm]
lemma freq_odd (m : ℕ) (hm : Odd m) : freq m = if m = 1 then 0 else m - 1 := by
  rcases hm with ⟨k, rfl⟩
  by_cases h1 : 2 * k + 1 = 1
  · rw [h1]; simp [freq]
  · have hne : ¬ Even (2 * k + 1) := by
      intro he; rcases he with ⟨a, h⟩; omega
    simp [freq, h1, hne]

/-! ## Elementary integrals on `[-pi/2, pi/2]` -/

/-- `∫_{-pi/2}^{pi/2} 1 dt = pi`. -/
lemma integral_const_pi : ∫ _ in (-Real.pi / 2)..(Real.pi / 2), 1 = Real.pi := by
  simp
  ring

/-- `∫_{-pi/2}^{pi/2} sin(k t) dt = 0` for any nonzero integer `k` (sin is odd and
    the interval is symmetric about 0). -/
lemma integral_sin_even (k : ℤ) (hk : k ≠ 0) :
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.sin (k * t) = 0 := by
  have hkR : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  calc
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.sin (k * t)
        = (1 / (k : ℝ)) * ∫ x in (k : ℝ) * (-Real.pi / 2)..(k : ℝ) * (Real.pi / 2), Real.sin x := by
      rw [intervalIntegral.integral_comp_mul_left hkR (f := fun x : ℝ => Real.sin x)]
      simp [one_div]
    _ = (1 / (k : ℝ)) *
        (Real.cos ((k : ℝ) * (-Real.pi / 2)) - Real.cos ((k : ℝ) * (Real.pi / 2))) := by
      rw [integral_sin]
    _ = (1 / (k : ℝ)) * (Real.cos (k * (Real.pi / 2)) - Real.cos (k * (Real.pi / 2))) := by
      have hneg : (k : ℝ) * (-Real.pi / 2) = -((k : ℝ) * (Real.pi / 2)) := by ring
      rw [hneg, Real.cos_neg]
    _ = 0 := by ring

/-- `sin(k * pi/2) = 0` when `k` is even : `k * pi/2` is then an integer multiple of pi. -/
lemma sin_even_k_half_pi (k : ℤ) (hke : Even k) : Real.sin ((k : ℝ) * (Real.pi / 2)) = 0 := by
  rcases hke with ⟨j, rfl⟩
  -- k = j + j, so k * pi/2 = j * pi : an integer multiple of pi
  have hj : (↑(j + j) : ℝ) * (Real.pi / 2) = (↑j : ℝ) * Real.pi := by
    simp only [Int.cast_add]
    ring
  rw [hj]
  exact Real.sin_int_mul_pi (j : ℤ)

/-- `∫_{-pi/2}^{pi/2} cos(k t) dt = 0` for a nonzero even integer `k`. -/
lemma integral_cos_even (k : ℤ) (hk : k ≠ 0) (hke : Even k) :
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.cos (k * t) = 0 := by
  have hkR : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  calc
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.cos (k * t)
        = (1 / (k : ℝ)) * ∫ x in (k : ℝ) * (-Real.pi / 2)..(k : ℝ) * (Real.pi / 2), Real.cos x := by
      rw [intervalIntegral.integral_comp_mul_left hkR (f := fun x : ℝ => Real.cos x)]
      simp [one_div]
    _ = (1 / (k : ℝ)) *
        (Real.sin ((k : ℝ) * (Real.pi / 2)) - Real.sin ((k : ℝ) * (-Real.pi / 2))) := by
      rw [integral_cos]
    _ = (1 / (k : ℝ)) * (0 - 0) := by
      rw [sin_even_k_half_pi k hke]
      rw [show (k : ℝ) * (-Real.pi / 2) = -((k : ℝ) * (Real.pi / 2)) by ring, Real.sin_neg,
        sin_even_k_half_pi k hke]
      simp
    _ = 0 := by ring

end
end CWave
