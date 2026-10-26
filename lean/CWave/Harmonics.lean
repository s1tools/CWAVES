module
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
public import Mathlib.Tactic
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

All these frequencies are even nonnegative integers, mutually orthogonal on the
symmetric interval `[-pi/2, pi/2]`. -/

namespace CWave
open Real Set MeasureTheory
public section

/-! ## Frequencies -/

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
    simp [freq, hne]

/-! ## Elementary integrals on `[-pi/2, pi/2]` -/

lemma integral_const_pi : ∫ _ in (-Real.pi / 2)..(Real.pi / 2), 1 = Real.pi := by
  simp
  ring

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

lemma sin_even_k_half_pi (k : ℤ) (hke : Even k) : Real.sin ((k : ℝ) * (Real.pi / 2)) = 0 := by
  rcases hke with ⟨j, rfl⟩
  have hj : (↑(j + j) : ℝ) * (Real.pi / 2) = (↑j : ℝ) * Real.pi := by
    simp only [Int.cast_add]
    ring
  rw [hj]
  exact Real.sin_int_mul_pi (j : ℤ)

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

/-! ## Product-to-sum integrals (orthogonality of `F_m`) -/

/-- `∫ cos(a t) cos(b t)` over `[-pi/2, pi/2]` for even natural `a, b`. -/
lemma integral_cos_cos (a b : ℕ) (ha : Even a) (hb : Even b) :
    ∫ t in (-π / 2)..(π / 2), cos (a * t) * cos (b * t) =
      if a = b then (if a = 0 then π else π / 2) else 0 := by
  let k : ℤ := (a : ℤ) - (b : ℤ)
  let l : ℤ := (a : ℤ) + (b : ℤ)
  have hId : ∀ t, 2 * (cos (a * t) * cos (b * t)) =
      cos (((a : ℝ) - (b : ℝ)) * t) + cos (((a : ℝ) + (b : ℝ)) * t) := by
    intro t
    have hsub : (a : ℝ) * t - (b : ℝ) * t = ((a : ℝ) - (b : ℝ)) * t := by ring
    have hadd : (a : ℝ) * t + (b : ℝ) * t = ((a : ℝ) + (b : ℝ)) * t := by ring
    calc
      2 * (cos (a * t) * cos (b * t)) = 2 * cos (a * t) * cos (b * t) := by ring
      _ = cos (a * t - b * t) + cos (a * t + b * t) := by rw [two_mul_cos_mul_cos]
      _ = cos (((a : ℝ) - (b : ℝ)) * t) + cos (((a : ℝ) + (b : ℝ)) * t) := by
        rw [hsub, hadd]
  by_cases heq : a = b
  · by_cases ha0 : a = 0
    · -- a = b = 0 : integrand is 1, integral is pi, RHS = pi
      have hb0 : b = 0 := by rw [← heq]; exact ha0
      simp [heq, ha0, hb0, zero_mul, cos_zero]
      ring
    · -- a = b >= 1 : ∫ cos(a t)^2 = pi/2
      let m2 : ℤ := 2 * (a : ℤ)
      have hI : ∫ t in (-π / 2)..(π / 2), cos (a * t) * cos (b * t) = π / 2 := by
        have hEq : ∀ t, 2 * (cos (a * t) * cos (b * t)) = 1 + cos (m2 * t) := by
          intro t
          rw [hId]
          have hd : ((a : ℝ) - (b : ℝ)) = (0 : ℝ) := by rw [heq]; ring
          have hs : ((a : ℝ) + (b : ℝ)) = (m2 : ℝ) := by
            have hmb : (m2 : ℝ) = (2 * a : ℝ) := by
              have : (m2 : ℤ) = 2 * (a : ℤ) := by simp [m2]
              rw [this]
              norm_cast
            rw [hmb, heq]
            ring
          rw [show cos (((a : ℝ) - (b : ℝ)) * t) = cos (0 * t) by rw [hd],
            show cos (((a : ℝ) + (b : ℝ)) * t) = cos (m2 * t) by rw [hs]]
          simp [zero_mul, cos_zero]
        have h2 : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) =
            ∫ t in (-π / 2)..(π / 2), 1 + cos (m2 * t) := by
          rw [intervalIntegral.integral_congr (fun t _ => hEq t)]
        have h3 : ∫ t in (-π / 2)..(π / 2), 1 + cos (m2 * t) =
            (∫ t in (-π / 2)..(π / 2), 1) + (∫ t in (-π / 2)..(π / 2), cos (m2 * t)) := by
          rw [intervalIntegral.integral_add] <;> (apply Continuous.intervalIntegrable; fun_prop)
        have hEven : Even m2 := by
          use (a : ℤ)
          ring
        have hNonz : m2 ≠ 0 := by
          by_contra h
          have : (a : ℤ) = 0 := by simpa [m2] using h
          have : a = 0 := by exact_mod_cast ‹(a : ℤ) = 0›
          exact ha0 ‹a = 0›
        have h4 : ∫ t in (-π / 2)..(π / 2), cos (m2 * t) = 0 :=
          integral_cos_even m2 hNonz hEven
        have h5 : ∫ t in (-π / 2)..(π / 2), 1 = π := by
          simp
          ring
        have h2s : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) = π := by
          rw [h2, h3, h5, h4]
          ring
        have hmul : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) =
            2 * (∫ t in (-π / 2)..(π / 2), cos (a * t) * cos (b * t)) := by
          rw [intervalIntegral.integral_const_mul]
        nlinarith [hmul, h2s]
      have hbne : b ≠ 0 := by simpa [heq] using ha0
      simpa [heq, hbne] using hI
  · -- a ≠ b : both frequencies (a-b), (a+b) are nonzero even -> both integrals 0
    have hsubn : k ≠ 0 := by
      intro h
      have hkn : (a : ℤ) - (b : ℤ) = 0 := by simpa [k] using h
      have : (a : ℤ) = (b : ℤ) := by linarith
      exact heq (by simpa [Int.natCast_inj] using this)
    have hsumn : l ≠ 0 := by
      intro h
      have hln : (a : ℤ) + (b : ℤ) = 0 := by simpa [l] using h
      have hae : (a : ℤ) ≥ 0 := by positivity
      have hbe : (b : ℤ) ≥ 0 := by positivity
      have ha0z : (a : ℤ) = 0 := by linarith
      have hb0z : (b : ℤ) = 0 := by linarith
      have : (a : ℤ) = (b : ℤ) := by simpa [ha0z, hb0z]
      exact heq (by simpa [Int.natCast_inj] using this)
    have hsubE : Even k := by
      rcases ha with ⟨x, rfl⟩
      rcases hb with ⟨y, rfl⟩
      dsimp only [k]
      rw [show (↑(x + x) : ℤ) = 2 * (x : ℤ) by norm_cast; ring,
          show (↑(y + y) : ℤ) = 2 * (y : ℤ) by norm_cast; ring]
      use (x - y : ℤ)
      ring
    have hsumE : Even l := by
      rcases ha with ⟨x, rfl⟩
      rcases hb with ⟨y, rfl⟩
      dsimp only [l]
      rw [show (↑(x + x) : ℤ) = 2 * (x : ℤ) by norm_cast; ring,
          show (↑(y + y) : ℤ) = 2 * (y : ℤ) by norm_cast; ring]
      use (x + y : ℤ)
      ring
    have h0 : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t) = 0 := by
      have hd : ((a : ℝ) - (b : ℝ)) = (k : ℝ) := by dsimp only [k]; simp [Int.cast_sub]
      have hcong : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t) =
          ∫ t in (-π / 2)..(π / 2), cos (k * t) := by
        apply intervalIntegral.integral_congr
        intro t _
        rw [show ((a : ℝ) - (b : ℝ)) * t = (k : ℝ) * t by rw [hd]]
      rw [hcong, integral_cos_even k hsubn hsubE]
    have h1 : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t) = 0 := by
      have hl : ((a : ℝ) + (b : ℝ)) = (l : ℝ) := by dsimp only [l]; simp [Int.cast_add]
      have hcong : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t) =
          ∫ t in (-π / 2)..(π / 2), cos (l * t) := by
        apply intervalIntegral.integral_congr
        intro t _
        rw [show ((a : ℝ) + (b : ℝ)) * t = (l : ℝ) * t by rw [hl]]
      rw [hcong, integral_cos_even l hsumn hsumE]
    have h2 : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) =
        (∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t)) +
        (∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t)) := by
      rw [intervalIntegral.integral_congr (fun t _ => hId t)]
      rw [intervalIntegral.integral_add] <;> (apply Continuous.intervalIntegrable; fun_prop)
    have h2s : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) = 0 := by
      rw [h2, h0, h1]
      ring
    have hmul : ∫ t in (-π / 2)..(π / 2), 2 * (cos (a * t) * cos (b * t)) =
        2 * (∫ t in (-π / 2)..(π / 2), cos (a * t) * cos (b * t)) := by
      rw [intervalIntegral.integral_const_mul]
    have hI : ∫ t in (-π / 2)..(π / 2), cos (a * t) * cos (b * t) = 0 := by
      rw [← hmul] at h2s
      nlinarith
    simp [hI, heq]

/-- `∫ sin(a t) sin(b t)` over `[-pi/2, pi/2]` for even natural `a, b`. -/
lemma integral_sin_sin (a b : ℕ) (ha : Even a) (hb : Even b) :
    ∫ t in (-π / 2)..(π / 2), sin (a * t) * sin (b * t) =
      if a = b then (if a = 0 then 0 else π / 2) else 0 := by
  let k : ℤ := (a : ℤ) - (b : ℤ)
  let l : ℤ := (a : ℤ) + (b : ℤ)
  have hId : ∀ t, 2 * (sin (a * t) * sin (b * t)) =
      cos (((a : ℝ) - (b : ℝ)) * t) - cos (((a : ℝ) + (b : ℝ)) * t) := by
    intro t
    have hsub : (a : ℝ) * t - (b : ℝ) * t = ((a : ℝ) - (b : ℝ)) * t := by ring
    have hadd : (a : ℝ) * t + (b : ℝ) * t = ((a : ℝ) + (b : ℝ)) * t := by ring
    calc
      2 * (sin (a * t) * sin (b * t)) = 2 * sin (a * t) * sin (b * t) := by ring
      _ = cos (a * t - b * t) - cos (a * t + b * t) := by rw [two_mul_sin_mul_sin]
      _ = cos (((a : ℝ) - (b : ℝ)) * t) - cos (((a : ℝ) + (b : ℝ)) * t) := by
        rw [hsub, hadd]
  by_cases heq : a = b
  · by_cases ha0 : a = 0
    · -- a = b = 0 : integrand is 0, integral is 0, RHS = 0
      have hb0 : b = 0 := by rw [← heq]; exact ha0
      simp [heq, ha0, hb0, zero_mul, sin_zero]
    · -- a = b >= 1 : ∫ sin(a t)^2 = pi/2
      let m2 : ℤ := 2 * (a : ℤ)
      have hI : ∫ t in (-π / 2)..(π / 2), sin (a * t) * sin (b * t) = π / 2 := by
        have hEq : ∀ t, 2 * (sin (a * t) * sin (b * t)) = 1 - cos (m2 * t) := by
          intro t
          rw [hId]
          have hd : ((a : ℝ) - (b : ℝ)) = (0 : ℝ) := by rw [heq]; ring
          have hs : ((a : ℝ) + (b : ℝ)) = (m2 : ℝ) := by
            have hmb : (m2 : ℝ) = (2 * a : ℝ) := by
              have : (m2 : ℤ) = 2 * (a : ℤ) := by simp [m2]
              rw [this]
              norm_cast
            rw [hmb, heq]
            ring
          rw [show cos (((a : ℝ) - (b : ℝ)) * t) = cos (0 * t) by rw [hd],
            show cos (((a : ℝ) + (b : ℝ)) * t) = cos (m2 * t) by rw [hs]]
          simp [zero_mul, cos_zero]
        have h2 : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) =
            ∫ t in (-π / 2)..(π / 2), 1 - cos (m2 * t) := by
          rw [intervalIntegral.integral_congr (fun t _ => hEq t)]
        have h3 : ∫ t in (-π / 2)..(π / 2), 1 - cos (m2 * t) =
            (∫ t in (-π / 2)..(π / 2), 1) - (∫ t in (-π / 2)..(π / 2), cos (m2 * t)) := by
          rw [intervalIntegral.integral_sub] <;> (apply Continuous.intervalIntegrable; fun_prop)
        have hEven : Even m2 := by
          use (a : ℤ)
          ring
        have hNonz : m2 ≠ 0 := by
          by_contra h
          have : (a : ℤ) = 0 := by simpa [m2] using h
          have : a = 0 := by exact_mod_cast ‹(a : ℤ) = 0›
          exact ha0 ‹a = 0›
        have h4 : ∫ t in (-π / 2)..(π / 2), cos (m2 * t) = 0 :=
          integral_cos_even m2 hNonz hEven
        have h5 : ∫ t in (-π / 2)..(π / 2), 1 = π := by
          simp
          ring
        have h2s : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) = π := by
          rw [h2, h3, h5, h4]
          ring
        have hmul : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) =
            2 * (∫ t in (-π / 2)..(π / 2), sin (a * t) * sin (b * t)) := by
          rw [intervalIntegral.integral_const_mul]
        nlinarith [hmul, h2s]
      have hbne : b ≠ 0 := by simpa [heq] using ha0
      simpa [heq, hbne] using hI
  · -- a ≠ b : both frequencies (a-b), (a+b) are nonzero even -> both integrals 0
    have hsubn : k ≠ 0 := by
      intro h
      have hkn : (a : ℤ) - (b : ℤ) = 0 := by simpa [k] using h
      have : (a : ℤ) = (b : ℤ) := by linarith
      exact heq (by simpa [Int.natCast_inj] using this)
    have hsumn : l ≠ 0 := by
      intro h
      have hln : (a : ℤ) + (b : ℤ) = 0 := by simpa [l] using h
      have hae : (a : ℤ) ≥ 0 := by positivity
      have hbe : (b : ℤ) ≥ 0 := by positivity
      have ha0z : (a : ℤ) = 0 := by linarith
      have hb0z : (b : ℤ) = 0 := by linarith
      have : (a : ℤ) = (b : ℤ) := by simpa [ha0z, hb0z]
      exact heq (by simpa [Int.natCast_inj] using this)
    have hsubE : Even k := by
      rcases ha with ⟨x, rfl⟩
      rcases hb with ⟨y, rfl⟩
      dsimp only [k]
      rw [show (↑(x + x) : ℤ) = 2 * (x : ℤ) by norm_cast; ring,
          show (↑(y + y) : ℤ) = 2 * (y : ℤ) by norm_cast; ring]
      use (x - y : ℤ)
      ring
    have hsumE : Even l := by
      rcases ha with ⟨x, rfl⟩
      rcases hb with ⟨y, rfl⟩
      dsimp only [l]
      rw [show (↑(x + x) : ℤ) = 2 * (x : ℤ) by norm_cast; ring,
          show (↑(y + y) : ℤ) = 2 * (y : ℤ) by norm_cast; ring]
      use (x + y : ℤ)
      ring
    have h0 : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t) = 0 := by
      have hd : ((a : ℝ) - (b : ℝ)) = (k : ℝ) := by dsimp only [k]; simp [Int.cast_sub]
      have hcong : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t) =
          ∫ t in (-π / 2)..(π / 2), cos (k * t) := by
        apply intervalIntegral.integral_congr
        intro t _
        rw [show ((a : ℝ) - (b : ℝ)) * t = (k : ℝ) * t by rw [hd]]
      rw [hcong, integral_cos_even k hsubn hsubE]
    have h1 : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t) = 0 := by
      have hl : ((a : ℝ) + (b : ℝ)) = (l : ℝ) := by dsimp only [l]; simp [Int.cast_add]
      have hcong : ∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t) =
          ∫ t in (-π / 2)..(π / 2), cos (l * t) := by
        apply intervalIntegral.integral_congr
        intro t _
        rw [show ((a : ℝ) + (b : ℝ)) * t = (l : ℝ) * t by rw [hl]]
      rw [hcong, integral_cos_even l hsumn hsumE]
    have h2 : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) =
        (∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) - (b : ℝ)) * t)) -
        (∫ t in (-π / 2)..(π / 2), cos (((a : ℝ) + (b : ℝ)) * t)) := by
      rw [intervalIntegral.integral_congr (fun t _ => hId t)]
      rw [intervalIntegral.integral_sub] <;> (apply Continuous.intervalIntegrable; fun_prop)
    have h2s : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) = 0 := by
      rw [h2, h0, h1]
      ring
    have hmul : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * sin (b * t)) =
        2 * (∫ t in (-π / 2)..(π / 2), sin (a * t) * sin (b * t)) := by
      rw [intervalIntegral.integral_const_mul]
    have hI : ∫ t in (-π / 2)..(π / 2), sin (a * t) * sin (b * t) = 0 := by
      rw [← hmul] at h2s
      nlinarith
    simp [hI, heq]

/-- `∫ sin(a t) cos(b t) = 0` over `[-pi/2, pi/2]` for even natural `a, b`. -/
lemma integral_sin_cos (a b : ℕ) (ha : Even a) (hb : Even b) :
    ∫ t in (-π / 2)..(π / 2), sin (a * t) * cos (b * t) = 0 := by
  let k : ℤ := (a : ℤ) - (b : ℤ)
  let l : ℤ := (a : ℤ) + (b : ℤ)
  have hId : ∀ t, 2 * (sin (a * t) * cos (b * t)) =
      sin (((a : ℝ) - (b : ℝ)) * t) + sin (((a : ℝ) + (b : ℝ)) * t) := by
    intro t
    have hsub : (a : ℝ) * t - (b : ℝ) * t = ((a : ℝ) - (b : ℝ)) * t := by ring
    have hadd : (a : ℝ) * t + (b : ℝ) * t = ((a : ℝ) + (b : ℝ)) * t := by ring
    calc
      2 * (sin (a * t) * cos (b * t)) = 2 * sin (a * t) * cos (b * t) := by ring
      _ = sin (a * t - b * t) + sin (a * t + b * t) := by rw [two_mul_sin_mul_cos]
      _ = sin (((a : ℝ) - (b : ℝ)) * t) + sin (((a : ℝ) + (b : ℝ)) * t) := by
        rw [hsub, hadd]
  have hk0 : ∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) - (b : ℝ)) * t) = 0 := by
    have hd : ((a : ℝ) - (b : ℝ)) = (k : ℝ) := by simp [Int.cast_sub]
    have hcong : ∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) - (b : ℝ)) * t) =
        ∫ t in (-π / 2)..(π / 2), sin (k * t) := by
      apply intervalIntegral.integral_congr
      intro t _
      rw [show ((a : ℝ) - (b : ℝ)) * t = (k : ℝ) * t by rw [hd]]
    rw [hcong]
    by_cases hkn : k ≠ 0
    · exact integral_sin_even k hkn
    · simp [hkn]
  have hl0 : ∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) + (b : ℝ)) * t) = 0 := by
    have hl : ((a : ℝ) + (b : ℝ)) = (l : ℝ) := by simp [Int.cast_add]
    have hcong : ∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) + (b : ℝ)) * t) =
        ∫ t in (-π / 2)..(π / 2), sin (l * t) := by
      apply intervalIntegral.integral_congr
      intro t _
      rw [show ((a : ℝ) + (b : ℝ)) * t = (l : ℝ) * t by rw [hl]]
    rw [hcong]
    by_cases hln : l ≠ 0
    · exact integral_sin_even l hln
    · simp [hln]
  have h2 : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * cos (b * t)) =
      (∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) - (b : ℝ)) * t)) +
      (∫ t in (-π / 2)..(π / 2), sin (((a : ℝ) + (b : ℝ)) * t)) := by
    rw [intervalIntegral.integral_congr (fun t _ => hId t)]
    rw [intervalIntegral.integral_add] <;> (apply Continuous.intervalIntegrable; fun_prop)
  have h2s : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * cos (b * t)) = 0 := by
    rw [h2, hk0, hl0]
    ring
  have hmul : ∫ t in (-π / 2)..(π / 2), 2 * (sin (a * t) * cos (b * t)) =
      2 * (∫ t in (-π / 2)..(π / 2), sin (a * t) * cos (b * t)) := by
    rw [intervalIntegral.integral_const_mul]
  rw [← hmul] at h2s
  nlinarith

/-! ## The angular harmonics `F_m` -/

lemma F_one : F 1 = fun _ => 1 / Real.sqrt Real.pi := by
  ext
  simp [F]

lemma F_even (m : ℕ) (hm : Even m) : F m = fun t => Real.sqrt (2 / Real.pi) * Real.sin (m * t) := by
  ext
  simp [F, hm]
  split_ifs <;> omega

lemma F_odd (m : ℕ) (hm : Odd m) (hm1 : m ≠ 1) :
    F m = fun t => Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t) := by
  ext
  simp [F, hm1]
  split_ifs <;>
  (rcases hm with ⟨k, rfl⟩; omega)

/-- `∫ F_n = 0` for `n >= 2` : the single-frequency integral vanishes. -/
lemma integral_F_zero (n : ℕ) (hn : 1 < n) :
    ∫ t in (-π / 2)..(π / 2), F n t = 0 := by
  by_cases hne : Even n
  · have hEven : Even ((n : ℤ)) := by
      rcases hne with ⟨x, rfl⟩
      use (x : ℤ)
      norm_cast
      ring
    have hNonz : (n : ℤ) ≠ 0 := by
      intro h
      have : (n : ℕ) = 0 := by exact_mod_cast h
      omega
    rw [F_even n hne]
    rw [intervalIntegral.integral_const_mul]
    exact (integral_sin_even n hNonz).symm
  · have hodd : Odd n := by
      exact Nat.not_even_iff_odd.mpr hne
    have hnm1 : (n - 1 : ℕ) ≠ 0 := by
      intro h
      omega
    have hEven : Even ((n - 1 : ℕ)) := by
      rcases hodd with ⟨k, rfl⟩
      use k
      omega
    rw [F_odd n hodd (by omega)]
    rw [intervalIntegral.integral_const_mul]
    exact (integral_cos_even (n - 1) (by simpa using hnm1) hEven).symm

/-! ## Orthonormality of the angular basis -/

theorem angular_orthonormal (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∫ t in (-π / 2)..(π / 2), F m t * F n t = if m = n then 1 else 0 := by
  by_cases hm1 : m = 1
  · by_cases hn1 : n = 1
    · -- m = n = 1
      have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t = 1 := by
        have hInt : ∀ t, F m t * F n t = 1 / Real.pi := by
          intro t
          simp [hm1, hn1, F_one]
          have hsq : (Real.sqrt Real.pi) ^ 2 = Real.pi := by
            rw [Real.sq_sqrt (by positivity : 0 ≤ Real.pi)]
          field_simp [hsq]
          nlinarith [hsq]
        rw [intervalIntegral.integral_congr (fun t _ => hInt t)]
        have hConst : (∫ t in (-π / 2)..(π / 2), (1 / Real.pi)) =
            (1 / Real.pi) * (∫ t in (-π / 2)..(π / 2), 1) := by
          rw [show (∫ t in (-π / 2)..(π / 2), (1 / Real.pi)) =
              (∫ t in (-π / 2)..(π / 2), (1 / Real.pi) * 1) by
                apply intervalIntegral.integral_congr
                intro t _
                ring,
            intervalIntegral.integral_const_mul]
        rw [hConst, integral_const_pi]
        field_simp
      simpa [hm1, hn1] using h
    · -- m = 1, n ≠ 1
      have hComm : (∫ t in (-π / 2)..(π / 2), F m t * F n t) =
          (∫ t in (-π / 2)..(π / 2), F n t * (1 / Real.sqrt Real.pi)) := by
        simp [hm1, F_one]
        apply intervalIntegral.integral_congr
        intro t _
        ring
      have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t = 0 := by
        rw [hComm, intervalIntegral.integral_mul_const,
          integral_F_zero n (Nat.lt_of_le_of_ne hn hn1)]
        ring
      have hne : m ≠ n := by
        intro h
        rw [hm1] at h
        exact hn1 (by simpa using h)
      simp [h, hne]
  · by_cases hmE : Even m
    · by_cases hnE : Even n
      · -- m, n even (>= 2)
        have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t =
            (2 / Real.pi) * (∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * Real.sin (n * t)) := by
          simp [hm1, hnE, F_even]
          have hRearr : (∫ t in (-π / 2)..(π / 2),
              (Real.sqrt (2 / Real.pi) * Real.sin (m * t)) *
              (Real.sqrt (2 / Real.pi) * Real.sin (n * t))) =
              (∫ t in (-π / 2)..(π / 2),
                (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
                (Real.sin (m * t) * Real.sin (n * t))) := by
            apply intervalIntegral.integral_congr
            intro t _
            ring_nf
          rw [hRearr, intervalIntegral.integral_const_mul]
          have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
            nlinarith [show (Real.sqrt (2 / Real.pi)) ^ 2 = 2 / Real.pi by
              field_simp [Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]]
          rw [hsc2]
        by_cases heq : m = n
        · have hI : ∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * Real.sin (n * t) = Real.pi / 2 := by
            simpa [heq] using integral_sin_sin m n hmE hnE
          calc
            _ = (2 / Real.pi) * (Real.pi / 2) := by rw [h, hI]
            _ = 1 := by field_simp
            _ = (if m = n then 1 else 0 : ℝ) := by simp [heq]
        · have hI : ∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * Real.sin (n * t) = 0 := by
            simpa [heq] using integral_sin_sin m n hmE hnE
          calc
            _ = (2 / Real.pi) * 0 := by rw [h, hI]
            _ = 0 := by ring
            _ = (if m = n then 1 else 0 : ℝ) := by simp [heq]
      · -- m even (>=2), n odd (>=3) or n = 1
        by_cases hn1 : n = 1
        · -- n = 1
          have hComm : (∫ t in (-π / 2)..(π / 2), F m t * F n t) =
              (∫ t in (-π / 2)..(π / 2), Real.sin (m * t) * (1 / Real.sqrt Real.pi)) := by
            simp [hm1, hn1, hmE, F_even, F_one]
            apply intervalIntegral.integral_congr
            intro t _
            ring
          have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t = 0 := by
            rw [hComm, intervalIntegral.integral_mul_const]
            have hNonz : (m : ℤ) ≠ 0 := by norm_cast; omega
            rw [integral_sin_even m hNonz]
            ring
          have hne : m ≠ n := by
            intro h
            rw [hn1] at h
            exact hm1 (by simpa [hmE] using h)
          simp [h, hne]
        · -- n odd (>=3)
          have hodd : Odd n := by
            exact Nat.not_even_iff_odd.mpr hnE
          have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t =
              (2 / Real.pi) * (∫ t in (-π / 2)..(π / 2),
                Real.sin (m * t) * Real.cos ((n - 1) * t)) := by
            simp [hm1, hn1, hmE, F_even]
            simp [hodd, hn1, F_odd]
            have hRearr : (∫ t in (-π / 2)..(π / 2),
                (Real.sqrt (2 / Real.pi) * Real.sin (m * t)) *
                (Real.sqrt (2 / Real.pi) * Real.cos ((n - 1) * t))) =
                (∫ t in (-π / 2)..(π / 2),
                  (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
                  (Real.sin (m * t) * Real.cos ((n - 1) * t))) := by
              apply intervalIntegral.integral_congr
              intro t _
              ring_nf
            rw [hRearr, intervalIntegral.integral_const_mul]
            have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
              nlinarith [show (Real.sqrt (2 / Real.pi)) ^ 2 = 2 / Real.pi by
                field_simp [Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]]
            rw [hsc2]
          have hnm1E : Even ((n - 1 : ℕ)) := by
            rcases hodd with ⟨k, rfl⟩
            use k
            omega
          have hI : ∫ t in (-π / 2)..(π / 2),
              Real.sin (m * t) * Real.cos ((n - 1) * t) = 0 := by
            simpa using integral_sin_cos m (n - 1) hmE hnm1E
          have hne : m ≠ n := by
            intro h
            have := hmE
            have := h
            omega
          calc
            _ = (2 / Real.pi) * 0 := by rw [h, hI]
            _ = 0 := by ring
            _ = (if m = n then 1 else 0 : ℝ) := by simp [hne]
    · -- m odd (>= 3)
      by_cases hnE : Even n
      · -- m odd (>=3), n even (>=2)
        have hodd : Odd m := by
          exact Nat.not_even_iff_odd.mpr hmE
        have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t =
            (2 / Real.pi) * (∫ t in (-π / 2)..(π / 2),
              Real.cos ((m - 1) * t) * Real.sin (n * t)) := by
          simp [hm1, hnE, F_even]
          simp [hodd, hm1, F_odd]
          have hRearr : (∫ t in (-π / 2)..(π / 2),
              (Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t)) *
              (Real.sqrt (2 / Real.pi) * Real.sin (n * t))) =
              (∫ t in (-π / 2)..(π / 2),
                (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
                (Real.cos ((m - 1) * t) * Real.sin (n * t))) := by
            apply intervalIntegral.integral_congr
            intro t _
            ring_nf
          rw [hRearr, intervalIntegral.integral_const_mul]
          have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
            nlinarith [show (Real.sqrt (2 / Real.pi)) ^ 2 = 2 / Real.pi by
              field_simp [Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]]
          rw [hsc2]
        have hnm1E : Even ((m - 1 : ℕ)) := by
          rcases hodd with ⟨k, rfl⟩
          use k
          omega
        have hI : ∫ t in (-π / 2)..(π / 2),
            Real.cos ((m - 1) * t) * Real.sin (n * t) = 0 := by
          have h : ∫ t in (-π / 2)..(π / 2),
              Real.sin (n * t) * Real.cos ((m - 1) * t) = 0 := by
            simpa using integral_sin_cos n (m - 1) hnE hnm1E
          simpa [mul_comm] using h
        have hne : m ≠ n := by
          intro h
          have := hmE
          have := h
          omega
        calc
          _ = (2 / Real.pi) * 0 := by rw [h, hI]
          _ = 0 := by ring
          _ = (if m = n then 1 else 0 : ℝ) := by simp [hne]
      · by_cases hn1 : n = 1
        · -- n = 1, m odd >= 3
          have hodd : Odd m := by
            exact Nat.not_even_iff_odd.mpr hmE
          have hComm : (∫ t in (-π / 2)..(π / 2), F m t * F n t) =
              (∫ t in (-π / 2)..(π / 2), Real.cos ((m - 1) * t) * (1 / Real.sqrt Real.pi)) := by
            simp [hm1, hn1]
            simp [hodd, hm1, F_odd]
            apply intervalIntegral.integral_congr
            intro t _
            ring
          have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t = 0 := by
            rw [hComm, intervalIntegral.integral_mul_const]
            have hnm1 : (m - 1 : ℕ) ≠ 0 := by
              intro h
              omega
            have hnm1E : Even ((m - 1 : ℕ)) := by
              rcases hodd with ⟨k, rfl⟩
              use k
              omega
            rw [integral_cos_even (m - 1) (by simpa using hnm1) hnm1E]
            ring
          have hne : m ≠ n := by
            intro h
            rw [hn1] at h
            exact hm1 (by simpa using h)
          simp [h, hne]
        · -- m, n odd (>= 3)
          have hoddm : Odd m := by
            exact Nat.not_even_iff_odd.mpr hmE
          have hoddn : Odd n := by
            exact Nat.not_even_iff_odd.mpr hnE
          have h : ∫ t in (-π / 2)..(π / 2), F m t * F n t =
              (2 / Real.pi) * (∫ t in (-π / 2)..(π / 2),
                Real.cos ((m - 1) * t) * Real.cos ((n - 1) * t)) := by
            simp [hm1, hn1]
            simp [hoddm, hm1, F_odd m hoddm hm1, hoddn, hn1, F_odd n hoddn hn1]
            have hRearr : (∫ t in (-π / 2)..(π / 2),
                (Real.sqrt (2 / Real.pi) * Real.cos ((m - 1) * t)) *
                (Real.sqrt (2 / Real.pi) * Real.cos ((n - 1) * t))) =
                (∫ t in (-π / 2)..(π / 2),
                  (Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi)) *
                  (Real.cos ((m - 1) * t) * Real.cos ((n - 1) * t))) := by
              apply intervalIntegral.integral_congr
              intro t _
              ring_nf
            rw [hRearr, intervalIntegral.integral_const_mul]
            have hsc2 : Real.sqrt (2 / Real.pi) * Real.sqrt (2 / Real.pi) = 2 / Real.pi := by
              nlinarith [show (Real.sqrt (2 / Real.pi)) ^ 2 = 2 / Real.pi by
                field_simp [Real.sq_sqrt (by positivity : 0 ≤ 2 / Real.pi)]]
            rw [hsc2]
          have hnm1E : Even ((m - 1 : ℕ)) := by
            rcases hoddm with ⟨k, rfl⟩
            use k
            omega
          have hnn1E : Even ((n - 1 : ℕ)) := by
            rcases hoddn with ⟨k, rfl⟩
            use k
            omega
          by_cases heq : m = n
          · have hI : ∫ t in (-π / 2)..(π / 2),
              Real.cos ((m - 1) * t) * Real.cos ((n - 1) * t) = Real.pi / 2 := by
              have hm1p : (m - 1 : ℕ) ≠ 0 := by
                intro h
                omega
              have : m - 1 = n - 1 := by rw [heq]
              simpa [this, hm1p] using integral_cos_cos (m - 1) (n - 1) hnm1E hnn1E
            calc
              _ = (2 / Real.pi) * (Real.pi / 2) := by rw [h, hI]
              _ = 1 := by field_simp
              _ = (if m = n then 1 else 0 : ℝ) := by simp [heq]
          · have hI : ∫ t in (-π / 2)..(π / 2),
              Real.cos ((m - 1) * t) * Real.cos ((n - 1) * t) = 0 := by
              have hne' : m - 1 ≠ n - 1 := by
                intro h
                have : m = n := by
                  rw [Nat.sub_add_cancel hm, Nat.sub_add_cancel hn, h]
                contradiction
              simpa [hne'] using integral_cos_cos (m - 1) (n - 1) hnm1E hnn1E
            have hne : m ≠ n := by
              intro h
              have hne' : m - 1 ≠ n - 1 := by
                intro h
                have : m = n := by
                  rw [Nat.sub_add_cancel hm, Nat.sub_add_cancel hn, h]
                contradiction
              contradiction
            calc
              _ = (2 / Real.pi) * 0 := by rw [h, hI]
              _ = 0 := by ring
              _ = (if m = n then 1 else 0 : ℝ) := by simp [hne]

end
end CWave
