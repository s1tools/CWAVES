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

/-! ## Frequencies -/

/-- The angular frequency carried by `F_m` : `0` for `m = 1`, `m` for even
    `m`, and `m-1` for odd `m`. In all cases it is an even nonnegative integer. -/
def freq (m : ℕ) : ℕ := if m = 1 then 0 else if Even m then m else m - 1

lemma freq_one : freq 1 = 0 := by simp [freq]
lemma freq_even (m : ℕ) (hm : Even m) : freq m = if m = 1 then 0 else m := by
  simp [freq, hm]
lemma freq_odd (m : ℕ) (hm : Odd m) : freq m = if m = 1 then 0 else m - 1 := by
  simp [freq, hm]

/-! ## Elementary integrals on `[-pi/2, pi/2]` -/

/-- `∫_{-pi/2}^{pi/2} 1 dt = pi`. -/
lemma integral_const_pi : ∫ t in (-Real.pi / 2)..(Real.pi / 2), 1 = Real.pi := by
  simp [integral_const]
  ring

/-- `∫_{-pi/2}^{pi/2} sin(k t) dt = 0` for a nonzero even integer `k`. -/
lemma integral_sin_even (k : ℤ) (hk : k ≠ 0) (hke : Even k) :
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.sin (k * t) = 0 := by
  have hkR : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  calc
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.sin (k * t)
        = (1 / (k : ℝ)) * (Real.cos (k * (-(Real.pi / 2))) - Real.cos (k * (Real.pi / 2))) := by
      have := integral_comp_mul_left hkR (f := fun x : ℝ => Real.sin x)
      simpa [mul_comm, mul_assoc] using this
    _ = 0 := by
      have : Real.cos (k * (-(Real.pi / 2))) = Real.cos (k * (Real.pi / 2)) := by
        ring_nf
        rw [Real.cos_neg]
        ring
      rw [this, sub_self]
      simp

/-- `∫_{-pi/2}^{pi/2} cos(k t) dt = 0` for a nonzero even integer `k`. -/
lemma integral_cos_even (k : ℤ) (hk : k ≠ 0) (hke : Even k) :
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.cos (k * t) = 0 := by
  have hkR : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  -- k * pi/2 = (k/2) * pi, an integer multiple of pi (k even)
  have hred : (k : ℝ) * (Real.pi / 2) = (k / 2 : ℝ) * Real.pi := by
    have hd : (2 : ℤ) ∣ k := by rw [even_iff_two_dvd] at hke; exact hke
    have := Int.ediv_mul_cancel hd
    ring_nf
    norm_cast at this ⊢
    simpa using this
  have hsin : Real.sin (k * (Real.pi / 2)) = 0 := by
    rw [show (k * (Real.pi / 2) : ℝ) = (k / 2 : ℝ) * Real.pi by
      calc
        k * (Real.pi / 2) = (k : ℝ) * Real.pi / 2 := by ring
        _ = (k / 2 : ℝ) * Real.pi := by
          convert hred
          ring]
    have : (k / 2 : ℝ) * Real.pi = (k / 2 : ℤ) * Real.pi := by norm_cast
    rw [this]
    simpa using Real.sin_int_mul_pi (k / 2)
  -- antiderivative of cos(k t) is sin(k t)/k, evaluated at the symmetric bounds
  calc
    ∫ t in (-Real.pi / 2)..(Real.pi / 2), Real.cos (k * t)
        = Real.sin (k * (Real.pi / 2)) / k - Real.sin (k * (-(Real.pi / 2))) / k := by
      have := integral_comp_mul_left hkR (f := fun x : ℝ => Real.cos x)
      simpa [mul_comm, mul_assoc] using this
    _ = Real.sin (k * (Real.pi / 2)) / k - (-Real.sin (k * (Real.pi / 2))) / k := by
      congr
      rw [show (k : ℝ) * (-(Real.pi / 2)) = -(k * (Real.pi / 2)) by ring, Real.sin_neg]
    _ = 2 * Real.sin (k * (Real.pi / 2)) / k := by ring
    _ = 0 := by
      rw [hsin]
      ring

end CWave
