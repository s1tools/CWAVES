/-
Copyright (c) 2026 Antoine Grouazel. All rights reserved.
Released under the MIT license as described in the file LICENSE.
Authors: Antoine Grouazel
-/
module
public import CWave.Definitions
public import Mathlib.Algebra.Polynomial.Derivative
public import Mathlib.Analysis.Calculus.Deriv.Polynomial
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.Topology.Algebra.Polynomial
public import Mathlib.Tactic

/-!
# CWAVE kernel orthonormality — radial part

The radial functions of the CWAVE kernels (paper B9) are
`G_n(x) sqrt(1 - x^2)` with `G n` of `CWave.Definitions`, i.e. the Gegenbauer
polynomials `C_n^(3/2)` scaled by `sqrt((n + 3/2) / ((n + 1)(n + 2)))`.
This file proves that they are orthonormal on `[-1, 1]`:

* `gegenbauer_orthogonal`: `∫ (1 - x^2) C_n C_m = 0` for `n ≠ m`;
* `gegenbauer_norm`: `∫ (1 - x^2) C_n^2 = 2 (n + 1)(n + 2) / (2 n + 3)`
  (Gradshteyn & Ryzhik 7.313 with `λ = 3/2`);
* `radial_orthonormal`: `∫ (1 - x^2) G_n G_m = δ_nm`, and
  `radial_orthonormal_sqrt` with the factor `sqrt (1 - x^2)` on each function.

The proof is algebraic. `P n : ℝ[X]` follows the three-term recurrence of
`Gegenbauer`; by a joint induction it satisfies the two derivative relations
`(1 - X^2) P'(n+1) = -(n+1) X P(n+1) + (n+3) P n` and
`(1 - X^2) P' n = (n+3) X P n - (n+1) P(n+1)`, hence the Sturm–Liouville form
`((1 - X^2)^2 P' n)' = -n(n+3) (1 - X^2) P n`. Orthogonality follows from the
fundamental theorem of calculus (the boundary term carries `(1 - X^2)^2`), and
the norms from the recurrence `(n+2) h(n+2) = (2n+5)(n+4)/(2n+7) h(n+1)`.

The change of variables from `(k_rg, k_az)` to the elliptic coordinates
(`eta^2 J = 1`) is not formalised here; it is checked numerically in the
Python test suite.
-/

namespace CWave

open Polynomial

public section

/-! ## Gegenbauer polynomials `C_n^(3/2)` as polynomials -/

/-- `C_n^(3/2)` as a real polynomial, with the recurrence of `Gegenbauer`. -/
noncomputable def P : ℕ → ℝ[X]
  | 0 => 1
  | 1 => 3 * X
  | n + 2 => C ((n + 2 : ℝ)⁻¹) * ((2 * (n : ℝ[X]) + 5) * X * P (n + 1) - ((n : ℝ[X]) + 3) * P n)

lemma P_zero : P 0 = 1 := by simp [P]

lemma P_one : P 1 = 3 * X := by simp [P]

/-- The three-term recurrence without division. -/
lemma P_rec (n : ℕ) :
    ((n : ℝ[X]) + 2) * P (n + 2) =
      (2 * (n : ℝ[X]) + 5) * X * P (n + 1) - ((n : ℝ[X]) + 3) * P n := by
  have h : ((n : ℝ[X]) + 2) = C ((n : ℝ) + 2) := by
    rw [map_add, map_natCast, map_ofNat]
  rw [P, h, ← mul_assoc, ← C_mul, mul_inv_cancel₀ (by positivity), C_1, one_mul]

lemma natCast_add_ne_zero (n : ℕ) (k : ℕ) (hk : 0 < k) : ((n : ℝ[X]) + k) ≠ 0 := by
  have h : ((n : ℝ[X]) + k) = C ((n : ℝ) + k) := by
    rw [map_add, map_natCast, map_natCast]
  rw [h, Ne, C_eq_zero]
  positivity

/-- `Gegenbauer` (a real function) is the evaluation of `P`. -/
lemma gegenbauer_eq_eval (n : ℕ) (x : ℝ) : Gegenbauer n x = (P n).eval x := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => simp [Gegenbauer, P]
    | 1 => simp [Gegenbauer, P]
    | n + 2 =>
      rw [Gegenbauer, ih n (by omega), ih (n + 1) (by omega), P]
      simp only [eval_mul, eval_C, eval_sub, eval_add, eval_natCast, eval_X, eval_ofNat]
      field_simp
      ring

/-! ## Derivative relations and Sturm–Liouville form -/

/-- Joint induction: `A n` and `B n`. -/
lemma P_deriv_rel (n : ℕ) :
    (1 - X ^ 2) * derivative (P (n + 1)) =
        -((n : ℝ[X]) + 1) * X * P (n + 1) + ((n : ℝ[X]) + 3) * P n ∧
      (1 - X ^ 2) * derivative (P n) =
        ((n : ℝ[X]) + 3) * X * P n - ((n : ℝ[X]) + 1) * P (n + 1) := by
  induction n with
  | zero =>
    rw [P_one, P_zero]
    refine ⟨?_, ?_⟩ <;>
      simp only [Nat.cast_zero, derivative_mul, derivative_X, derivative_one, derivative_ofNat] <;>
      ring
  | succ n ih =>
    obtain ⟨hA, hB⟩ := ih
    have hR := P_rec n
    have hR' := congrArg derivative hR
    simp only [derivative_mul, derivative_add, derivative_sub, derivative_natCast,
      derivative_ofNat, derivative_X] at hR'
    push_cast
    refine ⟨?_, ?_⟩
    · apply mul_left_cancel₀ (natCast_add_ne_zero n 2 (by norm_num))
      push_cast at hR' ⊢
      linear_combination (1 - X ^ 2) * hR' + (2 * (n : ℝ[X]) + 5) * X * hA
        - ((n : ℝ[X]) + 3) * hB - (-((n : ℝ[X]) + 2) * X) * hR
    · linear_combination hA + hR

/-- Sturm–Liouville form of the Gegenbauer equation with `λ = 3/2`. -/
lemma P_sturm_liouville (n : ℕ) :
    derivative ((1 - X ^ 2) ^ 2 * derivative (P n)) =
      -((n : ℝ[X]) * ((n : ℝ[X]) + 3)) * ((1 - X ^ 2) * P n) := by
  obtain ⟨hA, hB⟩ := P_deriv_rel n
  have hQ : (1 - X ^ 2) ^ 2 * derivative (P n) =
      (1 - X ^ 2) * (((n : ℝ[X]) + 3) * X * P n - ((n : ℝ[X]) + 1) * P (n + 1)) := by
    rw [← hB]; ring
  rw [hQ]
  simp only [derivative_mul, derivative_sub, derivative_add, derivative_one,
    derivative_X_pow, derivative_natCast, derivative_ofNat, derivative_X, Nat.cast_ofNat]
  push_cast
  have h2C : (C (2 : ℝ) : ℝ[X]) = 2 := map_ofNat C 2
  linear_combination ((n : ℝ[X]) + 3) * X * hB - ((n : ℝ[X]) + 1) * hA +
    (X * (n : ℝ[X]) * P (n + 1) + X * P (n + 1) - X ^ 2 * (n : ℝ[X]) * P n - 3 * X ^ 2 * P n) * h2C

/-! ## Integrals of polynomials on `[-1, 1]` -/

/-- `∫_{-1}^{1} p`. -/
noncomputable def I (p : ℝ[X]) : ℝ := ∫ x in (-1 : ℝ)..1, p.eval x

lemma I_add (p q : ℝ[X]) : I (p + q) = I p + I q := by
  simp only [I, eval_add]
  exact intervalIntegral.integral_add (p.continuous.intervalIntegrable _ _)
    (q.continuous.intervalIntegrable _ _)

lemma I_sub (p q : ℝ[X]) : I (p - q) = I p - I q := by
  simp only [I, eval_sub]
  exact intervalIntegral.integral_sub (p.continuous.intervalIntegrable _ _)
    (q.continuous.intervalIntegrable _ _)

lemma I_C_mul (c : ℝ) (p : ℝ[X]) : I (C c * p) = c * I p := by
  simp [I, eval_mul, intervalIntegral.integral_const_mul]

lemma I_natCast_mul (n : ℕ) (p : ℝ[X]) : I ((n : ℝ[X]) * p) = n * I p := by
  rw [← map_natCast C, I_C_mul]

lemma I_derivative (q : ℝ[X]) : I (derivative q) = q.eval 1 - q.eval (-1) := by
  unfold I
  rw [intervalIntegral.integral_deriv_eq_sub' (fun x => q.eval x)]
  · funext x; exact Polynomial.deriv q
  · intro x _; exact q.differentiableAt
  · exact (q.derivative.continuous).continuousOn

/-! ## Orthogonality -/

theorem P_orthogonal (n m : ℕ) (h : n ≠ m) : I ((1 - X ^ 2) * P n * P m) = 0 := by
  -- (λ_m - λ_n) w P_n P_m is the derivative of (1 - X^2)^2 (P_n' P_m - P_m' P_n)
  have hid : ((m : ℝ[X]) * ((m : ℝ[X]) + 3) - (n : ℝ[X]) * ((n : ℝ[X]) + 3)) *
      ((1 - X ^ 2) * P n * P m) =
      derivative ((1 - X ^ 2) ^ 2 *
        (derivative (P n) * P m - derivative (P m) * P n)) := by
    have hn := P_sturm_liouville n
    have hm := P_sturm_liouville m
    have e : derivative ((1 - X ^ 2) ^ 2 *
        (derivative (P n) * P m - derivative (P m) * P n)) =
        derivative ((1 - X ^ 2) ^ 2 * derivative (P n)) * P m -
          derivative ((1 - X ^ 2) ^ 2 * derivative (P m)) * P n := by
      simp only [derivative_mul, derivative_sub]; ring
    rw [e, hn, hm]; ring
  have hI := congrArg I hid
  rw [I_derivative] at hI
  simp only [eval_mul, eval_pow, eval_sub, eval_one, eval_X] at hI
  norm_num at hI
  -- the left side is (λ_m - λ_n) I(...)
  have hlam : ((m : ℝ) * (m + 3) - n * (n + 3)) ≠ 0 := by
    intro h0
    rcases Nat.lt_or_gt_of_ne h with hlt | hlt
    · have : (n : ℝ) < m := by exact_mod_cast hlt
      nlinarith
    · have : (m : ℝ) < n := by exact_mod_cast hlt
      nlinarith
  have hc : ((m : ℝ[X]) * ((m : ℝ[X]) + 3) - (n : ℝ[X]) * ((n : ℝ[X]) + 3)) =
      C ((m : ℝ) * (m + 3) - n * (n + 3)) := by
    simp only [map_sub, map_mul, map_add, map_natCast, map_ofNat]
  rw [hc, I_C_mul] at hI
  exact (mul_eq_zero.mp hI).resolve_left hlam

/-! ## Norms -/

/-- `h n = ∫ (1 - x^2) C_n^2`. -/
noncomputable def h (n : ℕ) : ℝ := I ((1 - X ^ 2) * P n * P n)

lemma I_X_pow (k : ℕ) : I (X ^ k) = (1 - (-1) ^ (k + 1)) / (k + 1) := by
  have hd : (X ^ k : ℝ[X]) = derivative (C ((k + 1 : ℝ)⁻¹) * X ^ (k + 1)) := by
    rw [derivative_C_mul_X_pow]
    push_cast
    rw [inv_mul_cancel₀ (by positivity), C_1, one_mul]
  rw [hd, I_derivative]
  simp [eval_mul]
  field_simp

lemma h_zero : h 0 = 4 / 3 := by
  have : (1 - X ^ 2) * P 0 * P 0 = X ^ 0 - X ^ 2 := by rw [P_zero]; ring
  rw [h, this, I_sub, I_X_pow, I_X_pow]; norm_num

lemma h_one : h 1 = 12 / 5 := by
  have : (1 - X ^ 2) * P 1 * P 1 = C 9 * X ^ 2 - C 9 * X ^ 4 := by
    rw [P_one]; simp only [map_ofNat]; ring
  rw [h, this, I_sub, I_C_mul, I_C_mul, I_X_pow, I_X_pow]; norm_num

/-- `∫ w X P(n+1) P(n+2) = (n+4)/(2n+7) h(n+1)`, from the recurrence at `n+1`. -/
lemma cross_term (n : ℕ) :
    (2 * (n : ℝ) + 7) * I ((1 - X ^ 2) * (X * P (n + 1) * P (n + 2))) =
      ((n : ℝ) + 4) * h (n + 1) := by
  have h3 := P_rec (n + 1)
  simp only [Nat.add_assoc, Nat.reduceAdd, Nat.cast_add, Nat.cast_one] at h3
  have E : C ((n : ℝ) + 3) * ((1 - X ^ 2) * P (n + 3) * P (n + 1)) =
      C (2 * (n : ℝ) + 7) * ((1 - X ^ 2) * (X * P (n + 1) * P (n + 2))) -
        C ((n : ℝ) + 4) * ((1 - X ^ 2) * P (n + 1) * P (n + 1)) := by
    simp only [map_add, map_mul, map_natCast, map_ofNat]
    linear_combination (1 - X ^ 2) * P (n + 1) * h3
  have e := congrArg I E
  rw [I_C_mul, I_sub, I_C_mul, I_C_mul, P_orthogonal (n + 3) (n + 1) (by omega)] at e
  rw [h]
  linarith

lemma h_rec (n : ℕ) :
    ((n : ℝ) + 2) * h (n + 2) = (2 * n + 5) * ((n + 4) / (2 * n + 7)) * h (n + 1) := by
  have hR := P_rec n
  have E : C ((n : ℝ) + 2) * ((1 - X ^ 2) * P (n + 2) * P (n + 2)) =
      C (2 * (n : ℝ) + 5) * ((1 - X ^ 2) * (X * P (n + 1) * P (n + 2))) -
        C ((n : ℝ) + 3) * ((1 - X ^ 2) * P n * P (n + 2)) := by
    simp only [map_add, map_mul, map_natCast, map_ofNat]
    linear_combination (1 - X ^ 2) * P (n + 2) * hR
  have e := congrArg I E
  rw [I_C_mul, I_sub, I_C_mul, I_C_mul, P_orthogonal n (n + 2) (by omega), mul_zero,
    sub_zero] at e
  have hc := cross_term n
  change ((n : ℝ) + 2) * I ((1 - X ^ 2) * P (n + 2) * P (n + 2)) = _
  have h7 : (2 * (n : ℝ) + 7) ≠ 0 := by positivity
  calc ((n : ℝ) + 2) * I ((1 - X ^ 2) * P (n + 2) * P (n + 2))
      = (2 * n + 5) * I ((1 - X ^ 2) * (X * P (n + 1) * P (n + 2))) := e
    _ = (2 * n + 5) * ((n + 4) / (2 * n + 7)) * h (n + 1) := by
      rw [show (2 * (n : ℝ) + 5) * ((n + 4) / (2 * n + 7)) * h (n + 1) =
          (2 * n + 5) * (((n + 4) * h (n + 1)) / (2 * n + 7)) by ring, ← hc]
      field_simp

theorem P_norm (n : ℕ) : h n = 2 * ((n : ℝ) + 1) * (n + 2) / (2 * n + 3) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => rw [h_zero]; norm_num
    | 1 => rw [h_one]; norm_num
    | n + 2 =>
      have hr := h_rec n
      rw [ih (n + 1) (by omega)] at hr
      have h2 : ((n : ℝ) + 2) ≠ 0 := by positivity
      have key : h (n + 2) = (2 * n + 5) * ((n + 4) / (2 * n + 7)) *
          (2 * ((n + 1 : ℕ) + 1) * ((n + 1 : ℕ) + 2) / (2 * (n + 1 : ℕ) + 3)) / (n + 2) := by
        rw [← hr]; field_simp
      rw [key]
      push_cast
      field_simp
      ring

/-! ## Statements on the real functions of `CWave.Definitions` -/

lemma integral_eq_I (n m : ℕ) :
    ∫ x in (-1 : ℝ)..1, (1 - x ^ 2) * Gegenbauer n x * Gegenbauer m x =
      I ((1 - X ^ 2) * P n * P m) := by
  simp [I, gegenbauer_eq_eval]

theorem gegenbauer_orthogonal (n m : ℕ) (hnm : n ≠ m) :
    ∫ x in (-1 : ℝ)..1, (1 - x ^ 2) * Gegenbauer n x * Gegenbauer m x = 0 := by
  rw [integral_eq_I]; exact P_orthogonal n m hnm

theorem gegenbauer_norm (n : ℕ) :
    ∫ x in (-1 : ℝ)..1, (1 - x ^ 2) * Gegenbauer n x * Gegenbauer n x =
      2 * ((n : ℝ) + 1) * (n + 2) / (2 * n + 3) := by
  rw [integral_eq_I]; exact P_norm n

/-- Orthonormality of the radial functions with the weight `1 - x^2`. -/
theorem radial_orthonormal (n m : ℕ) :
    ∫ x in (-1 : ℝ)..1, (1 - x ^ 2) * G n x * G m x = if n = m then 1 else 0 := by
  have e : ∀ x, (1 - x ^ 2) * G n x * G m x =
      (Real.sqrt ((n + 3 / 2) / ((n + 2) * (n + 1))) *
        Real.sqrt ((m + 3 / 2) / ((m + 2) * (m + 1)))) *
        ((1 - x ^ 2) * Gegenbauer n x * Gegenbauer m x) := by
    intro x; simp only [G]; ring
  simp_rw [e, intervalIntegral.integral_const_mul]
  split_ifs with hnm
  · subst hnm
    rw [gegenbauer_norm, Real.mul_self_sqrt (by positivity)]
    field_simp
  · rw [gegenbauer_orthogonal n m hnm, mul_zero]

/-- The same with the radial functions `G_n(x) sqrt(1 - x^2)` of paper B9. -/
theorem radial_orthonormal_sqrt (n m : ℕ) :
    ∫ x in (-1 : ℝ)..1, (G n x * Real.sqrt (1 - x ^ 2)) * (G m x * Real.sqrt (1 - x ^ 2)) =
      if n = m then 1 else 0 := by
  rw [← radial_orthonormal]
  apply intervalIntegral.integral_congr
  intro x hx
  rw [Set.uIcc_of_le (by norm_num)] at hx
  have h0 : 0 ≤ 1 - x ^ 2 := by nlinarith [hx.1, hx.2]
  simp only
  rw [show G n x * Real.sqrt (1 - x ^ 2) * (G m x * Real.sqrt (1 - x ^ 2)) =
      (Real.sqrt (1 - x ^ 2) * Real.sqrt (1 - x ^ 2)) * G n x * G m x by ring,
    Real.mul_self_sqrt h0]

end

end CWave
