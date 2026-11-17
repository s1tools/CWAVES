/-
Copyright (c) 2026 Antoine Grouazel. All rights reserved.
Released under the MIT license as described in the file LICENSE.
Authors: Antoine Grouazel
-/
module
public import CWave.Radial

/-!
# Executable cross-checks (`#eval` / `#guard`)

The real-valued definitions are `noncomputable`; this file mirrors the
Gegenbauer recurrence of `CWave.Definitions` on exact rationals and checks,
at compile time:

* values `C_n^(3/2)(1) = (n+1)(n+2)/2` and `C_2^(3/2)(1/2) = 3/8`,
  also used by the Python test suite;
* the weighted Gram matrix `∫_{-1}^{1} (1-x^2) C_n C_m` for `n, m ≤ 7`,
  computed exactly with rational polynomial arithmetic, equals
  `diag(2(n+1)(n+2)/(2n+3))` — the statement proved in `CWave.Radial`.

It also states the formal results it mirrors, so a change of their
statements breaks this file.
-/

set_option linter.hashCommand false

namespace CWave.Test

public section

/-- Gegenbauer `C_n^(3/2)` on `ℚ`, same recurrence as `CWave.Gegenbauer`. -/
def gegQ : ℕ → ℚ → ℚ
  | 0, _ => 1
  | 1, x => 3 * x
  | n + 2, x => ((2 * (n + 2) + 1) * x * gegQ (n + 1) x - (n + 3) * gegQ n x) / (n + 2)

/-- Polynomials as coefficient lists (constant term first). -/
abbrev Poly := List ℚ

def polyAdd : Poly → Poly → Poly
  | [], q => q
  | p, [] => p
  | a :: p, b :: q => (a + b) :: polyAdd p q

def pscale (c : ℚ) (p : Poly) : Poly := p.map (c * ·)

def pmulX (p : Poly) : Poly := 0 :: p

def pmul : Poly → Poly → Poly
  | [], _ => []
  | a :: p, q => polyAdd (pscale a q) (pmulX (pmul p q))

/-- `C_n^(3/2)` as a coefficient list, same recurrence. -/
def gegP : ℕ → Poly
  | 0 => [1]
  | 1 => [0, 3]
  | n + 2 => pscale (1 / (n + 2 : ℚ))
      (polyAdd (pscale (2 * (n + 2) + 1) (pmulX (gegP (n + 1)))) (pscale (-(n + 3 : ℚ)) (gegP n)))

/-- `∫_{-1}^{1} p`. -/
def pint (p : Poly) : ℚ :=
  (List.range p.length).zip p |>.foldl
    (fun s (k, a) => s + a * ((1 - (-1 : ℚ) ^ (k + 1)) / (k + 1))) 0

/-- `∫_{-1}^{1} (1 - x^2) C_n C_m`. -/
def gram (n m : ℕ) : ℚ := pint (pmul [1, 0, -1] (pmul (gegP n) (gegP m)))

#guard gegQ 2 (1 / 2) = 3 / 8
#guard (List.range 8).all fun n => gegQ n 1 = (n + 1) * (n + 2) / 2
-- the coefficient lists evaluate like the recurrence
#guard (List.range 8).all fun n =>
  ((gegP n).zip (List.range (gegP n).length)).foldl (fun s (a, k) => s + a * (1 / 3 : ℚ) ^ k) 0 =
    gegQ n (1 / 3)
-- weighted Gram matrix = diag(2(n+1)(n+2)/(2n+3)), n, m < 8
#guard (List.range 8).all fun n => (List.range 8).all fun m =>
  gram n m = if n = m then ((2 * (n + 1) * (n + 2) : ℕ) : ℚ) / ((2 * n + 3 : ℕ) : ℚ) else 0

#eval (List.range 6).map fun n => gram n n  -- [4/3, 12/5, 24/7, 40/9, 60/11, 84/13]

/-- The formal statements mirrored above. -/
example (n m : ℕ) :
    ∫ x in (-1 : ℝ)..1, (1 - x ^ 2) * G n x * G m x = if n = m then 1 else 0 :=
  radial_orthonormal n m

end

end CWave.Test
