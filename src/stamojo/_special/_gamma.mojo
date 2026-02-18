# ===----------------------------------------------------------------------=== #
# StaMojo - Regularized incomplete gamma functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Regularized incomplete gamma functions.

Provides:
- `gammainc(a, x)`: Regularized lower incomplete gamma function P(a, x).
- `gammaincc(a, x)`: Regularized upper incomplete gamma function Q(a, x).

These are defined as:
    P(a, x) = γ(a, x) / Γ(a)
    Q(a, x) = 1 - P(a, x) = Γ(a, x) / Γ(a)

where γ(a, x) is the lower incomplete gamma function and Γ(a, x) is the
upper incomplete gamma function.

Implementation strategy:
- **Series expansion** is used to compute P(a, x) for *all* x.  The series
  converges for every x; only the number of terms required increases
  with x.  With `_MAX_ITER = 1000` the series handles x up to ~1000,
  which covers all practical statistical use-cases.
- **Continued-fraction (Lentz)** expansion is used to compute Q(a, x)
  directly when x ≥ a + 1 *and* a is far from an integer (to avoid the
  known stalling of Lentz's method when a CF coefficient is zero).
  When the CF is not applicable, Q is obtained as 1 − P.

Reference:
    Press et al., Numerical Recipes, 3rd ed., Section 6.2.
"""

from math import lgamma, exp, log, nan, inf


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 3.0e-12  # Desired relative accuracy.
comptime _MAX_ITER = 1000  # Maximum number of iterations for series.
comptime _CF_MAX_ITER = 200  # Maximum number for continued-fraction.
comptime _FPMIN = 1.0e-30  # Near smallest representable floating-point number.


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _is_near_integer(a: Float64) -> Bool:
    """Return True if `a` is within 1e-12 of a positive integer.

    When a is (close to) an integer, the continued-fraction expansion
    has a zero numerator coefficient at step i == a, which causes
    Lentz's method to stall permanently.  Detecting this lets us fall
    back to the series expansion instead.
    """
    if a <= 0.0 or a > 1000.0:
        return False
    var rounded = Float64(Int(a + 0.5))
    return abs(a - rounded) < 1.0e-12


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn gammainc(a: Float64, x: Float64) -> Float64:
    """Regularized lower incomplete gamma function P(a, x).

    Computes P(a, x) = γ(a, x) / Γ(a), where γ(a, x) is the lower
    incomplete gamma integral from 0 to x.

    Args:
        a: Shape parameter. Must be positive.
        x: Integration upper limit. Must be non-negative.

    Returns:
        The value of the regularized lower incomplete gamma function,
        in the range [0, 1].
    """
    if x < 0.0 or a <= 0.0:
        return nan[DType.float64]()
    if x == 0.0:
        return 0.0

    # For the lower function, the series always gives P directly.
    # Use CF only when it is both efficient and safe.
    if x < a + 1.0 or _is_near_integer(a):
        return _gamma_series(a, x)
    else:
        return 1.0 - _gamma_cf(a, x)


fn gammaincc(a: Float64, x: Float64) -> Float64:
    """Regularized upper incomplete gamma function Q(a, x).

    Computes Q(a, x) = 1 - P(a, x) = Γ(a, x) / Γ(a), where Γ(a, x)
    is the upper incomplete gamma integral from x to infinity.

    Args:
        a: Shape parameter. Must be positive.
        x: Integration lower limit. Must be non-negative.

    Returns:
        The value of the regularized upper incomplete gamma function,
        in the range [0, 1].
    """
    if x < 0.0 or a <= 0.0:
        return nan[DType.float64]()
    if x == 0.0:
        return 1.0

    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    elif _is_near_integer(a):
        # CF stalls for integer a; use series and subtract.
        return 1.0 - _gamma_series(a, x)
    else:
        return _gamma_cf(a, x)


# ===----------------------------------------------------------------------=== #
# Internal implementations
# ===----------------------------------------------------------------------=== #


fn _gamma_series(a: Float64, x: Float64) -> Float64:
    """Evaluate the regularized lower incomplete gamma function P(a, x)
    by its series representation.

    P(a, x) = e^{-x} x^a / Γ(a) * Σ_{n=0}^{∞} x^n / (a(a+1)...(a+n))

    This converges quickly for x < a + 1.
    """
    var gln = lgamma(a)

    var ap = a
    var del_val = 1.0 / a
    var sum_val = del_val

    for _ in range(_MAX_ITER):
        ap += 1.0
        del_val *= x / ap
        sum_val += del_val
        if abs(del_val) < abs(sum_val) * _EPS:
            return sum_val * exp(-x + a * log(x) - gln)

    # Series did not converge; return best estimate.
    return sum_val * exp(-x + a * log(x) - gln)


fn _gamma_cf(a: Float64, x: Float64) -> Float64:
    """Evaluate the regularized upper incomplete gamma function Q(a, x)
    by its continued fraction representation (modified Lentz's method).

    This converges quickly for x >= a + 1 when a is **not** near an
    integer.  Callers should check `_is_near_integer(a)` first.
    """
    var gln = lgamma(a)

    # Set up for evaluating continued fraction by modified Lentz's method.
    var b = x + 1.0 - a
    var c = 1.0 / _FPMIN
    var d = 1.0 / b
    var h = d

    for i in range(1, _CF_MAX_ITER + 1):
        var an = -Float64(i) * (Float64(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        var del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < _EPS:
            return exp(-x + a * log(x) - gln) * h

    # Continued fraction did not converge; return best estimate.
    return exp(-x + a * log(x) - gln) * h
