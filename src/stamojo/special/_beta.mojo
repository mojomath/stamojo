# ===----------------------------------------------------------------------=== #
# StaMojo - Beta function and regularized incomplete beta function
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Beta function and regularized incomplete beta function.

Provides:
- `beta(a, b)`: Beta function B(a, b) = Γ(a)Γ(b) / Γ(a+b).
- `lbeta(a, b)`: Log of the beta function ln(B(a, b)).
- `betainc(a, b, x)`: Regularized incomplete beta function I_x(a, b).

The regularized incomplete beta function is defined as:
    I_x(a, b) = B(x; a, b) / B(a, b)

where B(x; a, b) = ∫₀ˣ t^{a-1} (1-t)^{b-1} dt is the incomplete beta
function and B(a, b) is the (complete) beta function.

The implementation uses the continued fraction representation of the
incomplete beta function, following Numerical Recipes.

Reference:
    Press et al., Numerical Recipes, 3rd ed., Section 6.4.
"""

from math import lgamma, exp, log, nan


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 3.0e-12
comptime _MAX_ITER = 200
comptime _FPMIN = 1.0e-30


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn lbeta(a: Float64, b: Float64) -> Float64:
    """Natural logarithm of the beta function.

    Computes ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b)).

    Args:
        a: First parameter. Must be positive.
        b: Second parameter. Must be positive.

    Returns:
        The value of ln(B(a, b)).
    """
    return lgamma(a) + lgamma(b) - lgamma(a + b)


fn beta(a: Float64, b: Float64) -> Float64:
    """Beta function B(a, b) = Γ(a)Γ(b) / Γ(a+b).

    Args:
        a: First parameter. Must be positive.
        b: Second parameter. Must be positive.

    Returns:
        The value of the beta function B(a, b).
    """
    return exp(lbeta(a, b))


fn betainc(a: Float64, b: Float64, x: Float64) -> Float64:
    """Regularized incomplete beta function I_x(a, b).

    Computes I_x(a, b) = B(x; a, b) / B(a, b), where B(x; a, b) is the
    incomplete beta function.

    Uses the continued fraction representation with the symmetry relation
    I_x(a, b) = 1 - I_{1-x}(b, a) for efficiency.

    Args:
        a: First shape parameter. Must be positive.
        b: Second shape parameter. Must be positive.
        x: Upper limit of integration. Must be in [0, 1].

    Returns:
        The value of the regularized incomplete beta function, in [0, 1].
    """
    if x < 0.0 or x > 1.0 or a <= 0.0 or b <= 0.0:
        return nan[DType.float64]()

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    var lb = lbeta(a, b)

    # Use the symmetry relation to ensure the continued fraction converges
    # quickly. The CF converges faster when x < (a + 1) / (a + b + 2).
    if x < (a + 1.0) / (a + b + 2.0):
        return _betainc_cf(a, b, x, lb)
    else:
        return 1.0 - _betainc_cf(b, a, 1.0 - x, lb)


# ===----------------------------------------------------------------------=== #
# Internal implementations
# ===----------------------------------------------------------------------=== #


fn _betainc_cf(
    a: Float64, b: Float64, x: Float64, lbeta_ab: Float64
) -> Float64:
    """Evaluate the regularized incomplete beta function using Lentz's
    continued fraction method.

    Computes:
        I_x(a, b) = x^a (1-x)^b / (a B(a,b)) * CF(a, b, x)

    where CF is the continued fraction representation.
    """
    # The front factor x^a * (1-x)^b / (a * B(a,b)).
    var front = exp(a * log(x) + b * log(1.0 - x) - lbeta_ab) / a

    # Modified Lentz's method for the continued fraction.
    var c = 1.0
    var d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    var h = d

    for m in range(1, _MAX_ITER + 1):
        var fm = Float64(m)

        # Even step: d_{2m}
        var aa = fm * (b - fm) * x / ((a + 2.0 * fm - 1.0) * (a + 2.0 * fm))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c

        # Odd step: d_{2m+1}
        aa = (
            -(a + fm)
            * (a + b + fm)
            * x
            / ((a + 2.0 * fm) * (a + 2.0 * fm + 1.0))
        )
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        var del_val = d * c
        h *= del_val

        if abs(del_val - 1.0) < _EPS:
            return front * h

    # Did not converge; return best estimate.
    return front * h
