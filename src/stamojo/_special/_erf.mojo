# ===----------------------------------------------------------------------=== #
# StaMojo - Inverse error function and inverse normal CDF
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Inverse error function and related functions.

Provides:
- `erfinv(p)`: Inverse of the error function, so that erf(erfinv(p)) = p.
- `ndtri(p)`:  Inverse of the standard normal CDF (quantile / PPF).

The implementation uses Peter Acklam's rational approximation for the
inverse normal CDF, which achieves ~1.15 × 10⁻⁹ absolute error, then
refines with Newton–Raphson steps to reach full Float64 precision.

    erfinv(p) = ndtri((1 + p) / 2) / √2

Reference:
    Peter J. Acklam, "An algorithm for computing the inverse normal
    cumulative distribution function," 2010.
    https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
"""

from math import sqrt, log, exp, erf, erfc, nan, inf


# ===----------------------------------------------------------------------=== #
# Acklam coefficients for the inverse normal CDF
# ===----------------------------------------------------------------------=== #

# Break-points for the three regions.
comptime _P_LOW = 0.02425
comptime _P_HIGH = 1.0 - _P_LOW  # 0.97575

# Coefficients for the central region rational approximation.
comptime _A1 = -3.969683028665376e1
comptime _A2 = 2.209460984245205e2
comptime _A3 = -2.759285104469687e2
comptime _A4 = 1.383577518672690e2
comptime _A5 = -3.066479806614716e1
comptime _A6 = 2.506628277459239e0

comptime _B1 = -5.447609879822406e1
comptime _B2 = 1.615858368580409e2
comptime _B3 = -1.556989798598866e2
comptime _B4 = 6.680131188771972e1
comptime _B5 = -1.328068155288572e1

# Coefficients for the tail region rational approximation.
comptime _C1 = -7.784894002430293e-3
comptime _C2 = -3.223964580411365e-1
comptime _C3 = -2.400758277161838e0
comptime _C4 = -2.549732539343734e0
comptime _C5 = 4.374664141464968e0
comptime _C6 = 2.938163982698783e0

comptime _D1 = 7.784695709041462e-3
comptime _D2 = 3.224671290700398e-1
comptime _D3 = 2.445134137142996e0
comptime _D4 = 3.754408661907416e0


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn ndtri(p: Float64) -> Float64:
    """Inverse of the standard normal CDF (quantile / PPF).

    Computes x such that Φ(x) = p, where Φ is the CDF of N(0,1).

    Args:
        p: Probability value in (0, 1).

    Returns:
        The standard normal quantile.  Returns -inf for p=0, +inf for p=1,
        NaN for p outside [0, 1].
    """
    if p < 0.0 or p > 1.0:
        return nan[DType.float64]()
    if p == 0.0:
        return -inf[DType.float64]()
    if p == 1.0:
        return inf[DType.float64]()
    if p == 0.5:
        return 0.0

    var x: Float64

    if _P_LOW <= p and p <= _P_HIGH:
        # Central region.
        var q = p - 0.5
        var r = q * q
        x = (
            ((((_A1 * r + _A2) * r + _A3) * r + _A4) * r + _A5) * r + _A6
        ) * q / (
            ((((_B1 * r + _B2) * r + _B3) * r + _B4) * r + _B5) * r + 1.0
        )
    elif p < _P_LOW:
        # Lower tail.
        var q = sqrt(-2.0 * log(p))
        x = (
            ((((_C1 * q + _C2) * q + _C3) * q + _C4) * q + _C5) * q + _C6
        ) / (
            (((_D1 * q + _D2) * q + _D3) * q + _D4) * q + 1.0
        )
    else:
        # Upper tail.
        var q = sqrt(-2.0 * log(1.0 - p))
        x = -(
            ((((_C1 * q + _C2) * q + _C3) * q + _C4) * q + _C5) * q + _C6
        ) / (
            (((_D1 * q + _D2) * q + _D3) * q + _D4) * q + 1.0
        )

    # One Newton–Raphson refinement step for full double precision.
    # Φ(x) = 0.5 * erfc(-x / √2)
    # Φ'(x) = (1/√(2π)) * exp(-x²/2)
    comptime INV_SQRT_2PI = 0.3989422804014326779399460599343819
    var phi = 0.5 * erfc(-x / sqrt(2.0))
    var phi_prime = INV_SQRT_2PI * exp(-0.5 * x * x)
    x = x - (phi - p) / phi_prime

    return x


fn erfinv(p: Float64) -> Float64:
    """Inverse error function.

    Computes the value x such that erf(x) = p.

    Args:
        p: Input value. Must be in the range (-1, 1).

    Returns:
        The value x such that erf(x) = p.
        Returns NaN if p is outside (-1, 1).
        Returns -inf if p == -1, +inf if p == 1.
    """
    if p < -1.0 or p > 1.0:
        return nan[DType.float64]()
    if p == -1.0:
        return -inf[DType.float64]()
    if p == 1.0:
        return inf[DType.float64]()
    if p == 0.0:
        return 0.0

    # erfinv(p) = ndtri((1 + p) / 2) / sqrt(2)
    comptime INV_SQRT2 = 0.7071067811865475244008443621048490
    return ndtri((1.0 + p) / 2.0) * INV_SQRT2
