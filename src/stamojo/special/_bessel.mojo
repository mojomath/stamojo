# ===----------------------------------------------------------------------=== #
# Stamojo - Special - Bessel functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Bessel functions.

This module provides implementations of Bessel functions of the first and
second kind, as well as modified Bessel functions and their exponentially
scaled variants.

Functions:
    - j0, j1, jn: Bessel functions of the first kind (orders 0, 1, n)
    - i0, i1, i0e, i1e: Modified Bessel functions of the first kind
    - y0, y1: Bessel functions of the second kind (orders 0, 1)

References:
    - https://en.wikipedia.org/wiki/Bessel_function
"""

from std.math import cos, exp, inf, log, nan, sin, sqrt

# === --------------------------------------------------------------------=== #
# General notes:
# TODO: Asymptotic expansions need to be implemented for large arguments to
# ensure accuracy and efficiency. The threshold for switching to asymptotic
# expansions should be determined empirically based on accuracy requirements.
# === ----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _MAX_SERIES_ITER: Int = 50
comptime _PI: Float64 = 3.141592653589793
comptime _PI_INV: Float64 = 1.0 / _PI
comptime _EULER_GAMMA: Float64 = 0.5772156649015328606

# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


fn _factorial(n: Int) -> Float64:
    var res = 1.0
    for i in range(2, n + 1):
        res *= Float64(i)
    return res


# ===----------------------------------------------------------------------=== #
# Bessel functions of the first kind
# ===----------------------------------------------------------------------=== #


fn j0(x: Float64) -> Float64:
    """Bessel function of the first kind of order 0.

    Args:
        x: Input value.

    Returns:
        J₀(x).

    Examples:
        ```mojo
        from stamojo.special import j0
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(j0(1.0), 0.7651976865579666, atol=1e-12)
        ```
    """
    # J₀ is even: J₀(-x) = J₀(x).
    var ax = abs(x)

    # TODO: Determine asymptotic threshold empirically.
    if ax > 10.0:
        var t = ax - _PI * 0.25
        return sqrt(2.0 * _PI_INV / ax) * cos(t)

    var term = 1.0
    var res = term
    var x2 = ax * ax * 0.25
    for k in range(1, _MAX_SERIES_ITER):
        term *= -x2 / (Float64(k) * Float64(k))
        res += term
    return res


fn j1(x: Float64) -> Float64:
    """Bessel function of the first kind of order 1.

    Args:
        x: Input value.

    Returns:
        J₁(x).

    Examples:
        ```mojo
        from stamojo.special import j1
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(j1(1.0), 0.44005058574493355, atol=1e-12)
        ```
    """
    # J₁ is odd: J₁(-x) = -J₁(x).
    var ax = abs(x)
    var sign: Float64 = 1.0 if x >= 0.0 else -1.0

    # TODO: Determine asymptotic threshold empirically.
    if ax > 10.0:
        var t = ax - 3.0 * _PI * 0.25
        return sign * sqrt(2.0 * _PI_INV / ax) * cos(t)

    var term = ax * 0.5
    var res = term
    var x2 = ax * ax * 0.25
    for k in range(1, _MAX_SERIES_ITER):
        term *= -x2 / (Float64(k) * Float64(k + 1))
        res += term
    return sign * res


fn jn[n: Int](x: Float64) -> Float64:
    """Bessel function of the first kind of order *n*.

    Parameters:
        n: Order of the Bessel function (integer).

    Args:
        x: Input value.

    Returns:
        Jₙ(x).
    """

    comptime if n == 0:
        return j0(x)

    comptime if n == 1:
        return j1(x)

    comptime m = n if n >= 0 else -n
    comptime sign: Float64 = -1.0 if (n < 0 and (m % 2 == 1)) else 1.0

    var ax = abs(x)

    # For m <= ax, use the forward recurrence:
    # J_{k+1}(x) = (2k/x) J_k(x) - J_{k-1}(x)
    if Float64(m) <= ax:
        var jm1 = j0(x)  # J_0
        var jcur = j1(x)  # J_1
        for k in range(1, m):
            var jnext = (2.0 * Float64(k) / x) * jcur - jm1
            jm1 = jcur
            jcur = jnext
        return sign * jcur

    # For m > ax, use power series.
    var fact = _factorial(m)
    var term = 1.0
    for _ in range(m):
        term *= x * 0.5
    term /= fact

    var res = term
    var x2 = x * x * 0.25
    for k in range(1, _MAX_SERIES_ITER):
        term *= -x2 / (Float64(k) * Float64(k + m))
        res += term
    return sign * res


# ===----------------------------------------------------------------------=== #
# Modified Bessel functions of the first kind and their scaled forms
# ===----------------------------------------------------------------------=== #


fn i0(x: Float64) -> Float64:
    """Modified Bessel function of the first kind of order 0.

    Args:
        x: Input value.

    Returns:
        I₀(x).

    Examples:
        ```mojo
        from stamojo.special import i0
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(i0(1.0), 1.2660658777520082, atol=1e-12)
        ```
    """
    var term = 1.0
    var res = term
    var x2 = x * x * 0.25
    for k in range(1, _MAX_SERIES_ITER):
        term *= x2 / (Float64(k) * Float64(k))
        res += term
    return res


fn i1(x: Float64) -> Float64:
    """Modified Bessel function of the first kind of order 1.

    Args:
        x: Input value.

    Returns:
        I₁(x).

    Examples:
        ```mojo
        from stamojo.special import i1
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(i1(1.0), 0.5651591039924851, atol=1e-12)
        ```
    """
    var term = x * 0.5
    var res = term
    var x2 = x * x * 0.25
    for k in range(1, _MAX_SERIES_ITER):
        term *= x2 / (Float64(k) * Float64(k + 1))
        res += term
    return res


fn i0e(x: Float64) -> Float64:
    """Exponentially scaled modified Bessel function of the first kind
    of order 0: ``i0e(x) = exp(-|x|) * i0(x)``.

    Args:
        x: Input value.

    Returns:
        Value of exp(-|x|) * I₀(x).

    Examples:
        ```mojo
        from stamojo.special import i0e
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(i0e(1.0), 0.4657596075936405, atol=1e-12)
        ```
    """
    return i0(x) * exp(-abs(x))


fn i1e(x: Float64) -> Float64:
    """Exponentially scaled modified Bessel function of the first kind
    of order 1: ``i1e(x) = exp(-|x|) * i1(x)``.

    Args:
        x: Input value.

    Returns:
        Value of exp(-|x|) * I₁(x).

    Examples:
        ```mojo
        from stamojo.special import i1e
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(i1e(1.0), 0.2079104153497085, atol=1e-12)
        ```
    """
    return i1(x) * exp(-abs(x))


# ===----------------------------------------------------------------------=== #
# Bessel functions of the second kind
# ===----------------------------------------------------------------------=== #


fn y0(x: Float64) -> Float64:
    """Bessel function of the second kind of order 0.

    Defined for x > 0.  Returns -∞ at x = 0 and NaN for x < 0.

    Args:
        x: Input value (must be positive).

    Returns:
        Y₀(x).

    Examples:
        ```mojo
        from stamojo.special import y0
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(y0(1.0), 0.08825696421567697, atol=1e-12)
        ```
    """
    if x == 0.0:
        return -inf[DType.float64]()
    if x < 0.0:
        return nan[DType.float64]()

    if x < 8.0:
        var j0x = j0(x)
        var x2 = x * x * 0.25
        var term = x2
        var sum = 0.0
        var h = 1.0
        for k in range(1, _MAX_SERIES_ITER):
            if k > 1:
                h += 1.0 / Float64(k)
            sum += term * h
            term *= -x2 / (Float64(k + 1) * Float64(k + 1))
        return (2.0 / _PI) * ((log(x * 0.5) + _EULER_GAMMA) * j0x + sum)

    var t = x - _PI * 0.25
    return sqrt(2.0 / (_PI * x)) * sin(t)


fn y1(x: Float64) -> Float64:
    """Bessel function of the second kind of order 1.

    Defined for x > 0.  Returns -∞ at x = 0 and NaN for x < 0.

    Args:
        x: Input value (must be positive).

    Returns:
        Y₁(x).

    Examples:
        ```mojo
        from stamojo.special import y1
        from std.testing import assert_almost_equal

        fn main() raises:
            assert_almost_equal(y1(1.0), -0.7812128213002887, atol=1e-12)
        ```
    """
    if x == 0.0:
        return -inf[DType.float64]()
    if x < 0.0:
        return nan[DType.float64]()

    if x < 8.0:
        var j1x = j1(x)
        var x2 = x * x * 0.25
        var term = x * 0.5
        var sum = 0.0
        var hk = 0.0

        for k in range(_MAX_SERIES_ITER):
            var hk1 = hk + 1.0 / Float64(k + 1)
            sum += term * (hk + hk1)
            term *= -x2 / (Float64(k + 1) * Float64(k + 2))
            hk = hk1

        return (
            (-2.0 * _PI_INV / x)
            + (2.0 * _PI_INV) * (log(x * 0.5) + _EULER_GAMMA) * j1x
            - sum * _PI_INV
        )

    var t = x - 3.0 * _PI * 0.25
    return sqrt(2.0 / (_PI * x)) * sin(t)
