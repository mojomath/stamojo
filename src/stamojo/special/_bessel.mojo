# ===----------------------------------------------------------------------=== #
# StaMojo - Bessel
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Bessel functions
"""

from math import factorial
from math import cos, sin
from utils.numerics import inf

comptime _MAX_SERIES_ITER: Int = 10

fn j0[width: Int](x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """Bessel function of the first kind of order 0.

    Args:
        x: Input scalar.

    Returns:
        Bessel function of the first kind of order 0 evaluated at `x`

    Examples:
        ```mojo
        from stamojo.special import j0
        from testing import assert_equal

        fn main() raises:
            var x: Float64 = 1.0
            var res: Float64 = j0(x)
            assert_equal(res, 0.7651976865579666)
        ```
    """
    var res: SIMD[DType.float64, width] = 0.0
    for i in range(_MAX_SERIES_ITER):
        res += ((-1)**i / factorial(i)**2) * (x / 2.0)**(2 * i)

    return res

fn j1[width: Int](x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """Bessel function of the first kind of order 1.

    Args:
        x: Input scalar.

    Returns:
        Bessel function of the first kind of order 1 evaluated at `x`.

    Examples:
        ```mojo
        from stamojo.special import j1
        from testing import assert_equal

        fn main() raises:
            var x: Float64 = 1.0
            var res: Float64 = j1(x)
            assert_equal(res, 0.44005058574493355)
        ```
    """
    var res: SIMD[DType.float64, width] = 0.0
    for i in range(_MAX_SERIES_ITER):
        res += ((-1)**i / (factorial(i) * factorial(i + 1))) * (x / 2.0)**(2 * i + 1)
    return res

fn jn[width: Int](n: Int, x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """Bessel function of the first kind of order `n`.

    Args:
        n: Order of the Bessel function.
        x: Input scalar.

    Returns:
        Bessel function of the first kind of order `n` evaluated at `x`.
    """
    var res: SIMD[DType.float64, width] = 0.0
    for i in range(_MAX_SERIES_ITER):
        res += ((-1)**i / (factorial(i) * factorial(i + n))) * (x / 2.0)**(2 * i + n)
    return res

fn y0[width: Int](x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """Bessel function of the second kind of order 0.

    Args:
        x: Input scalar.

    Returns:
        Bessel function of the second kind of order 0 evaluated at `x`.

    Examples:
        ```mojo
        from stamojo.special import y0
        from testing import assert_equal

        fn main() raises:
            var x: Float64 = 1.0
            var res: Float64 = y0(x)
            assert_equal(res, 0.08825696421567697)
        ```
    """
    if x == 0.0:
        return inf[DType.float64]()

    comptime PI: Float64 = 3.141592653589793

    return (j1(x) * cos(PI * 1) + j1(x)) / sin(PI * 1)
