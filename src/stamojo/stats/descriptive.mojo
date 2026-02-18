# ===----------------------------------------------------------------------=== #
# Stamojo - Stats - Descriptive statistics
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Descriptive statistics functions.

Provides functions for computing summary statistics of ``List[Float64]`` data:

- ``mean``     — Arithmetic mean
- ``variance`` — Variance (population or sample via *ddof*)
- ``std``      — Standard deviation
- ``median``   — Median
- ``quantile`` — Quantile (linear interpolation, same as NumPy default)
- ``skewness`` — Fisher's skewness (bias-corrected)
- ``kurtosis`` — (Excess) kurtosis (bias-corrected)
- ``data_min`` — Minimum value
- ``data_max`` — Maximum value
"""

from math import sqrt, nan


# ===----------------------------------------------------------------------=== #
# Internal helpers
# ===----------------------------------------------------------------------=== #


fn _sorted_copy(data: List[Float64]) -> List[Float64]:
    """Return a sorted copy of *data* (ascending, insertion sort)."""
    var result = data.copy()
    var n = len(result)
    for i in range(1, n):
        var key = result[i]
        var j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn mean(data: List[Float64]) -> Float64:
    """Arithmetic mean of *data*.

    Args:
        data: A list of values.

    Returns:
        The arithmetic mean.  Returns NaN for an empty list.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()
    var s = 0.0
    for i in range(n):
        s += data[i]
    return s / Float64(n)


fn variance(data: List[Float64], ddof: Int = 0) -> Float64:
    """Variance of *data*.

    Args:
        data: A list of values.
        ddof: Delta degrees of freedom.  Use 0 for population variance,
              1 for sample variance.  Default is 0.

    Returns:
        The variance.  Returns NaN if ``len(data) <= ddof``.
    """
    var n = len(data)
    if n <= ddof:
        return nan[DType.float64]()
    var m = mean(data)
    var ss = 0.0
    for i in range(n):
        var d = data[i] - m
        ss += d * d
    return ss / Float64(n - ddof)


fn std(data: List[Float64], ddof: Int = 0) -> Float64:
    """Standard deviation of *data*.

    Args:
        data: A list of values.
        ddof: Delta degrees of freedom.  Default is 0.

    Returns:
        The standard deviation.
    """
    return sqrt(variance(data, ddof))


fn median(data: List[Float64]) -> Float64:
    """Median of *data*.

    Args:
        data: A list of values.

    Returns:
        The median.  Returns NaN for an empty list.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()

    var sorted_data = _sorted_copy(data)

    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0


fn quantile(data: List[Float64], q: Float64) -> Float64:
    """Quantile of *data* using linear interpolation (NumPy default).

    Args:
        data: A list of values.
        q: Quantile to compute, must be in [0, 1].

    Returns:
        The *q*-th quantile.  Returns NaN for invalid inputs.
    """
    var n = len(data)
    if n == 0 or q < 0.0 or q > 1.0:
        return nan[DType.float64]()

    var sorted_data = _sorted_copy(data)

    if q == 0.0:
        return sorted_data[0]
    if q == 1.0:
        return sorted_data[n - 1]

    var idx = q * Float64(n - 1)
    var lo = Int(idx)
    var hi = lo + 1
    if hi >= n:
        return sorted_data[n - 1]
    var frac = idx - Float64(lo)
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


fn skewness(data: List[Float64]) -> Float64:
    """Fisher's skewness (bias-corrected) of *data*.

    Computes the adjusted Fisher-Pearson standardized moment coefficient::

        G₁ = n / ((n−1)(n−2)) · Σ((xᵢ − x̄) / s)³

    where *s* is the sample standard deviation (ddof=1).

    Args:
        data: A list of values.

    Returns:
        The skewness.  Returns NaN if ``n < 3``.
    """
    var n = len(data)
    if n < 3:
        return nan[DType.float64]()

    var m = mean(data)
    var s = std(data, ddof=1)
    if s == 0.0:
        return 0.0

    var m3 = 0.0
    for i in range(n):
        var z = (data[i] - m) / s
        m3 += z * z * z

    var fn_ = Float64(n)
    return m3 * fn_ / ((fn_ - 1.0) * (fn_ - 2.0))


fn kurtosis(data: List[Float64], excess: Bool = True) -> Float64:
    """Kurtosis of *data* (bias-corrected).

    Uses the standard bias-corrected formula matching ``scipy.stats.kurtosis``
    with ``fisher=True, bias=False``.

    Args:
        data: A list of values.
        excess: If True (default), return excess kurtosis (normal = 0).
                If False, return regular kurtosis (normal = 3).

    Returns:
        The kurtosis.  Returns NaN if ``n < 4``.
    """
    var n = len(data)
    if n < 4:
        return nan[DType.float64]()

    var m = mean(data)
    var s2 = 0.0
    var s4 = 0.0
    for i in range(n):
        var d = data[i] - m
        var d2 = d * d
        s2 += d2
        s4 += d2 * d2

    var fn_ = Float64(n)
    # Bias-corrected excess kurtosis:
    # G₂ = [(n−1)/((n−2)(n−3))] · [(n+1)·n·S₄/S₂² − 3(n−1)]
    var kurt = (
        (fn_ * (fn_ + 1.0) * s4 / (s2 * s2) - 3.0 * (fn_ - 1.0))
        * (fn_ - 1.0)
        / ((fn_ - 2.0) * (fn_ - 3.0))
    )

    if excess:
        return kurt
    else:
        return kurt + 3.0


fn data_min(data: List[Float64]) -> Float64:
    """Minimum value in *data*.

    Args:
        data: A list of values.

    Returns:
        The minimum value.  Returns NaN for an empty list.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()
    var result = data[0]
    for i in range(1, n):
        if data[i] < result:
            result = data[i]
    return result


fn data_max(data: List[Float64]) -> Float64:
    """Maximum value in *data*.

    Args:
        data: A list of values.

    Returns:
        The maximum value.  Returns NaN for an empty list.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()
    var result = data[0]
    for i in range(1, n):
        if data[i] > result:
            result = data[i]
    return result
