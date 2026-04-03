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
- ``gmean`` — Geometric mean
- ``hmean`` — Harmonic mean
"""

from math import sqrt, nan, log, exp


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


fn stddev(data: List[Float64], ddof: Int = 0) -> Float64:
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
    var s = stddev(data, ddof=1)
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


# TODO: Due to limitation in mojo compiler in 0.26.1 (resolved in nightly), we can't have Optional[List].
# Once we have that, we can make weights optional and handle the unweighted case more cleanly.
# For now, we can just require an empty list for unweighted case.
fn gmean(data: List[Float64], weights: List[Float64]) -> Float64:
    """Compute the weighted geometric mean of a list of values.

    The geometric mean is the nth root of the product of n values. If weights are provided,
    computes the weighted geometric mean using the formula:

        exp(Σ(wᵢ · log(xᵢ)) / Σwᵢ)

    where xᵢ are the data values and wᵢ are the corresponding weights.

    Args:
        data: A list of values, which must all be non-negative.
        weights: A list of weights corresponding to each value in `data`.
            If weights are provided, they must be the same length as `data`,
            all weights must be non-negative, and the sum of weights must be greater than zero.
            If an empty list is provided for weights, all values in `data` are treated as equally weighted.

    Returns:
        The weighted geometric mean of the values.
        If `data` is empty, or if weights are provided and have a different length than `data`,
        or if any weight is negative, or if the total weight is zero,
        or if any value in `data` is negative, the function returns NaN.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()

    var ws = len(weights) > 0
    if ws:
        if len(weights) != n:
            return nan[DType.float64]()
        var total: Float64 = 0.0
        for w in weights:
            if w < 0.0:
                return nan[DType.float64]()
            total += w

        if total == 0.0:
            return nan[DType.float64]()
        var log_sum: Float64 = 0.0
        for i in range(n):
            if data[i] < 0.0:
                return nan[DType.float64]()
            log_sum += weights[i] * log(data[i])
        return exp(log_sum / total)

    var log_sum: Float64 = 0.0
    for i in range(n):
        if data[i] < 0.0:
            return nan[DType.float64]()
        log_sum += log(data[i])
    return exp(log_sum / Float64(n))


fn hmean(data: List[Float64], weights: List[Float64]) -> Float64:
    """
    Compute the weighted harmonic mean of a list of values.

    The harmonic mean is defined as n / (Σ(1/xᵢ)) for n values. If weights are provided,
    computes the weighted harmonic mean using the formula:

        Σwᵢ / Σ(wᵢ / xᵢ)

    where xᵢ are the data values and wᵢ are the corresponding weights.

    Args:
        data: A list of positive values (must all be strictly greater than zero).
        weights: A list of weights corresponding to each value in `data`.
            If weights are provided, they must be the same length as `data`,
            all weights must be non-negative, and the sum of weights must be greater than zero.
            If an empty list is provided for weights, all values in `data` are treated as equally weighted.

    Returns:
        The weighted harmonic mean of the values.
        If `data` is empty, or if weights are provided and have a different length than `data`,
        or if any weight is negative, or if the total weight is zero,
        or if any value in `data` is zero or negative, the function returns NaN.
    """
    var n = len(data)
    if n == 0:
        return nan[DType.float64]()

    var ws = len(weights) > 0
    if ws:
        if len(weights) != n:
            return nan[DType.float64]()
        var total: Float64 = 0.0
        for w in weights:
            if w < 0.0:
                return nan[DType.float64]()
            total += w

        if total == 0.0:
            return nan[DType.float64]()
        var inv_sum: Float64 = 0.0
        for i in range(n):
            if data[i] <= 0.0:
                return nan[DType.float64]()
            inv_sum += weights[i] / data[i]
        return total / inv_sum

    var inv_sum: Float64 = 0.0
    for i in range(n):
        if data[i] <= 0.0:
            return nan[DType.float64]()
        inv_sum += 1.0 / data[i]
    return Float64(n) / inv_sum
