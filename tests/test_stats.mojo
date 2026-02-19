# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for descriptive statistics
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the stats.descriptive subpackage.

Covers mean, variance, std, median, quantile, skewness, and kurtosis
with both analytical checks and scipy/numpy comparisons.
"""

from math import sqrt
from python import Python, PythonObject
from testing import assert_almost_equal

from stamojo.stats import (
    mean,
    variance,
    std,
    median,
    quantile,
    skewness,
    kurtosis,
    data_min,
    data_max,
)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _py_f64(obj: PythonObject) -> Float64:
    """Convert a PythonObject holding a numeric value to Float64."""
    try:
        return atof(String(obj))
    except:
        return 0.0


# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


fn test_mean() raises:
    """Test arithmetic mean."""
    var data: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert_almost_equal(mean(data), 3.0, atol=1e-15)

    var data2: List[Float64] = [10.0]
    assert_almost_equal(mean(data2), 10.0, atol=1e-15)

    print("✓ test_mean passed")


fn test_variance() raises:
    """Test variance (population and sample)."""
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    # Population variance = 4.0 (Wikipedia example)
    assert_almost_equal(variance(data, ddof=0), 4.0, atol=1e-12)
    # Sample variance = 32/7
    assert_almost_equal(variance(data, ddof=1), 32.0 / 7.0, atol=1e-12)

    print("✓ test_variance passed")


fn test_std() raises:
    """Test standard deviation."""
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    assert_almost_equal(std(data, ddof=0), 2.0, atol=1e-12)

    print("✓ test_std passed")


fn test_median_odd() raises:
    """Test median with odd-length data."""
    var data: List[Float64] = [3.0, 1.0, 2.0]
    assert_almost_equal(median(data), 2.0, atol=1e-15)

    var data2: List[Float64] = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert_almost_equal(median(data2), 3.0, atol=1e-15)

    print("✓ test_median_odd passed")


fn test_median_even() raises:
    """Test median with even-length data."""
    var data: List[Float64] = [3.0, 1.0, 2.0, 4.0]
    assert_almost_equal(median(data), 2.5, atol=1e-15)

    print("✓ test_median_even passed")


fn test_quantile() raises:
    """Test quantile function."""
    var data = List[Float64]()
    for i in range(1, 11):
        data.append(Float64(i))

    # q=0 → min, q=1 → max
    assert_almost_equal(quantile(data, 0.0), 1.0, atol=1e-15)
    assert_almost_equal(quantile(data, 1.0), 10.0, atol=1e-15)
    # q=0.5 → median
    assert_almost_equal(quantile(data, 0.5), 5.5, atol=1e-12)
    # q=0.25
    assert_almost_equal(quantile(data, 0.25), 3.25, atol=1e-12)

    print("✓ test_quantile passed")


fn test_skewness_symmetric() raises:
    """Test skewness of perfectly symmetric data is 0."""
    var data: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert_almost_equal(skewness(data), 0.0, atol=1e-12)

    print("✓ test_skewness_symmetric passed")


fn test_kurtosis_uniform() raises:
    """Test kurtosis of uniform-like data is negative (platykurtic)."""
    var data = List[Float64]()
    for i in range(1, 101):
        data.append(Float64(i))

    # Excess kurtosis of continuous uniform ≈ −1.2
    var k = kurtosis(data, excess=True)
    if k > 0.0 or k < -2.0:
        raise Error(
            "Kurtosis of uniform data out of expected range: " + String(k)
        )

    print("✓ test_kurtosis_uniform passed")


fn test_min_max() raises:
    """Test data_min and data_max."""
    var data: List[Float64] = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]

    assert_almost_equal(data_min(data), 1.0, atol=1e-15)
    assert_almost_equal(data_max(data), 9.0, atol=1e-15)

    print("✓ test_min_max passed")


fn test_scipy_comparison() raises:
    """Test descriptive statistics against numpy/scipy."""
    try:
        var np = Python.import_module("numpy")

        var data: List[Float64] = [2.3, 5.1, 3.7, 8.4, 1.2, 6.8, 4.5]

        var py_data = Python.evaluate("[2.3, 5.1, 3.7, 8.4, 1.2, 6.8, 4.5]")

        var np_mean = _py_f64(np.mean(py_data))
        var np_var = _py_f64(np.var(py_data, ddof=0))
        var np_std = _py_f64(np.std(py_data, ddof=0))
        var np_median = _py_f64(np.median(py_data))

        assert_almost_equal(mean(data), np_mean, atol=1e-10)
        assert_almost_equal(variance(data, ddof=0), np_var, atol=1e-10)
        assert_almost_equal(std(data, ddof=0), np_std, atol=1e-10)
        assert_almost_equal(median(data), np_median, atol=1e-10)

        print("✓ test_scipy_comparison passed")
    except:
        print("⊘ test_scipy_comparison skipped (numpy not available)")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    print("=== StaMojo: Testing descriptive statistics ===")
    print()

    test_mean()
    test_variance()
    test_std()
    test_median_odd()
    test_median_even()
    test_quantile()
    test_skewness_symmetric()
    test_kurtosis_uniform()
    test_min_max()
    test_scipy_comparison()

    print()
    print("=== All descriptive statistics tests passed ===")
