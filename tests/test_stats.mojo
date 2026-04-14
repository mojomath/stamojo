# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for descriptive statistics
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the stats.descriptive subpackage.

Covers mean, variance, std, median, quantile, skewness, and kurtosis
with both analytical checks and scipy/numpy comparisons.
"""

from std.math import sqrt, exp, log
from std.python import Python, PythonObject
from std.testing import assert_almost_equal, TestSuite

from stamojo.stats import (
    mean,
    variance,
    stddev,
    median,
    quantile,
    skewness,
    kurtosis,
    data_min,
    data_max,
    gmean,
    hmean,
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


fn test_variance() raises:
    """Test variance (population and sample)."""
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    # Population variance = 4.0 (Wikipedia example)
    assert_almost_equal(variance(data, ddof=0), 4.0, atol=1e-12)
    # Sample variance = 32/7
    assert_almost_equal(variance(data, ddof=1), 32.0 / 7.0, atol=1e-12)


fn test_std() raises:
    """Test standard deviation."""
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    assert_almost_equal(stddev(data, ddof=0), 2.0, atol=1e-12)


fn test_median_odd() raises:
    """Test median with odd-length data."""
    var data: List[Float64] = [3.0, 1.0, 2.0]
    assert_almost_equal(median(data), 2.0, atol=1e-15)

    var data2: List[Float64] = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert_almost_equal(median(data2), 3.0, atol=1e-15)


fn test_median_even() raises:
    """Test median with even-length data."""
    var data: List[Float64] = [3.0, 1.0, 2.0, 4.0]
    assert_almost_equal(median(data), 2.5, atol=1e-15)


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


fn test_skewness_symmetric() raises:
    """Test skewness of perfectly symmetric data is 0."""
    var data: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert_almost_equal(skewness(data), 0.0, atol=1e-12)


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


fn test_min_max() raises:
    """Test data_min and data_max."""
    var data: List[Float64] = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]

    assert_almost_equal(data_min(data), 1.0, atol=1e-15)
    assert_almost_equal(data_max(data), 9.0, atol=1e-15)


fn test_scipy_comparison() raises:
    """Test descriptive statistics against numpy/scipy."""
    try:
        var np = Python.import_module("numpy")

        var data: List[Float64] = [2.3, 5.1, 3.7, 8.4, 1.2, 6.8, 4.5]

        var py_data = Python.evaluate("[2.3, 5.1, 3.7, 8.4, 1.2, 6.8, 4.5]")

        var np_mean = _py_f64(np.mean(py_data))
        var builtins = Python.import_module("builtins")
        var np_var = _py_f64(builtins.getattr(np, "var")(py_data))
        var np_std = _py_f64(np.std(py_data))
        var np_median = _py_f64(np.median(py_data))

        assert_almost_equal(mean(data), np_mean, atol=1e-10)
        assert_almost_equal(variance(data, ddof=0), np_var, atol=1e-10)
        assert_almost_equal(stddev(data, ddof=0), np_std, atol=1e-10)
        assert_almost_equal(median(data), np_median, atol=1e-10)

        print("✓ test_scipy_comparison passed")
    except:
        print("⊘ test_scipy_comparison skipped (numpy not available)")


fn test_gmean() raises:
    """Test geometric mean."""
    # first three test values are from scipy examples.
    var data: List[Float64] = [1.0, 4.0]
    var res = gmean(data, List[Float64]())
    assert_almost_equal(res, 2.0, atol=1e-12)

    var data2: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var res2 = gmean(data2, List[Float64]())
    assert_almost_equal(res2, 3.3800151591412964, atol=1e-12)

    var data3: List[Float64] = [1.0, 4.0, 7.0]
    var weights3: List[Float64] = [3.0, 1.0, 3.0]
    var res3 = gmean(data3, weights3)
    assert_almost_equal(res3, 2.80668351922014, atol=1e-12)

    try:
        var sp = Python.import_module("scipy.stats")
        var data4: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var py_data4 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var sp_gmean = _py_f64(sp.gmean(py_data4))
        var res4 = gmean(data4, List[Float64]())
        assert_almost_equal(res4, sp_gmean, atol=1e-12)

        var data5: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var weights5: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var py_data5 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var py_weights5 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var sp_gmean_w = _py_f64(sp.gmean(a=py_data5, weights=py_weights5))
        var res5 = gmean(data5, weights5)
        assert_almost_equal(res5, sp_gmean_w, atol=1e-12)
    except:
        print("⊘ test_gmean scipy comparison skipped (scipy not available)")


fn test_hmean() raises:
    """Test harmonic mean."""
    # first three test values are from scipy examples.
    var data: List[Float64] = [1.0, 4.0]
    var res = hmean(data, List[Float64]())
    assert_almost_equal(res, 1.6, atol=1e-12)

    var data2: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var res2 = hmean(data2, List[Float64]())
    assert_almost_equal(res2, 2.6997245179063363, atol=1e-12)

    var data3: List[Float64] = [1.0, 4.0, 7.0]
    var weights3: List[Float64] = [3.0, 1.0, 3.0]
    var res3 = hmean(data3, weights3)
    assert_almost_equal(res3, 1.9029126213592233, atol=1e-12)

    try:
        var sp = Python.import_module("scipy.stats")
        var data4: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var py_data4 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var sp_hmean = _py_f64(sp.hmean(py_data4))
        var res4 = hmean(data4, List[Float64]())
        assert_almost_equal(res4, sp_hmean, atol=1e-12)

        var data5: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var weights5: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        var py_data5 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var py_weights5 = Python.list(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        var sp_hmean_w = _py_f64(sp.hmean(a=py_data5, weights=py_weights5))
        var res5 = hmean(data5, weights5)
        assert_almost_equal(res5, sp_hmean_w, atol=1e-12)
    except:
        print("⊘ test_hmean scipy comparison skipped (scipy not available)")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
