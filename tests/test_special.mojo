# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for special mathematical functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the special subpackage.

Two verification strategies are used:

1. **Analytical identities** – boundary values, symmetry relations, and
   closed-form cases (Poisson sum, exponential CDF, erf, etc.) that do not
   depend on any external library.

2. **scipy.special benchmarks** – when the test runs inside the ``test``
   pixi environment (which includes scipy), each result is additionally
   compared against scipy.special.  On failure the scipy reference value
   is printed for easy diagnosis.
"""

from math import exp, log, lgamma, erf, sqrt
from python import Python, PythonObject
from testing import assert_almost_equal, TestSuite

from stamojo.special import gammainc, gammaincc, beta, lbeta, betainc, erfinv


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _poisson_cdf(n: Int, x: Float64) -> Float64:
    """Exact Q(n, x) = e^{-x} Σ_{k=0}^{n-1} x^k/k! for positive integer n."""
    var term = 1.0
    var s = 1.0
    for k in range(1, n):
        term *= x / Float64(k)
        s += term
    return exp(-x) * s


fn _load_scipy() -> PythonObject:
    """Try to import scipy.special. Returns Python None if unavailable."""
    try:
        return Python.import_module("scipy.special")
    except:
        return PythonObject(None)


fn _py_f64(obj: PythonObject) -> Float64:
    """Convert a PythonObject holding a numeric value to Float64."""
    try:
        return atof(String(obj))
    except:
        return 0.0


fn _assert_with_scipy(
    actual: Float64,
    expected: Float64,
    sp: PythonObject,
    sp_val: Float64,
    label: String,
    atol: Float64 = 1e-12,
) raises:
    """Assert actual ≈ expected. On failure, print scipy reference if available."""
    try:
        assert_almost_equal(actual, expected, atol=atol)
    except e:
        if sp is not None:
            print(
                "  FAIL:", label,
                "\n    got:     ", actual,
                "\n    expected:", expected,
                "\n    scipy:   ", sp_val,
            )
        raise e^


# ===----------------------------------------------------------------------=== #
# Tests for regularized incomplete gamma
# ===----------------------------------------------------------------------=== #


fn test_gammainc_boundary() raises:
    """Test boundary conditions."""
    assert_almost_equal(gammainc(1.0, 0.0), 0.0, atol=1e-15)
    assert_almost_equal(gammainc(5.0, 0.0), 0.0, atol=1e-15)
    assert_almost_equal(gammaincc(1.0, 0.0), 1.0, atol=1e-15)
    assert_almost_equal(gammaincc(5.0, 0.0), 1.0, atol=1e-15)
    print("✓ test_gammainc_boundary passed")


fn test_gammainc_exponential() raises:
    """Test P(1, x) = 1 - e^{-x} (exponential distribution CDF)."""
    var sp = _load_scipy()
    var test_x: List[Float64] = [0.5, 1.0, 2.0, 5.0, 10.0]

    for i in range(len(test_x)):
        var x = test_x[i]
        var expected = 1.0 - exp(-x)
        var sp_val = _py_f64(sp.gammainc(1.0, x)) if sp is not None else 0.0
        _assert_with_scipy(
            gammainc(1.0, x), expected, sp, sp_val,
            "gammainc(1, " + String(x) + ")",
            atol=1e-12,
        )

    print("✓ test_gammainc_exponential passed")


fn test_gammainc_half() raises:
    """Test P(0.5, x) = erf(sqrt(x))."""
    var sp = _load_scipy()
    var test_x: List[Float64] = [0.25, 0.5, 1.0, 2.0, 4.0]

    for i in range(len(test_x)):
        var x = test_x[i]
        var expected = erf(sqrt(x))
        var sp_val = _py_f64(sp.gammainc(0.5, x)) if sp is not None else 0.0
        _assert_with_scipy(
            gammainc(0.5, x), expected, sp, sp_val,
            "gammainc(0.5, " + String(x) + ")",
            atol=1e-10,
        )

    print("✓ test_gammainc_half passed")


fn test_gammainc_integer_a() raises:
    """Test gammainc/gammaincc against the Poisson sum formula for integer a."""
    var sp = _load_scipy()

    var test_a: List[Int] = [2, 3, 5, 5, 10, 10, 20]
    var test_x: List[Float64] = [3.0, 2.0, 3.0, 10.0, 5.0, 15.0, 25.0]

    for i in range(len(test_a)):
        var a_int = test_a[i]
        var a = Float64(a_int)
        var x = test_x[i]

        var q_exact = _poisson_cdf(a_int, x)
        var p_exact = 1.0 - q_exact

        var sp_p = _py_f64(sp.gammainc(a, x)) if sp is not None else 0.0
        var sp_q = _py_f64(sp.gammaincc(a, x)) if sp is not None else 0.0

        _assert_with_scipy(
            gammainc(a, x), p_exact, sp, sp_p,
            "gammainc(" + String(a_int) + ", " + String(x) + ")",
        )
        _assert_with_scipy(
            gammaincc(a, x), q_exact, sp, sp_q,
            "gammaincc(" + String(a_int) + ", " + String(x) + ")",
        )

    print("✓ test_gammainc_integer_a passed")


fn test_gammainc_scipy() raises:
    """Test gammainc/gammaincc against scipy for non-integer a values."""
    var sp = _load_scipy()
    if sp is None:
        print("⊘ test_gammainc_scipy skipped (scipy not available)")
        return

    var test_a: List[Float64] = [0.7, 1.5, 2.5, 3.7, 7.3, 0.1, 15.5]
    var test_x: List[Float64] = [0.3, 2.0, 4.0, 1.2, 12.0, 0.01, 20.0]

    for i in range(len(test_a)):
        var a = test_a[i]
        var x = test_x[i]

        var sp_p = _py_f64(sp.gammainc(a, x))
        var sp_q = _py_f64(sp.gammaincc(a, x))

        _assert_with_scipy(
            gammainc(a, x), sp_p, sp, sp_p,
            "gammainc(" + String(a) + ", " + String(x) + ")",
            atol=1e-10,
        )
        _assert_with_scipy(
            gammaincc(a, x), sp_q, sp, sp_q,
            "gammaincc(" + String(a) + ", " + String(x) + ")",
            atol=1e-10,
        )

    print("✓ test_gammainc_scipy passed")


fn test_gammainc_complementary() raises:
    """Test P(a,x) + Q(a,x) = 1."""
    var test_cases: List[Tuple[Float64, Float64]] = [
        (0.5, 0.5), (1.0, 2.0), (2.5, 3.5), (3.0, 1.0),
        (5.0, 5.0), (5.0, 10.0), (10.0, 20.0), (0.1, 0.01),
    ]

    for i in range(len(test_cases)):
        var a = test_cases[i][0]
        var x = test_cases[i][1]
        assert_almost_equal(gammainc(a, x) + gammaincc(a, x), 1.0, atol=1e-12)

    print("✓ test_gammainc_complementary passed")


# ===----------------------------------------------------------------------=== #
# Tests for beta and incomplete beta
# ===----------------------------------------------------------------------=== #


fn test_beta_basic() raises:
    """Test beta function against known exact values."""
    assert_almost_equal(beta(1.0, 1.0), 1.0, atol=1e-12)
    assert_almost_equal(beta(2.0, 2.0), 1.0 / 6.0, atol=1e-12)
    assert_almost_equal(beta(0.5, 0.5), 3.141592653589793, atol=1e-10)
    assert_almost_equal(beta(3.0, 4.0), 1.0 / 60.0, atol=1e-12)

    var a = 3.7
    var b = 2.3
    var expected = exp(lgamma(a) + lgamma(b) - lgamma(a + b))
    assert_almost_equal(beta(a, b), expected, atol=1e-12)

    print("✓ test_beta_basic passed")


fn test_betainc_boundary() raises:
    """Test betainc boundary values."""
    assert_almost_equal(betainc(2.0, 3.0, 0.0), 0.0, atol=1e-15)
    assert_almost_equal(betainc(2.0, 3.0, 1.0), 1.0, atol=1e-15)
    assert_almost_equal(betainc(1.0, 1.0, 0.5), 0.5, atol=1e-12)
    print("✓ test_betainc_boundary passed")


fn test_betainc_symmetric() raises:
    """Test I_{0.5}(a, a) = 0.5."""
    var test_a: List[Float64] = [1.0, 2.0, 5.0, 10.0]

    for i in range(len(test_a)):
        var a = test_a[i]
        assert_almost_equal(betainc(a, a, 0.5), 0.5, atol=1e-10)

    print("✓ test_betainc_symmetric passed")


fn test_betainc_symmetry_identity() raises:
    """Test I_x(a,b) = 1 - I_{1-x}(b,a)."""
    assert_almost_equal(
        betainc(3.0, 5.0, 0.4),
        1.0 - betainc(5.0, 3.0, 0.6),
        atol=1e-10,
    )
    assert_almost_equal(
        betainc(2.0, 7.0, 0.3),
        1.0 - betainc(7.0, 2.0, 0.7),
        atol=1e-10,
    )
    print("✓ test_betainc_symmetry_identity passed")


fn test_betainc_known_values() raises:
    """Test I_x(1, n) = 1 - (1-x)^n for integer n."""
    var x = 0.3
    assert_almost_equal(betainc(1.0, 1.0, x), x, atol=1e-12)
    assert_almost_equal(betainc(1.0, 2.0, x), 1.0 - (1.0 - x) ** 2, atol=1e-10)
    assert_almost_equal(betainc(1.0, 5.0, x), 1.0 - (1.0 - x) ** 5, atol=1e-10)
    print("✓ test_betainc_known_values passed")


fn test_betainc_scipy() raises:
    """Test betainc against scipy.special for general parameters."""
    var sp = _load_scipy()
    if sp is None:
        print("⊘ test_betainc_scipy skipped (scipy not available)")
        return

    var test_a: List[Float64] = [0.5, 2.0, 5.0, 0.1, 10.0, 3.0]
    var test_b: List[Float64] = [0.5, 5.0, 2.0, 0.1, 3.0, 10.0]
    var test_x: List[Float64] = [0.3, 0.4, 0.6, 0.5, 0.8, 0.2]

    for i in range(len(test_a)):
        var a = test_a[i]
        var b = test_b[i]
        var x = test_x[i]

        var sp_val = _py_f64(sp.betainc(a, b, x))
        _assert_with_scipy(
            betainc(a, b, x), sp_val, sp, sp_val,
            "betainc(" + String(a) + ", " + String(b) + ", " + String(x) + ")",
            atol=1e-10,
        )

    print("✓ test_betainc_scipy passed")


# ===----------------------------------------------------------------------=== #
# Tests for inverse error function
# ===----------------------------------------------------------------------=== #


fn test_erfinv_basic() raises:
    """Test erfinv by checking erf(erfinv(p)) ≈ p (round-trip)."""
    assert_almost_equal(erfinv(0.0), 0.0, atol=1e-15)

    var test_vals: List[Float64] = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, -0.5, -0.9]

    for i in range(len(test_vals)):
        var p = test_vals[i]
        var x = erfinv(p)
        assert_almost_equal(erf(x), p, atol=1e-8)

    print("✓ test_erfinv_basic passed")


fn test_erfinv_symmetry() raises:
    """Test erfinv(-p) = -erfinv(p)."""
    var test_vals: List[Float64] = [0.1, 0.5, 0.9]

    for i in range(len(test_vals)):
        var p = test_vals[i]
        assert_almost_equal(erfinv(-p), -erfinv(p), atol=1e-12)

    print("✓ test_erfinv_symmetry passed")


fn test_erfinv_scipy() raises:
    """Test erfinv against scipy.special.erfinv."""
    var sp = _load_scipy()
    if sp is None:
        print("⊘ test_erfinv_scipy skipped (scipy not available)")
        return

    var test_vals: List[Float64] = [
        0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, -0.3, -0.9, -0.999,
    ]

    for i in range(len(test_vals)):
        var p = test_vals[i]
        var sp_val = _py_f64(sp.erfinv(p))
        _assert_with_scipy(
            erfinv(p), sp_val, sp, sp_val,
            "erfinv(" + String(p) + ")",
            atol=1e-10,
        )

    print("✓ test_erfinv_scipy passed")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    print("=== StaMojo: Testing special functions ===")
    print()

    TestSuite.discover_tests[__functions_in_module()]().run()

    print()
    print("=== All special function tests passed ===")
