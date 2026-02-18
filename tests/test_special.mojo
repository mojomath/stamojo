# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for special mathematical functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the _special subpackage.

Reference values are computed from independent formulas rather than
hard-coded constants to avoid transcription errors:
- For integer a, Q(a,x) = e^{-x} Σ_{k=0}^{a-1} x^k/k!  (Poisson CDF).
- For a = 0.5, P(0.5, x) = erf(√x).
- For a = 1, P(1, x) = 1 - e^{-x}.
"""

from math import exp, log, lgamma, erf, sqrt
from testing import assert_almost_equal, TestSuite

from stamojo._special import gammainc, gammaincc, beta, lbeta, betainc, erfinv


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _poisson_cdf(n: Int, x: Float64) -> Float64:
    """Exact Q(n, x) = e^{-x} Σ_{k=0}^{n-1} x^k/k! for positive integer n.

    This is an independent computation of the upper regularized incomplete
    gamma function that does NOT use our gammainc/gammaincc code.
    """
    var term = 1.0
    var s = 1.0
    for k in range(1, n):
        term *= x / Float64(k)
        s += term
    return exp(-x) * s


# ===----------------------------------------------------------------------=== #
# Tests for regularized incomplete gamma
# ===----------------------------------------------------------------------=== #


fn test_gammainc_boundary() raises:
    """Test boundary conditions."""
    # P(a, 0) = 0 for any a > 0
    assert_almost_equal(gammainc(1.0, 0.0), 0.0, atol=1e-15)
    assert_almost_equal(gammainc(5.0, 0.0), 0.0, atol=1e-15)

    # Q(a, 0) = 1 for any a > 0
    assert_almost_equal(gammaincc(1.0, 0.0), 1.0, atol=1e-15)
    assert_almost_equal(gammaincc(5.0, 0.0), 1.0, atol=1e-15)

    print("✓ test_gammainc_boundary passed")


fn test_gammainc_exponential() raises:
    """Test P(1, x) = 1 - e^{-x} (exponential distribution CDF)."""
    var test_x = List[Float64]()
    test_x.append(0.5)
    test_x.append(1.0)
    test_x.append(2.0)
    test_x.append(5.0)
    test_x.append(10.0)

    for i in range(len(test_x)):
        var x = test_x[i]
        var expected = 1.0 - exp(-x)
        assert_almost_equal(gammainc(1.0, x), expected, atol=1e-12)

    print("✓ test_gammainc_exponential passed")


fn test_gammainc_half() raises:
    """Test P(0.5, x) = erf(sqrt(x)) for the half-integer case."""
    var test_x = List[Float64]()
    test_x.append(0.25)
    test_x.append(0.5)
    test_x.append(1.0)
    test_x.append(2.0)
    test_x.append(4.0)

    for i in range(len(test_x)):
        var x = test_x[i]
        var expected = erf(sqrt(x))
        assert_almost_equal(gammainc(0.5, x), expected, atol=1e-10)

    print("✓ test_gammainc_half passed")


fn test_gammainc_integer_a() raises:
    """Test against the Poisson sum formula for integer a.

    P(n, x) = 1 - e^{-x} Σ_{k=0}^{n-1} x^k/k!
    """
    # (a, x) pairs where x covers both the series and CF regimes.
    var test_a = List[Int]()
    var test_x = List[Float64]()
    test_a.append(2);  test_x.append(3.0)
    test_a.append(3);  test_x.append(2.0)
    test_a.append(5);  test_x.append(3.0)
    test_a.append(5);  test_x.append(10.0)
    test_a.append(10); test_x.append(5.0)
    test_a.append(10); test_x.append(15.0)
    test_a.append(20); test_x.append(25.0)

    for i in range(len(test_a)):
        var a_int = test_a[i]
        var a = Float64(a_int)
        var x = test_x[i]

        var q_exact = _poisson_cdf(a_int, x)
        var p_exact = 1.0 - q_exact

        # Our functions should match the Poisson sum to high precision.
        assert_almost_equal(
            gammainc(a, x), p_exact, atol=1e-12
        )
        assert_almost_equal(
            gammaincc(a, x), q_exact, atol=1e-12
        )

    print("✓ test_gammainc_integer_a passed")


fn test_gammainc_complementary() raises:
    """Test that P(a,x) + Q(a,x) = 1 for various parameters."""
    var test_cases = List[Tuple[Float64, Float64]]()
    test_cases.append((0.5, 0.5))
    test_cases.append((1.0, 2.0))
    test_cases.append((2.5, 3.5))
    test_cases.append((3.0, 1.0))
    test_cases.append((5.0, 5.0))
    test_cases.append((5.0, 10.0))
    test_cases.append((10.0, 20.0))
    test_cases.append((0.1, 0.01))

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
    # B(1, 1) = 1
    assert_almost_equal(beta(1.0, 1.0), 1.0, atol=1e-12)

    # B(2, 2) = 1/6
    assert_almost_equal(beta(2.0, 2.0), 1.0 / 6.0, atol=1e-12)

    # B(0.5, 0.5) = π
    assert_almost_equal(beta(0.5, 0.5), 3.141592653589793, atol=1e-10)

    # B(3, 4) = 1/60
    assert_almost_equal(beta(3.0, 4.0), 1.0 / 60.0, atol=1e-12)

    # B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b), verify with lgamma
    var a = 3.7
    var b = 2.3
    var expected = exp(lgamma(a) + lgamma(b) - lgamma(a + b))
    assert_almost_equal(beta(a, b), expected, atol=1e-12)

    print("✓ test_beta_basic passed")


fn test_betainc_boundary() raises:
    """Test betainc boundary values."""
    # I_0(a, b) = 0 and I_1(a, b) = 1 for any a, b > 0
    assert_almost_equal(betainc(2.0, 3.0, 0.0), 0.0, atol=1e-15)
    assert_almost_equal(betainc(2.0, 3.0, 1.0), 1.0, atol=1e-15)

    # I_{0.5}(1, 1) = 0.5 (uniform distribution)
    assert_almost_equal(betainc(1.0, 1.0, 0.5), 0.5, atol=1e-12)

    print("✓ test_betainc_boundary passed")


fn test_betainc_symmetric() raises:
    """Test betainc with symmetric parameters: I_{0.5}(a, a) = 0.5."""
    var test_a = List[Float64]()
    test_a.append(1.0)
    test_a.append(2.0)
    test_a.append(5.0)
    test_a.append(10.0)

    for i in range(len(test_a)):
        var a = test_a[i]
        assert_almost_equal(betainc(a, a, 0.5), 0.5, atol=1e-10)

    print("✓ test_betainc_symmetric passed")


fn test_betainc_symmetry_identity() raises:
    """Test betainc symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)."""
    var a = 3.0
    var b = 5.0
    var x = 0.4

    var left = betainc(a, b, x)
    var right = 1.0 - betainc(b, a, 1.0 - x)
    assert_almost_equal(left, right, atol=1e-10)

    # Another pair
    assert_almost_equal(
        betainc(2.0, 7.0, 0.3),
        1.0 - betainc(7.0, 2.0, 0.7),
        atol=1e-10,
    )

    print("✓ test_betainc_symmetry_identity passed")


fn test_betainc_known_values() raises:
    """Test betainc against analytically known values.

    For a=1, b=n (integer): I_x(1, n) = 1 - (1-x)^n
    """
    var x = 0.3

    # I_x(1, 1) = x
    assert_almost_equal(betainc(1.0, 1.0, x), x, atol=1e-12)

    # I_x(1, 2) = 1 - (1-x)^2 = 2x - x^2
    assert_almost_equal(betainc(1.0, 2.0, x), 1.0 - (1.0 - x) ** 2, atol=1e-10)

    # I_x(1, 5) = 1 - (1-x)^5
    assert_almost_equal(betainc(1.0, 5.0, x), 1.0 - (1.0 - x) ** 5, atol=1e-10)

    print("✓ test_betainc_known_values passed")


# ===----------------------------------------------------------------------=== #
# Tests for inverse error function
# ===----------------------------------------------------------------------=== #


fn test_erfinv_basic() raises:
    """Test erfinv by checking erf(erfinv(p)) ≈ p (round-trip)."""
    # erfinv(0) = 0
    assert_almost_equal(erfinv(0.0), 0.0, atol=1e-15)

    # Round-trip test: erf(erfinv(p)) should equal p.
    var test_vals = List[Float64]()
    test_vals.append(0.1)
    test_vals.append(0.3)
    test_vals.append(0.5)
    test_vals.append(0.7)
    test_vals.append(0.9)
    test_vals.append(0.99)
    test_vals.append(0.999)
    test_vals.append(-0.5)
    test_vals.append(-0.9)

    for i in range(len(test_vals)):
        var p = test_vals[i]
        var x = erfinv(p)
        assert_almost_equal(erf(x), p, atol=1e-8)

    print("✓ test_erfinv_basic passed")


fn test_erfinv_symmetry() raises:
    """Test erfinv antisymmetry: erfinv(-p) = -erfinv(p)."""
    var test_vals = List[Float64]()
    test_vals.append(0.1)
    test_vals.append(0.5)
    test_vals.append(0.9)

    for i in range(len(test_vals)):
        var p = test_vals[i]
        assert_almost_equal(erfinv(-p), -erfinv(p), atol=1e-12)

    print("✓ test_erfinv_symmetry passed")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    print("=== StaMojo: Testing special functions ===")
    print()

    TestSuite.discover_tests[__functions_in_module()]().run()

    print()
    print("=== All special function tests passed ===")
