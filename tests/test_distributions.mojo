# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for probability distributions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the distributions subpackage.

Covers Normal, Student's t, Chi-squared, and F distributions.
Each distribution is tested for:
  - Known analytical values
  - CDF/PPF round-trip consistency
  - Symmetry and boundary conditions
  - Comparison against scipy.stats (when available)
"""

from math import exp, log, sqrt, erf, erfc
from python import Python, PythonObject
from testing import assert_almost_equal

from stamojo.distributions import Normal, StudentT, ChiSquared, FDist


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _load_scipy_stats() -> PythonObject:
    """Try to import scipy.stats.  Returns Python None if unavailable."""
    try:
        return Python.import_module("scipy.stats")
    except:
        return PythonObject(None)


fn _py_f64(obj: PythonObject) -> Float64:
    """Convert a PythonObject holding a numeric value to Float64."""
    try:
        return atof(String(obj))
    except:
        return 0.0


# ===----------------------------------------------------------------------=== #
# Normal distribution tests
# ===----------------------------------------------------------------------=== #


fn test_normal_pdf() raises:
    """Test Normal PDF at known values."""
    var n = Normal(0.0, 1.0)
    # PDF at 0 for standard normal ≈ 1/√(2π) ≈ 0.3989422804014327
    assert_almost_equal(n.pdf(0.0), 0.3989422804014327, atol=1e-12)
    # PDF is symmetric
    assert_almost_equal(n.pdf(1.0), n.pdf(-1.0), atol=1e-15)
    assert_almost_equal(n.pdf(2.0), n.pdf(-2.0), atol=1e-15)
    # Non-standard normal: N(5, 2), pdf at mean = 1/(σ√(2π))
    var n2 = Normal(5.0, 2.0)
    assert_almost_equal(n2.pdf(5.0), 0.19947114020071635, atol=1e-12)
    print("✓ test_normal_pdf passed")


fn test_normal_cdf() raises:
    """Test Normal CDF at known values."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.cdf(0.0), 0.5, atol=1e-15)
    # Φ(x) + Φ(−x) = 1
    assert_almost_equal(n.cdf(1.5) + n.cdf(-1.5), 1.0, atol=1e-15)
    # Tails
    assert_almost_equal(n.cdf(-10.0), 0.0, atol=1e-15)
    assert_almost_equal(n.cdf(10.0), 1.0, atol=1e-15)
    print("✓ test_normal_cdf passed")


fn test_normal_ppf() raises:
    """Test Normal PPF (inverse CDF)."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.ppf(0.5), 0.0, atol=1e-12)
    # Round-trip: ppf(cdf(x)) ≈ x
    assert_almost_equal(n.ppf(n.cdf(1.5)), 1.5, atol=1e-10)
    assert_almost_equal(n.ppf(n.cdf(-2.3)), -2.3, atol=1e-10)
    # Non-standard normal
    var n2 = Normal(10.0, 3.0)
    assert_almost_equal(n2.ppf(0.5), 10.0, atol=1e-10)
    assert_almost_equal(n2.ppf(n2.cdf(15.0)), 15.0, atol=1e-10)
    print("✓ test_normal_ppf passed")


fn test_normal_cdf_ppf_roundtrip() raises:
    """Test CDF(PPF(p)) ≈ p for many probability values."""
    var n = Normal(0.0, 1.0)
    var ps = List[Float64]()
    ps.append(0.01)
    ps.append(0.05)
    ps.append(0.1)
    ps.append(0.25)
    ps.append(0.5)
    ps.append(0.75)
    ps.append(0.9)
    ps.append(0.95)
    ps.append(0.99)

    for i in range(len(ps)):
        var p = ps[i]
        assert_almost_equal(n.cdf(n.ppf(p)), p, atol=1e-10)

    print("✓ test_normal_cdf_ppf_roundtrip passed")


fn test_normal_sf() raises:
    """Test Normal survival function."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.sf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(n.cdf(1.5) + n.sf(1.5), 1.0, atol=1e-15)
    print("✓ test_normal_sf passed")


fn test_normal_stats() raises:
    """Test Normal distribution statistics."""
    var n = Normal(3.0, 2.0)
    assert_almost_equal(n.mean(), 3.0, atol=1e-15)
    assert_almost_equal(n.variance(), 4.0, atol=1e-15)
    assert_almost_equal(n.std(), 2.0, atol=1e-15)
    print("✓ test_normal_stats passed")


fn test_normal_scipy() raises:
    """Test Normal distribution against scipy.stats.norm."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_normal_scipy skipped (scipy not available)")
        return

    var n = Normal(0.0, 1.0)
    var xs = List[Float64]()
    xs.append(-3.0)
    xs.append(-1.0)
    xs.append(0.0)
    xs.append(1.0)
    xs.append(3.0)

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.norm.pdf(x))
        var sp_cdf = _py_f64(sp.norm.cdf(x))
        assert_almost_equal(n.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(n.cdf(x), sp_cdf, atol=1e-12)

    print("✓ test_normal_scipy passed")


# ===----------------------------------------------------------------------=== #
# Student's t distribution tests
# ===----------------------------------------------------------------------=== #


fn test_t_pdf_symmetry() raises:
    """Test Student's t PDF is symmetric about 0."""
    var t = StudentT(5.0)
    assert_almost_equal(t.pdf(1.0), t.pdf(-1.0), atol=1e-15)
    assert_almost_equal(t.pdf(2.5), t.pdf(-2.5), atol=1e-15)
    print("✓ test_t_pdf_symmetry passed")


fn test_t_cdf() raises:
    """Test Student's t CDF at known values."""
    # Cauchy distribution (df=1): CDF(0)=0.5, CDF(1)=0.75
    var t1 = StudentT(1.0)
    assert_almost_equal(t1.cdf(0.0), 0.5, atol=1e-12)
    assert_almost_equal(t1.cdf(1.0), 0.75, atol=1e-6)

    # Symmetry: CDF(x) + CDF(−x) = 1
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.cdf(0.0), 0.5, atol=1e-12)
    assert_almost_equal(t5.cdf(2.0) + t5.cdf(-2.0), 1.0, atol=1e-10)
    print("✓ test_t_cdf passed")


fn test_t_ppf() raises:
    """Test Student's t PPF."""
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.ppf(0.5), 0.0, atol=1e-10)
    # Round-trip
    assert_almost_equal(t5.cdf(t5.ppf(0.975)), 0.975, atol=1e-6)
    assert_almost_equal(t5.cdf(t5.ppf(0.025)), 0.025, atol=1e-6)
    assert_almost_equal(t5.cdf(t5.ppf(0.9)), 0.9, atol=1e-6)
    print("✓ test_t_ppf passed")


fn test_t_stats() raises:
    """Test Student's t distribution statistics."""
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.mean(), 0.0, atol=1e-15)
    assert_almost_equal(t5.variance(), 5.0 / 3.0, atol=1e-12)
    print("✓ test_t_stats passed")


fn test_t_scipy() raises:
    """Test Student's t distribution against scipy.stats.t."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_t_scipy skipped (scipy not available)")
        return

    var dfs = List[Float64]()
    dfs.append(1.0)
    dfs.append(3.0)
    dfs.append(5.0)
    dfs.append(10.0)
    dfs.append(30.0)

    for i in range(len(dfs)):
        var df = dfs[i]
        var t = StudentT(df)
        var xs = List[Float64]()
        xs.append(-2.0)
        xs.append(0.0)
        xs.append(1.5)

        for j in range(len(xs)):
            var x = xs[j]
            var sp_pdf = _py_f64(sp.t.pdf(x, df))
            var sp_cdf = _py_f64(sp.t.cdf(x, df))
            assert_almost_equal(t.pdf(x), sp_pdf, atol=1e-10)
            assert_almost_equal(t.cdf(x), sp_cdf, atol=1e-6)

    print("✓ test_t_scipy passed")


# ===----------------------------------------------------------------------=== #
# Chi-squared distribution tests
# ===----------------------------------------------------------------------=== #


fn test_chi2_cdf() raises:
    """Test Chi-squared CDF at known values.

    For df=2: CDF(x) = 1 − exp(−x/2).
    """
    var c2 = ChiSquared(2.0)
    assert_almost_equal(c2.cdf(2.0), 1.0 - exp(-1.0), atol=1e-10)
    assert_almost_equal(c2.cdf(4.0), 1.0 - exp(-2.0), atol=1e-10)
    assert_almost_equal(c2.cdf(0.0), 0.0, atol=1e-15)
    print("✓ test_chi2_cdf passed")


fn test_chi2_ppf() raises:
    """Test Chi-squared PPF (round-trip)."""
    var c5 = ChiSquared(5.0)
    assert_almost_equal(c5.cdf(c5.ppf(0.95)), 0.95, atol=1e-6)
    assert_almost_equal(c5.cdf(c5.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(c5.cdf(c5.ppf(0.01)), 0.01, atol=1e-6)
    print("✓ test_chi2_ppf passed")


fn test_chi2_stats() raises:
    """Test Chi-squared distribution statistics."""
    var c5 = ChiSquared(5.0)
    assert_almost_equal(c5.mean(), 5.0, atol=1e-15)
    assert_almost_equal(c5.variance(), 10.0, atol=1e-15)
    print("✓ test_chi2_stats passed")


fn test_chi2_scipy() raises:
    """Test Chi-squared distribution against scipy.stats.chi2."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_chi2_scipy skipped (scipy not available)")
        return

    var dfs = List[Float64]()
    dfs.append(1.0)
    dfs.append(3.0)
    dfs.append(5.0)
    dfs.append(10.0)

    for i in range(len(dfs)):
        var df = dfs[i]
        var c = ChiSquared(df)
        var xs = List[Float64]()
        xs.append(1.0)
        xs.append(3.0)
        xs.append(7.0)

        for j in range(len(xs)):
            var x = xs[j]
            var sp_cdf = _py_f64(sp.chi2.cdf(x, df))
            assert_almost_equal(c.cdf(x), sp_cdf, atol=1e-6)

    print("✓ test_chi2_scipy passed")


# ===----------------------------------------------------------------------=== #
# F-distribution tests
# ===----------------------------------------------------------------------=== #


fn test_f_cdf_boundary() raises:
    """Test F-distribution CDF boundary and monotonicity."""
    var f = FDist(5.0, 10.0)
    assert_almost_equal(f.cdf(0.0), 0.0, atol=1e-15)
    # Monotonically increasing
    var c1 = f.cdf(1.0)
    var c2 = f.cdf(2.0)
    var c3 = f.cdf(5.0)
    if not (c1 < c2 and c2 < c3):
        raise Error("F CDF not monotonically increasing")
    print("✓ test_f_cdf_boundary passed")


fn test_f_ppf() raises:
    """Test F-distribution PPF (round-trip)."""
    var f = FDist(5.0, 10.0)
    assert_almost_equal(f.cdf(f.ppf(0.95)), 0.95, atol=1e-6)
    assert_almost_equal(f.cdf(f.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(f.cdf(f.ppf(0.1)), 0.1, atol=1e-6)
    print("✓ test_f_ppf passed")


fn test_f_stats() raises:
    """Test F-distribution statistics."""
    var f = FDist(5.0, 10.0)
    # mean = d2 / (d2 - 2) = 10/8 = 1.25
    assert_almost_equal(f.mean(), 1.25, atol=1e-12)
    print("✓ test_f_stats passed")


fn test_f_scipy() raises:
    """Test F-distribution against scipy.stats.f."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_f_scipy skipped (scipy not available)")
        return

    var f = FDist(5.0, 10.0)
    var xs = List[Float64]()
    xs.append(0.5)
    xs.append(1.0)
    xs.append(2.0)
    xs.append(5.0)

    for i in range(len(xs)):
        var x = xs[i]
        var sp_cdf = _py_f64(sp.f.cdf(x, 5.0, 10.0))
        assert_almost_equal(f.cdf(x), sp_cdf, atol=1e-6)

    print("✓ test_f_scipy passed")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    print("=== StaMojo: Testing distributions ===")
    print()

    # Normal
    test_normal_pdf()
    test_normal_cdf()
    test_normal_ppf()
    test_normal_cdf_ppf_roundtrip()
    test_normal_sf()
    test_normal_stats()
    test_normal_scipy()
    print()

    # Student's t
    test_t_pdf_symmetry()
    test_t_cdf()
    test_t_ppf()
    test_t_stats()
    test_t_scipy()
    print()

    # Chi-squared
    test_chi2_cdf()
    test_chi2_ppf()
    test_chi2_stats()
    test_chi2_scipy()
    print()

    # F-distribution
    test_f_cdf_boundary()
    test_f_ppf()
    test_f_stats()
    test_f_scipy()

    print()
    print("=== All distribution tests passed ===")
