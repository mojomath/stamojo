# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for probability distributions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the distributions subpackage.

Covers Normal, Student's t, Chi-squared, F, Exponential, and Binomial distributions.
Each distribution is tested for:
  - Known analytical values
  - CDF/PPF round-trip consistency
  - Symmetry and boundary conditions
  - Comparison against scipy.stats (when available)
"""

from std.math import exp, log, sqrt
from std.python import Python, PythonObject
from std.testing import assert_almost_equal, assert_true, TestSuite

from stamojo.distributions import (
    Normal,
    StudentT,
    ChiSquared,
    FDist,
    Exponential,
    Binomial,
    Gamma,
    Beta,
    Poisson,
    Uniform,
    LogNormal,
    Weibull,
    Cauchy,
    Laplace,
    Logistic,
    Pareto,
    Rayleigh,
    HalfNormal,
    HalfCauchy,
    Gumbel,
    Triangular,
    VonMises,
    TruncNorm,
)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _load_scipy_stats() -> PythonObject:
    """Try to import scipy.stats.  Returns Python None if unavailable."""
    try:
        return Python.import_module("scipy.stats")
    except:
        return PythonObject(None)


def _py_f64(obj: PythonObject) -> Float64:
    """Converts a PythonObject holding a numeric value to Float64."""
    try:
        return atof(String(obj))
    except:
        return 0.0


# ===----------------------------------------------------------------------=== #
# Normal distribution tests
# ===----------------------------------------------------------------------=== #


def test_normal_pdf() raises:
    """Tests Normal PDF at known values."""
    var n = Normal(0.0, 1.0)
    # PDF at 0 for standard normal ≈ 1/√(2π) ≈ 0.3989422804014327
    assert_almost_equal(n.pdf(0.0), 0.3989422804014327, atol=1e-12)
    # PDF is symmetric
    assert_almost_equal(n.pdf(1.0), n.pdf(-1.0), atol=1e-15)
    assert_almost_equal(n.pdf(2.0), n.pdf(-2.0), atol=1e-15)
    # Non-standard normal: N(5, 2), pdf at mean = 1/(σ√(2π))
    var n2 = Normal(5.0, 2.0)
    assert_almost_equal(n2.pdf(5.0), 0.19947114020071635, atol=1e-12)


def test_normal_cdf() raises:
    """Tests Normal CDF at known values."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.cdf(0.0), 0.5, atol=1e-15)
    # Φ(x) + Φ(−x) = 1
    assert_almost_equal(n.cdf(1.5) + n.cdf(-1.5), 1.0, atol=1e-15)
    # Tails
    assert_almost_equal(n.cdf(-10.0), 0.0, atol=1e-15)
    assert_almost_equal(n.cdf(10.0), 1.0, atol=1e-15)


def test_normal_ppf() raises:
    """Tests Normal PPF (inverse CDF)."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.ppf(0.5), 0.0, atol=1e-12)
    # Round-trip: ppf(cdf(x)) ≈ x
    assert_almost_equal(n.ppf(n.cdf(1.5)), 1.5, atol=1e-10)
    assert_almost_equal(n.ppf(n.cdf(-2.3)), -2.3, atol=1e-10)
    # Non-standard normal
    var n2 = Normal(10.0, 3.0)
    assert_almost_equal(n2.ppf(0.5), 10.0, atol=1e-10)
    assert_almost_equal(n2.ppf(n2.cdf(15.0)), 15.0, atol=1e-10)


def test_normal_cdf_ppf_roundtrip() raises:
    """Tests CDF(PPF(p)) ≈ p for many probability values."""
    var n = Normal(0.0, 1.0)
    var ps: List[Float64] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    for i in range(len(ps)):
        var p = ps[i]
        assert_almost_equal(n.cdf(n.ppf(p)), p, atol=1e-10)


def test_normal_sf() raises:
    """Tests Normal survival function."""
    var n = Normal(0.0, 1.0)
    assert_almost_equal(n.sf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(n.cdf(1.5) + n.sf(1.5), 1.0, atol=1e-15)


def test_normal_stats() raises:
    """Tests Normal distribution statistics."""
    var n = Normal(3.0, 2.0)
    assert_almost_equal(n.mean(), 3.0, atol=1e-15)
    assert_almost_equal(n.variance(), 4.0, atol=1e-15)
    assert_almost_equal(n.std(), 2.0, atol=1e-15)


def test_normal_scipy() raises:
    """Tests Normal distribution against scipy.stats.norm."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_normal_scipy skipped (scipy not available)")
        return

    var n = Normal(0.0, 1.0)
    var xs: List[Float64] = [-3.0, -1.0, 0.0, 1.0, 3.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.norm.pdf(x))
        var sp_cdf = _py_f64(sp.norm.cdf(x))
        assert_almost_equal(n.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(n.cdf(x), sp_cdf, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# Student's t distribution tests
# ===----------------------------------------------------------------------=== #


def test_t_pdf_symmetry() raises:
    """Tests Student's t PDF is symmetric about 0."""
    var t = StudentT(5.0)
    assert_almost_equal(t.pdf(1.0), t.pdf(-1.0), atol=1e-15)
    assert_almost_equal(t.pdf(2.5), t.pdf(-2.5), atol=1e-15)


def test_t_cdf() raises:
    """Tests Student's t CDF at known values."""
    # Cauchy distribution (df=1): CDF(0)=0.5, CDF(1)=0.75
    var t1 = StudentT(1.0)
    assert_almost_equal(t1.cdf(0.0), 0.5, atol=1e-12)
    assert_almost_equal(t1.cdf(1.0), 0.75, atol=1e-6)

    # Symmetry: CDF(x) + CDF(−x) = 1
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.cdf(0.0), 0.5, atol=1e-12)
    assert_almost_equal(t5.cdf(2.0) + t5.cdf(-2.0), 1.0, atol=1e-10)


def test_t_ppf() raises:
    """Tests Student's t PPF."""
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.ppf(0.5), 0.0, atol=1e-10)
    # Round-trip
    assert_almost_equal(t5.cdf(t5.ppf(0.975)), 0.975, atol=1e-6)
    assert_almost_equal(t5.cdf(t5.ppf(0.025)), 0.025, atol=1e-6)
    assert_almost_equal(t5.cdf(t5.ppf(0.9)), 0.9, atol=1e-6)


def test_t_stats() raises:
    """Tests Student's t distribution statistics."""
    var t5 = StudentT(5.0)
    assert_almost_equal(t5.mean(), 0.0, atol=1e-15)
    assert_almost_equal(t5.variance(), 5.0 / 3.0, atol=1e-12)


def test_t_scipy() raises:
    """Tests Student's t distribution against scipy.stats.t."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_t_scipy skipped (scipy not available)")
        return

    var dfs: List[Float64] = [1.0, 3.0, 5.0, 10.0, 30.0]

    for i in range(len(dfs)):
        var df = dfs[i]
        var t = StudentT(df)
        var xs: List[Float64] = [-2.0, 0.0, 1.5]

        for j in range(len(xs)):
            var x = xs[j]
            var sp_pdf = _py_f64(sp.t.pdf(x, df))
            var sp_cdf = _py_f64(sp.t.cdf(x, df))
            assert_almost_equal(t.pdf(x), sp_pdf, atol=1e-10)
            assert_almost_equal(t.cdf(x), sp_cdf, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Chi-squared distribution tests
# ===----------------------------------------------------------------------=== #


def test_chi2_cdf() raises:
    """Tests Chi-squared CDF at known values.

    For df=2: CDF(x) = 1 − exp(−x/2).
    """
    var c2 = ChiSquared(2.0)
    assert_almost_equal(c2.cdf(2.0), 1.0 - exp(-1.0), atol=1e-10)
    assert_almost_equal(c2.cdf(4.0), 1.0 - exp(-2.0), atol=1e-10)
    assert_almost_equal(c2.cdf(0.0), 0.0, atol=1e-15)


def test_chi2_ppf() raises:
    """Tests Chi-squared PPF (round-trip)."""
    var c5 = ChiSquared(5.0)
    assert_almost_equal(c5.cdf(c5.ppf(0.95)), 0.95, atol=1e-6)
    assert_almost_equal(c5.cdf(c5.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(c5.cdf(c5.ppf(0.01)), 0.01, atol=1e-6)


def test_chi2_stats() raises:
    """Tests Chi-squared distribution statistics."""
    var c5 = ChiSquared(5.0)
    assert_almost_equal(c5.mean(), 5.0, atol=1e-15)
    assert_almost_equal(c5.variance(), 10.0, atol=1e-15)


def test_chi2_scipy() raises:
    """Tests Chi-squared distribution against scipy.stats.chi2."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_chi2_scipy skipped (scipy not available)")
        return

    var dfs: List[Float64] = [1.0, 3.0, 5.0, 10.0]

    for i in range(len(dfs)):
        var df = dfs[i]
        var c = ChiSquared(df)
        var xs: List[Float64] = [1.0, 3.0, 7.0]

        for j in range(len(xs)):
            var x = xs[j]
            var sp_cdf = _py_f64(sp.chi2.cdf(x, df))
            assert_almost_equal(c.cdf(x), sp_cdf, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# F-distribution tests
# ===----------------------------------------------------------------------=== #


def test_f_cdf_boundary() raises:
    """Tests F-distribution CDF boundary and monotonicity."""
    var f = FDist(5.0, 10.0)
    assert_almost_equal(f.cdf(0.0), 0.0, atol=1e-15)
    # Monotonically increasing
    var c1 = f.cdf(1.0)
    var c2 = f.cdf(2.0)
    var c3 = f.cdf(5.0)
    if not (c1 < c2 and c2 < c3):
        raise Error("F CDF not monotonically increasing")


def test_f_ppf() raises:
    """Tests F-distribution PPF (round-trip)."""
    var f = FDist(5.0, 10.0)
    assert_almost_equal(f.cdf(f.ppf(0.95)), 0.95, atol=1e-6)
    assert_almost_equal(f.cdf(f.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(f.cdf(f.ppf(0.1)), 0.1, atol=1e-6)


def test_f_stats() raises:
    """Tests F-distribution statistics."""
    var f = FDist(5.0, 10.0)
    # mean = d2 / (d2 - 2) = 10/8 = 1.25
    assert_almost_equal(f.mean(), 1.25, atol=1e-12)


def test_f_scipy() raises:
    """Tests F-distribution against scipy.stats.f."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_f_scipy skipped (scipy not available)")
        return

    var f = FDist(5.0, 10.0)
    var xs: List[Float64] = [0.5, 1.0, 2.0, 5.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_cdf = _py_f64(sp.f.cdf(x, 5.0, 10.0))
        assert_almost_equal(f.cdf(x), sp_cdf, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Exponential distribution tests
# ===----------------------------------------------------------------------=== #


def test_expon_pdf() raises:
    """Tests Exponential PDF at known values."""
    var e = Exponential()
    # Standard exponential: pdf(0) = 1.0
    assert_almost_equal(e.pdf(0.0), 1.0, atol=1e-15)
    # pdf(1) = exp(-1)
    assert_almost_equal(e.pdf(1.0), exp(-1.0), atol=1e-15)
    # pdf(2) = exp(-2)
    assert_almost_equal(e.pdf(2.0), exp(-2.0), atol=1e-15)
    # pdf(x < 0) = 0
    assert_almost_equal(e.pdf(-1.0), 0.0, atol=1e-15)
    # With scale=2: pdf(x) = (1/2)*exp(-x/2)
    var e2 = Exponential(scale=2.0)
    assert_almost_equal(e2.pdf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(e2.pdf(2.0), 0.5 * exp(-1.0), atol=1e-15)
    # With loc=1: pdf(1) = 1.0, pdf(0) = 0.0
    var e3 = Exponential(loc=1.0)
    assert_almost_equal(e3.pdf(1.0), 1.0, atol=1e-15)
    assert_almost_equal(e3.pdf(0.5), 0.0, atol=1e-15)


def test_expon_logpdf() raises:
    """Tests Exponential log-PDF at known values."""
    var e = Exponential()
    # logpdf(0) = 0.0 for standard exponential
    assert_almost_equal(e.logpdf(0.0), 0.0, atol=1e-15)
    # logpdf(1) = -1.0
    assert_almost_equal(e.logpdf(1.0), -1.0, atol=1e-15)
    # logpdf(x) = log(pdf(x))
    assert_almost_equal(e.logpdf(2.0), log(e.pdf(2.0)), atol=1e-15)
    # With scale=3: logpdf(x) = -x/3 - log(3)
    var e2 = Exponential(scale=3.0)
    assert_almost_equal(e2.logpdf(3.0), -1.0 - log(3.0), atol=1e-15)


def test_expon_cdf() raises:
    """Tests Exponential CDF at known values."""
    var e = Exponential()
    # CDF(0) = 0
    assert_almost_equal(e.cdf(0.0), 0.0, atol=1e-15)
    # CDF(1) = 1 - exp(-1)
    assert_almost_equal(e.cdf(1.0), 1.0 - exp(-1.0), atol=1e-15)
    # CDF(x < 0) = 0
    assert_almost_equal(e.cdf(-1.0), 0.0, atol=1e-15)
    # CDF should be monotonically increasing
    var c1 = e.cdf(0.5)
    var c2 = e.cdf(1.0)
    var c3 = e.cdf(5.0)
    if not (c1 < c2 and c2 < c3):
        raise Error("Exponential CDF not monotonically increasing")
    # With scale=0.5 (rate=2): CDF(x) = 1 - exp(-2x)
    var e2 = Exponential(scale=0.5)
    assert_almost_equal(e2.cdf(1.0), 1.0 - exp(-2.0), atol=1e-15)


def test_expon_sf() raises:
    """Tests Exponential survival function: SF(x) = 1 - CDF(x)."""
    var e = Exponential()
    assert_almost_equal(e.sf(0.0), 1.0, atol=1e-15)
    assert_almost_equal(e.sf(1.0), exp(-1.0), atol=1e-15)
    # CDF + SF = 1
    var xs: List[Float64] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for i in range(len(xs)):
        assert_almost_equal(e.cdf(xs[i]) + e.sf(xs[i]), 1.0, atol=1e-15)
    # SF(x < loc) = 1
    assert_almost_equal(e.sf(-1.0), 1.0, atol=1e-15)


def test_expon_ppf() raises:
    """Tests Exponential PPF (inverse CDF)."""
    var e = Exponential()
    # PPF(0) = 0 (loc)
    assert_almost_equal(e.ppf(0.0), 0.0, atol=1e-15)
    # PPF(1 - exp(-1)) = 1 (since CDF(1) = 1 - exp(-1))
    assert_almost_equal(e.ppf(1.0 - exp(-1.0)), 1.0, atol=1e-12)
    # PPF(0.5) = ln(2) (median of standard exponential)
    assert_almost_equal(e.ppf(0.5), log(2.0), atol=1e-12)
    # With loc and scale
    var e2 = Exponential(loc=1.0, scale=2.0)
    assert_almost_equal(e2.ppf(0.5), 1.0 + 2.0 * log(2.0), atol=1e-12)


def test_expon_cdf_ppf_roundtrip() raises:
    """Tests CDF(PPF(p)) ≈ p for many probability values."""
    var e = Exponential()
    var ps: List[Float64] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for i in range(len(ps)):
        var p = ps[i]
        assert_almost_equal(e.cdf(e.ppf(p)), p, atol=1e-12)
    # Also test with loc and scale
    var e2 = Exponential(loc=2.0, scale=3.0)
    for i in range(len(ps)):
        var p = ps[i]
        assert_almost_equal(e2.cdf(e2.ppf(p)), p, atol=1e-12)


def test_expon_isf() raises:
    """Tests Exponential ISF (inverse survival function)."""
    var e = Exponential()
    # ISF(1) = loc = 0
    assert_almost_equal(e.isf(1.0), 0.0, atol=1e-15)
    # ISF(exp(-1)) = 1 (since SF(1) = exp(-1))
    assert_almost_equal(e.isf(exp(-1.0)), 1.0, atol=1e-12)
    # ISF(q) = PPF(1 - q)
    var qs: List[Float64] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for i in range(len(qs)):
        var q = qs[i]
        assert_almost_equal(e.isf(q), e.ppf(1.0 - q), atol=1e-12)


def test_expon_logcdf_logsf() raises:
    """Tests log-CDF and log-SF against log of CDF and SF."""
    var e = Exponential()
    var xs: List[Float64] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    for i in range(len(xs)):
        var x = xs[i]
        assert_almost_equal(e.logcdf(x), log(e.cdf(x)), atol=1e-12)
        assert_almost_equal(e.logsf(x), log(e.sf(x)), atol=1e-15)


def test_expon_stats() raises:
    """Tests Exponential distribution summary statistics."""
    var e = Exponential()
    # Standard exponential: mean=1, var=1, std=1, median=ln(2)
    assert_almost_equal(e.mean(), 1.0, atol=1e-15)
    assert_almost_equal(e.variance(), 1.0, atol=1e-15)
    assert_almost_equal(e.std(), 1.0, atol=1e-15)
    assert_almost_equal(e.median(), log(2.0), atol=1e-15)
    # With loc=2, scale=3: mean=5, var=9, std=3, median=2+3*ln(2)
    var e2 = Exponential(loc=2.0, scale=3.0)
    assert_almost_equal(e2.mean(), 5.0, atol=1e-15)
    assert_almost_equal(e2.variance(), 9.0, atol=1e-15)
    assert_almost_equal(e2.std(), 3.0, atol=1e-15)
    assert_almost_equal(e2.median(), 2.0 + 3.0 * log(2.0), atol=1e-15)


def test_expon_loc_scale() raises:
    """Tests Exponential with non-default loc and scale across all functions."""
    var loc = 5.0
    var scale = 2.0
    var e = Exponential(loc, scale)
    # PDF at loc should be 1/scale
    assert_almost_equal(e.pdf(loc), 1.0 / scale, atol=1e-15)
    # CDF at loc should be 0
    assert_almost_equal(e.cdf(loc), 0.0, atol=1e-15)
    # SF at loc should be 1
    assert_almost_equal(e.sf(loc), 1.0, atol=1e-15)
    # CDF(loc + scale) = 1 - exp(-1)
    assert_almost_equal(e.cdf(loc + scale), 1.0 - exp(-1.0), atol=1e-15)


def test_expon_scipy() raises:
    """Tests Exponential distribution against scipy.stats.expon."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_expon_scipy skipped (scipy not available)")
        return

    var e = Exponential()
    var xs: List[Float64] = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.expon.pdf(x))
        var sp_cdf = _py_f64(sp.expon.cdf(x))
        var sp_sf = _py_f64(sp.expon.sf(x))
        assert_almost_equal(e.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(e.cdf(x), sp_cdf, atol=1e-12)
        assert_almost_equal(e.sf(x), sp_sf, atol=1e-12)

    # Test PPF
    var ps: List[Float64] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for i in range(len(ps)):
        var p = ps[i]
        var sp_ppf = _py_f64(sp.expon.ppf(p))
        assert_almost_equal(e.ppf(p), sp_ppf, atol=1e-12)

    # Test with loc and scale
    var loc = 2.0
    var scale = 3.0
    var e2 = Exponential(loc, scale)
    for i in range(len(xs)):
        var x = xs[i] + loc
        var sp_pdf2 = _py_f64(sp.expon.pdf(x, loc, scale))
        var sp_cdf2 = _py_f64(sp.expon.cdf(x, loc, scale))
        assert_almost_equal(e2.pdf(x), sp_pdf2, atol=1e-12)
        assert_almost_equal(e2.cdf(x), sp_cdf2, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# Binomial distribution tests
# ===----------------------------------------------------------------------=== #


def test_binomial_pmf_basic() raises:
    """Tests Binomial PMF at known values."""
    var b = Binomial(10, 0.5)
    assert_almost_equal(b.pmf(0), 1.0 / 1024.0, atol=1e-12)
    assert_almost_equal(b.pmf(5), 252.0 / 1024.0, atol=1e-12)
    assert_almost_equal(b.pmf(10), 1.0 / 1024.0, atol=1e-12)
    assert_almost_equal(b.pmf(11), 0.0, atol=1e-12)


def test_binomial_cdf_sf() raises:
    """Tests Binomial CDF and SF at known values."""
    var b = Binomial(4, 0.5)
    assert_almost_equal(b.cdf(2), 0.6875, atol=1e-12)
    assert_almost_equal(b.sf(2), 0.3125, atol=1e-12)
    assert_almost_equal(b.cdf(4), 1.0, atol=1e-12)


def test_binomial_edge_p() raises:
    """Tests Binomial behavior for p=0 and p=1."""
    var b0 = Binomial(5, 0.0)
    assert_almost_equal(b0.pmf(0), 1.0, atol=1e-12)
    assert_almost_equal(b0.pmf(1), 0.0, atol=1e-12)
    assert_almost_equal(b0.cdf(0), 1.0, atol=1e-12)
    assert_almost_equal(b0.sf(0), 0.0, atol=1e-12)

    var b1 = Binomial(5, 1.0)
    assert_almost_equal(b1.pmf(5), 1.0, atol=1e-12)
    assert_almost_equal(b1.pmf(4), 0.0, atol=1e-12)
    assert_almost_equal(b1.cdf(4), 0.0, atol=1e-12)
    assert_almost_equal(b1.cdf(5), 1.0, atol=1e-12)


def test_binomial_logpmf() raises:
    """Tests Binomial log-PMF consistency."""
    var b = Binomial(6, 0.3)
    var k = 2
    assert_almost_equal(b.logpmf(k), log(b.pmf(k)), atol=1e-12)


def test_binomial_symmetry_p_half() raises:
    """Tests Binomial symmetry for p=0.5."""
    var b = Binomial(10, 0.5)
    for k in range(0, 11):
        assert_almost_equal(b.pmf(k), b.pmf(10 - k), atol=1e-12)


def test_binomial_ppf_isf_roundtrip() raises:
    """Tests Binomial PPF/ISF consistency with CDF/SF."""
    var b = Binomial(12, 0.4)
    var qs: List[Float64] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    for i in range(len(qs)):
        var q = qs[i]
        var k = b.ppf(q)
        if b.cdf(k) < q:
            raise Error("binomial ppf: cdf(k) < q")
        if k > 0 and b.cdf(k - 1) >= q:
            raise Error("binomial ppf: cdf(k-1) >= q")

        var k2 = b.isf(q)
        if b.sf(k2) > q:
            raise Error("binomial isf: sf(k) > q")
        if k2 > 0 and b.sf(k2 - 1) <= q:
            raise Error("binomial isf: sf(k-1) <= q")


def test_binomial_scipy() raises:
    """Tests Binomial distribution against scipy.stats.binom."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_binomial_scipy skipped (scipy not available)")
        return

    var n: Int = 10
    var p = 0.3
    var b = Binomial(n, p)
    var ks: List[Int] = [0, 1, 2, 5, 10]

    for i in range(len(ks)):
        var k = ks[i]
        var sp_pmf = _py_f64(sp.binom.pmf(k, n, p))
        var sp_cdf = _py_f64(sp.binom.cdf(k, n, p))
        var sp_sf = _py_f64(sp.binom.sf(k, n, p))
        assert_almost_equal(b.pmf(k), sp_pmf, atol=1e-10)
        assert_almost_equal(b.cdf(k), sp_cdf, atol=1e-10)
        assert_almost_equal(b.sf(k), sp_sf, atol=1e-10)

    var qs: List[Float64] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for i in range(len(qs)):
        var q = qs[i]
        var sp_ppf = _py_f64(sp.binom.ppf(q, n, p))
        assert_almost_equal(Float64(b.ppf(q)), sp_ppf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Gamma distribution tests
# ===----------------------------------------------------------------------=== #


def test_gamma_pdf() raises:
    """Tests Gamma PDF at known values."""
    var g = Gamma(2.0, 1.0)
    # PDF at x=1 for Gamma(2,1): 1*exp(-1) = exp(-1)
    assert_almost_equal(g.pdf(1.0), exp(-1.0), atol=1e-12)
    # PDF(x < 0) = 0
    assert_almost_equal(g.pdf(-1.0), 0.0, atol=1e-15)


def test_gamma_cdf() raises:
    """Tests Gamma CDF at known values."""
    var g = Gamma(1.0, 1.0)
    # Gamma(1,1) = Exponential(1): CDF(x) = 1 - exp(-x)
    assert_almost_equal(g.cdf(1.0), 1.0 - exp(-1.0), atol=1e-12)
    assert_almost_equal(g.cdf(2.0), 1.0 - exp(-2.0), atol=1e-12)


def test_gamma_ppf_roundtrip() raises:
    """Tests Gamma PPF round-trip."""
    var g = Gamma(3.0, 2.0)
    assert_almost_equal(g.cdf(g.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(g.cdf(g.ppf(0.95)), 0.95, atol=1e-6)
    assert_almost_equal(g.cdf(g.ppf(0.1)), 0.1, atol=1e-6)


def test_gamma_stats() raises:
    """Tests Gamma distribution statistics."""
    var g = Gamma(4.0, 3.0)
    # mean = a*scale = 12
    assert_almost_equal(g.mean(), 12.0, atol=1e-12)
    # variance = a*scale^2 = 36
    assert_almost_equal(g.variance(), 36.0, atol=1e-12)
    # std = 6
    assert_almost_equal(g.std(), 6.0, atol=1e-12)


def test_gamma_scipy() raises:
    """Tests Gamma distribution against scipy.stats.gamma."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_gamma_scipy skipped (scipy not available)")
        return

    var g = Gamma(2.5, 1.5)
    var xs: List[Float64] = [1.0, 2.0, 5.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.gamma.pdf(x, 2.5, scale=1.5))
        var sp_cdf = _py_f64(sp.gamma.cdf(x, 2.5, scale=1.5))
        assert_almost_equal(g.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(g.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Beta distribution tests
# ===----------------------------------------------------------------------=== #


def test_beta_pdf() raises:
    """Tests Beta PDF at known values."""
    var b = Beta(2.0, 2.0)
    # PDF at 0.5 for Beta(2,2): 6 * 0.25 = 1.5
    assert_almost_equal(b.pdf(0.5), 1.5, atol=1e-12)
    # PDF outside [0,1] = 0
    assert_almost_equal(b.pdf(-0.1), 0.0, atol=1e-15)
    assert_almost_equal(b.pdf(1.1), 0.0, atol=1e-15)


def test_beta_cdf() raises:
    """Tests Beta CDF at known values."""
    var b = Beta(1.0, 1.0)
    # Beta(1,1) = Uniform(0,1): CDF(x) = x
    assert_almost_equal(b.cdf(0.3), 0.3, atol=1e-12)
    assert_almost_equal(b.cdf(0.7), 0.7, atol=1e-12)


def test_beta_ppf_roundtrip() raises:
    """Tests Beta PPF round-trip."""
    var b = Beta(3.0, 5.0)
    assert_almost_equal(b.cdf(b.ppf(0.5)), 0.5, atol=1e-6)
    assert_almost_equal(b.cdf(b.ppf(0.9)), 0.9, atol=1e-6)
    assert_almost_equal(b.cdf(b.ppf(0.1)), 0.1, atol=1e-6)


def test_beta_stats() raises:
    """Tests Beta distribution statistics."""
    var b = Beta(2.0, 3.0)
    # mean = a/(a+b) = 2/5 = 0.4
    assert_almost_equal(b.mean(), 0.4, atol=1e-12)
    # variance = ab/((a+b)^2*(a+b+1)) = 6/(25*6) = 0.04
    assert_almost_equal(b.variance(), 0.04, atol=1e-12)


def test_beta_scipy() raises:
    """Tests Beta distribution against scipy.stats.beta."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_beta_scipy skipped (scipy not available)")
        return

    var b = Beta(2.5, 3.5)
    var xs: List[Float64] = [0.2, 0.5, 0.8]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.beta.pdf(x, 2.5, 3.5))
        var sp_cdf = _py_f64(sp.beta.cdf(x, 2.5, 3.5))
        assert_almost_equal(b.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(b.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Poisson distribution tests
# ===----------------------------------------------------------------------=== #


def test_poisson_pmf() raises:
    """Tests Poisson PMF at known values."""
    var p = Poisson(3.0)
    # PMF at k=0: exp(-3)
    assert_almost_equal(p.pmf(0), exp(-3.0), atol=1e-12)
    # PMF at k=3: 3^3 * exp(-3) / 6 = 27*exp(-3)/6
    assert_almost_equal(p.pmf(3), 27.0 * exp(-3.0) / 6.0, atol=1e-12)
    # PMF for k < 0 = 0
    assert_almost_equal(p.pmf(-1), 0.0, atol=1e-15)


def test_poisson_cdf() raises:
    """Tests Poisson CDF at known values."""
    var p = Poisson(1.0)
    # CDF at k=0: exp(-1)
    assert_almost_equal(p.cdf(0), exp(-1.0), atol=1e-12)
    # CDF at k=1: exp(-1) + exp(-1) = 2*exp(-1)
    assert_almost_equal(p.cdf(1), 2.0 * exp(-1.0), atol=1e-12)


def test_poisson_stats() raises:
    """Tests Poisson distribution statistics."""
    var p = Poisson(5.0)
    assert_almost_equal(p.mean(), 5.0, atol=1e-15)
    assert_almost_equal(p.variance(), 5.0, atol=1e-15)
    assert_almost_equal(p.std(), sqrt(5.0), atol=1e-12)


def test_poisson_scipy() raises:
    """Tests Poisson distribution against scipy.stats.poisson."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_poisson_scipy skipped (scipy not available)")
        return

    var p = Poisson(4.0)
    var ks: List[Int] = [0, 1, 2, 4, 8]

    for i in range(len(ks)):
        var k = ks[i]
        var sp_pmf = _py_f64(sp.poisson.pmf(k, 4.0))
        var sp_cdf = _py_f64(sp.poisson.cdf(k, 4.0))
        assert_almost_equal(p.pmf(k), sp_pmf, atol=1e-10)
        assert_almost_equal(p.cdf(k), sp_cdf, atol=1e-10)


def test_poisson_ppf_large_mu() raises:
    """Tests Poisson PPF for large mu (regression test for underflow bug)."""
    var p = Poisson(1000.0)
    # ppf(0.5) should be near 1000, not _MAX_K
    var k = p.ppf(0.5)
    assert_true(
        k >= 990 and k <= 1010, "Poisson(1000).ppf(0.5) should be ~1000"
    )
    # Round-trip: CDF(ppf(q)) >= q
    assert_true(p.cdf(k) >= 0.5, "CDF(ppf(0.5)) must be >= 0.5")
    assert_true(p.cdf(k - 1) < 0.5, "CDF(ppf(0.5)-1) must be < 0.5")


# ===----------------------------------------------------------------------=== #
# Uniform distribution tests
# ===----------------------------------------------------------------------=== #


def test_uniform_pdf_cdf() raises:
    """Tests Uniform PDF/CDF at known values."""
    var u = Uniform(0.0, 2.0)
    assert_almost_equal(u.pdf(1.0), 0.5, atol=1e-15)
    assert_almost_equal(u.pdf(-1.0), 0.0, atol=1e-15)
    assert_almost_equal(u.pdf(3.0), 0.0, atol=1e-15)
    assert_almost_equal(u.cdf(0.0), 0.0, atol=1e-15)
    assert_almost_equal(u.cdf(1.0), 0.5, atol=1e-15)
    assert_almost_equal(u.cdf(2.0), 1.0, atol=1e-15)


def test_uniform_ppf_roundtrip() raises:
    """Tests Uniform PPF round-trip."""
    var u = Uniform(-3.0, 5.0)
    assert_almost_equal(u.cdf(u.ppf(0.25)), 0.25, atol=1e-12)
    assert_almost_equal(u.cdf(u.ppf(0.75)), 0.75, atol=1e-12)


def test_uniform_stats() raises:
    """Tests Uniform distribution statistics."""
    var u = Uniform(0.0, 4.0)
    assert_almost_equal(u.mean(), 2.0, atol=1e-15)
    assert_almost_equal(u.variance(), 16.0 / 12.0, atol=1e-12)


def test_uniform_scipy() raises:
    """Tests Uniform distribution against scipy.stats.uniform."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_uniform_scipy skipped (scipy not available)")
        return

    var u = Uniform(-2.0, 3.0)
    var xs: List[Float64] = [-2.0, -1.0, 0.0, 1.5, 3.0]

    for i in range(len(xs)):
        var x = xs[i]
        # scipy.stats.uniform is parameterized as (loc, scale) with scale = b - a.
        var sp_pdf = _py_f64(sp.uniform.pdf(x, loc=-2.0, scale=5.0))
        var sp_cdf = _py_f64(sp.uniform.cdf(x, loc=-2.0, scale=5.0))
        assert_almost_equal(u.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(u.cdf(x), sp_cdf, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# LogNormal distribution tests
# ===----------------------------------------------------------------------=== #


def test_lognormal_pdf_cdf() raises:
    """Tests LogNormal PDF/CDF at known values."""
    var ln = LogNormal(0.0, 1.0)
    assert_almost_equal(ln.pdf(-1.0), 0.0, atol=1e-15)
    assert_almost_equal(ln.cdf(0.0), 0.0, atol=1e-15)
    # median = exp(mu) = 1
    assert_almost_equal(ln.cdf(1.0), 0.5, atol=1e-10)


def test_lognormal_ppf_roundtrip() raises:
    """Tests LogNormal PPF round-trip."""
    var ln = LogNormal(0.5, 0.75)
    assert_almost_equal(ln.cdf(ln.ppf(0.3)), 0.3, atol=1e-8)
    assert_almost_equal(ln.cdf(ln.ppf(0.7)), 0.7, atol=1e-8)


def test_lognormal_stats() raises:
    """Tests LogNormal distribution statistics."""
    var ln = LogNormal(0.0, 1.0)
    assert_almost_equal(ln.median(), 1.0, atol=1e-12)
    assert_almost_equal(ln.mean(), exp(0.5), atol=1e-12)


def test_lognormal_scipy() raises:
    """Tests LogNormal distribution against scipy.stats.lognorm."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_lognormal_scipy skipped (scipy not available)")
        return

    var ln = LogNormal(0.2, 0.8)
    var xs: List[Float64] = [0.5, 1.0, 2.0, 5.0]

    for i in range(len(xs)):
        var x = xs[i]
        # scipy.stats.lognorm(s, scale=exp(mu)) matches our (mu, sigma).
        var sp_pdf = _py_f64(sp.lognorm.pdf(x, 0.8, scale=exp(0.2)))
        var sp_cdf = _py_f64(sp.lognorm.cdf(x, 0.8, scale=exp(0.2)))
        assert_almost_equal(ln.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(ln.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Weibull distribution tests
# ===----------------------------------------------------------------------=== #


def test_weibull_pdf_cdf() raises:
    """Tests Weibull PDF/CDF at known values."""
    var w = Weibull(1.0, 1.0)
    # Weibull(1,1) = Exponential(1).
    assert_almost_equal(w.pdf(1.0), exp(-1.0), atol=1e-12)
    assert_almost_equal(w.cdf(1.0), 1.0 - exp(-1.0), atol=1e-12)
    assert_almost_equal(w.pdf(-1.0), 0.0, atol=1e-15)


def test_weibull_ppf_roundtrip() raises:
    """Tests Weibull PPF round-trip."""
    var w = Weibull(2.0, 3.0)
    assert_almost_equal(w.cdf(w.ppf(0.4)), 0.4, atol=1e-10)
    assert_almost_equal(w.cdf(w.ppf(0.9)), 0.9, atol=1e-10)


def test_weibull_stats() raises:
    """Tests Weibull distribution statistics."""
    var w = Weibull(1.0, 2.0)
    # Weibull(k=1, lam) = Exponential(1/lam): mean = lam.
    assert_almost_equal(w.mean(), 2.0, atol=1e-8)


def test_weibull_scipy() raises:
    """Tests Weibull distribution against scipy.stats.weibull_min."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_weibull_scipy skipped (scipy not available)")
        return

    var w = Weibull(1.5, 2.0)
    var xs: List[Float64] = [0.5, 1.0, 2.0, 4.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.weibull_min.pdf(x, 1.5, scale=2.0))
        var sp_cdf = _py_f64(sp.weibull_min.cdf(x, 1.5, scale=2.0))
        assert_almost_equal(w.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(w.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Cauchy distribution tests
# ===----------------------------------------------------------------------=== #


def test_cauchy_pdf_cdf() raises:
    """Tests Cauchy PDF/CDF at known values."""
    var c = Cauchy(0.0, 1.0)
    # PDF at 0 for standard Cauchy = 1/π.
    assert_almost_equal(c.pdf(0.0), 1.0 / 3.1415926535897932, atol=1e-12)
    assert_almost_equal(c.cdf(0.0), 0.5, atol=1e-12)
    # Symmetric about loc.
    assert_almost_equal(c.pdf(1.0), c.pdf(-1.0), atol=1e-15)


def test_cauchy_ppf_roundtrip() raises:
    """Tests Cauchy PPF round-trip."""
    var c = Cauchy(2.0, 1.5)
    assert_almost_equal(c.cdf(c.ppf(0.2)), 0.2, atol=1e-10)
    assert_almost_equal(c.cdf(c.ppf(0.8)), 0.8, atol=1e-10)


def test_cauchy_undefined_moments() raises:
    """Tests that Cauchy mean/variance are undefined (NaN)."""
    var c = Cauchy(0.0, 1.0)
    assert_true(c.mean() != c.mean(), "Cauchy mean should be NaN")
    assert_true(c.variance() != c.variance(), "Cauchy variance should be NaN")
    assert_almost_equal(c.median(), 0.0, atol=1e-15)


def test_cauchy_scipy() raises:
    """Tests Cauchy distribution against scipy.stats.cauchy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_cauchy_scipy skipped (scipy not available)")
        return

    var c = Cauchy(1.0, 2.0)
    var xs: List[Float64] = [-3.0, -1.0, 1.0, 3.0, 6.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.cauchy.pdf(x, loc=1.0, scale=2.0))
        var sp_cdf = _py_f64(sp.cauchy.cdf(x, loc=1.0, scale=2.0))
        assert_almost_equal(c.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(c.cdf(x), sp_cdf, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# Laplace distribution tests
# ===----------------------------------------------------------------------=== #


def test_laplace_pdf_cdf() raises:
    """Tests Laplace PDF/CDF at known values."""
    var l = Laplace(0.0, 1.0)
    assert_almost_equal(l.pdf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(l.cdf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(l.pdf(1.0), l.pdf(-1.0), atol=1e-15)


def test_laplace_ppf_roundtrip() raises:
    """Tests Laplace PPF round-trip."""
    var l = Laplace(1.0, 2.0)
    assert_almost_equal(l.cdf(l.ppf(0.3)), 0.3, atol=1e-12)
    assert_almost_equal(l.cdf(l.ppf(0.7)), 0.7, atol=1e-12)


def test_laplace_stats() raises:
    """Tests Laplace distribution statistics."""
    var l = Laplace(2.0, 3.0)
    assert_almost_equal(l.mean(), 2.0, atol=1e-15)
    assert_almost_equal(l.variance(), 18.0, atol=1e-12)


def test_laplace_scipy() raises:
    """Tests Laplace distribution against scipy.stats.laplace."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_laplace_scipy skipped (scipy not available)")
        return

    var l = Laplace(0.5, 1.5)
    var xs: List[Float64] = [-2.0, -0.5, 0.5, 2.0, 4.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.laplace.pdf(x, loc=0.5, scale=1.5))
        var sp_cdf = _py_f64(sp.laplace.cdf(x, loc=0.5, scale=1.5))
        assert_almost_equal(l.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(l.cdf(x), sp_cdf, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# Logistic distribution tests
# ===----------------------------------------------------------------------=== #


def test_logistic_pdf_cdf() raises:
    """Tests Logistic PDF/CDF at known values."""
    var lg = Logistic(0.0, 1.0)
    assert_almost_equal(lg.pdf(0.0), 0.25, atol=1e-15)
    assert_almost_equal(lg.cdf(0.0), 0.5, atol=1e-15)
    assert_almost_equal(lg.pdf(1.0), lg.pdf(-1.0), atol=1e-15)


def test_logistic_ppf_roundtrip() raises:
    """Tests Logistic PPF round-trip."""
    var lg = Logistic(1.0, 2.0)
    assert_almost_equal(lg.cdf(lg.ppf(0.25)), 0.25, atol=1e-12)
    assert_almost_equal(lg.cdf(lg.ppf(0.85)), 0.85, atol=1e-12)


def test_logistic_stats() raises:
    """Tests Logistic distribution statistics."""
    var lg = Logistic(0.0, 1.0)
    assert_almost_equal(lg.mean(), 0.0, atol=1e-15)
    # variance = pi^2/3
    assert_almost_equal(lg.variance(), 3.28986813369645, atol=1e-10)


def test_logistic_scipy() raises:
    """Tests Logistic distribution against scipy.stats.logistic."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_logistic_scipy skipped (scipy not available)")
        return

    var lg = Logistic(-1.0, 2.0)
    var xs: List[Float64] = [-4.0, -1.0, 0.0, 2.0, 5.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.logistic.pdf(x, loc=-1.0, scale=2.0))
        var sp_cdf = _py_f64(sp.logistic.cdf(x, loc=-1.0, scale=2.0))
        assert_almost_equal(lg.pdf(x), sp_pdf, atol=1e-12)
        assert_almost_equal(lg.cdf(x), sp_cdf, atol=1e-12)


# ===----------------------------------------------------------------------=== #
# Pareto distribution tests
# ===----------------------------------------------------------------------=== #


def test_pareto_pdf_cdf() raises:
    """Tests Pareto PDF/CDF at known values."""
    var p = Pareto(2.0, 1.0)
    assert_almost_equal(p.pdf(1.0), 2.0, atol=1e-12)
    assert_almost_equal(p.pdf(0.5), 0.0, atol=1e-15)
    assert_almost_equal(p.cdf(1.0), 0.0, atol=1e-15)


def test_pareto_ppf_roundtrip() raises:
    """Tests Pareto PPF round-trip."""
    var p = Pareto(3.0, 2.0)
    assert_almost_equal(p.cdf(p.ppf(0.4)), 0.4, atol=1e-10)
    assert_almost_equal(p.cdf(p.ppf(0.9)), 0.9, atol=1e-10)


def test_pareto_stats() raises:
    """Tests Pareto distribution statistics."""
    var p = Pareto(3.0, 2.0)
    # mean = alpha*x_m/(alpha-1) = 3*2/2 = 3
    assert_almost_equal(p.mean(), 3.0, atol=1e-12)


def test_pareto_scipy() raises:
    """Tests Pareto distribution against scipy.stats.pareto."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_pareto_scipy skipped (scipy not available)")
        return

    var p = Pareto(2.5, 1.0)
    var xs: List[Float64] = [1.0, 1.5, 3.0, 8.0]

    for i in range(len(xs)):
        var x = xs[i]
        # scipy.stats.pareto(b, scale=x_m) matches our (alpha, x_m).
        var sp_pdf = _py_f64(sp.pareto.pdf(x, 2.5, scale=1.0))
        var sp_cdf = _py_f64(sp.pareto.cdf(x, 2.5, scale=1.0))
        assert_almost_equal(p.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(p.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Rayleigh distribution tests
# ===----------------------------------------------------------------------=== #


def test_rayleigh_pdf_cdf() raises:
    """Tests Rayleigh PDF/CDF at known values."""
    var r = Rayleigh(1.0)
    assert_almost_equal(r.pdf(-1.0), 0.0, atol=1e-15)
    assert_almost_equal(r.cdf(0.0), 0.0, atol=1e-15)
    # PDF at x=1, sigma=1: 1 * exp(-0.5) = exp(-0.5)
    assert_almost_equal(r.pdf(1.0), exp(-0.5), atol=1e-12)


def test_rayleigh_ppf_roundtrip() raises:
    """Tests Rayleigh PPF round-trip."""
    var r = Rayleigh(2.0)
    assert_almost_equal(r.cdf(r.ppf(0.3)), 0.3, atol=1e-10)
    assert_almost_equal(r.cdf(r.ppf(0.8)), 0.8, atol=1e-10)


def test_rayleigh_stats() raises:
    """Tests Rayleigh distribution statistics."""
    var r = Rayleigh(1.0)
    # mean = sigma * sqrt(pi/2)
    assert_almost_equal(r.mean(), sqrt(3.1415926535897932 / 2.0), atol=1e-10)


def test_rayleigh_scipy() raises:
    """Tests Rayleigh distribution against scipy.stats.rayleigh."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_rayleigh_scipy skipped (scipy not available)")
        return

    var r = Rayleigh(2.0)
    var xs: List[Float64] = [0.5, 1.0, 3.0, 5.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.rayleigh.pdf(x, scale=2.0))
        var sp_cdf = _py_f64(sp.rayleigh.cdf(x, scale=2.0))
        assert_almost_equal(r.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(r.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# HalfNormal distribution tests
# ===----------------------------------------------------------------------=== #


def test_halfnormal_pdf_cdf() raises:
    """Tests HalfNormal PDF/CDF at known values."""
    var h = HalfNormal(0.0, 1.0)
    assert_almost_equal(h.pdf(-1.0), 0.0, atol=1e-15)
    assert_almost_equal(h.cdf(0.0), 0.0, atol=1e-15)
    # PDF at 0: sqrt(2/pi)
    assert_almost_equal(h.pdf(0.0), 0.7978845608028654, atol=1e-12)


def test_halfnormal_ppf_roundtrip() raises:
    """Tests HalfNormal PPF round-trip."""
    var h = HalfNormal(1.0, 2.0)
    assert_almost_equal(h.cdf(h.ppf(0.25)), 0.25, atol=1e-10)
    assert_almost_equal(h.cdf(h.ppf(0.9)), 0.9, atol=1e-10)


def test_halfnormal_stats() raises:
    """Tests HalfNormal distribution statistics."""
    var h = HalfNormal(0.0, 1.0)
    # mean = sqrt(2/pi)
    assert_almost_equal(h.mean(), 0.7978845608028654, atol=1e-12)


def test_halfnormal_scipy() raises:
    """Tests HalfNormal distribution against scipy.stats.halfnorm."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_halfnormal_scipy skipped (scipy not available)")
        return

    var h = HalfNormal(0.5, 1.5)
    var xs: List[Float64] = [0.5, 1.0, 2.0, 4.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.halfnorm.pdf(x, loc=0.5, scale=1.5))
        var sp_cdf = _py_f64(sp.halfnorm.cdf(x, loc=0.5, scale=1.5))
        assert_almost_equal(h.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(h.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# HalfCauchy distribution tests
# ===----------------------------------------------------------------------=== #


def test_halfcauchy_pdf_cdf() raises:
    """Tests HalfCauchy PDF/CDF at known values."""
    var h = HalfCauchy(0.0, 1.0)
    assert_almost_equal(h.pdf(-1.0), 0.0, atol=1e-15)
    assert_almost_equal(h.cdf(0.0), 0.0, atol=1e-15)
    # PDF at 0: 2/pi
    assert_almost_equal(h.pdf(0.0), 2.0 / 3.1415926535897932, atol=1e-12)


def test_halfcauchy_ppf_roundtrip() raises:
    """Tests HalfCauchy PPF round-trip."""
    var h = HalfCauchy(1.0, 2.0)
    assert_almost_equal(h.cdf(h.ppf(0.3)), 0.3, atol=1e-10)
    assert_almost_equal(h.cdf(h.ppf(0.7)), 0.7, atol=1e-10)


def test_halfcauchy_undefined_moments() raises:
    """Tests that HalfCauchy mean/variance are undefined (NaN)."""
    var h = HalfCauchy(0.0, 1.0)
    assert_true(h.mean() != h.mean(), "HalfCauchy mean should be NaN")
    assert_true(
        h.variance() != h.variance(), "HalfCauchy variance should be NaN"
    )


def test_halfcauchy_scipy() raises:
    """Tests HalfCauchy distribution against scipy.stats.halfcauchy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_halfcauchy_scipy skipped (scipy not available)")
        return

    var h = HalfCauchy(0.5, 1.5)
    var xs: List[Float64] = [0.5, 1.0, 3.0, 8.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.halfcauchy.pdf(x, loc=0.5, scale=1.5))
        var sp_cdf = _py_f64(sp.halfcauchy.cdf(x, loc=0.5, scale=1.5))
        assert_almost_equal(h.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(h.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Gumbel distribution tests
# ===----------------------------------------------------------------------=== #


def test_gumbel_pdf_cdf() raises:
    """Tests Gumbel PDF/CDF at known values."""
    var g = Gumbel(0.0, 1.0)
    # CDF at 0: exp(-1)
    assert_almost_equal(g.cdf(0.0), exp(-1.0), atol=1e-12)


def test_gumbel_ppf_roundtrip() raises:
    """Tests Gumbel PPF round-trip."""
    var g = Gumbel(1.0, 2.0)
    assert_almost_equal(g.cdf(g.ppf(0.2)), 0.2, atol=1e-10)
    assert_almost_equal(g.cdf(g.ppf(0.9)), 0.9, atol=1e-10)


def test_gumbel_stats() raises:
    """Tests Gumbel distribution statistics."""
    var g = Gumbel(0.0, 1.0)
    # mean = Euler-Mascheroni constant
    assert_almost_equal(g.mean(), 0.5772156649015329, atol=1e-12)


def test_gumbel_scipy() raises:
    """Tests Gumbel distribution against scipy.stats.gumbel_r."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_gumbel_scipy skipped (scipy not available)")
        return

    var g = Gumbel(0.5, 1.5)
    var xs: List[Float64] = [-2.0, 0.0, 1.0, 3.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.gumbel_r.pdf(x, loc=0.5, scale=1.5))
        var sp_cdf = _py_f64(sp.gumbel_r.cdf(x, loc=0.5, scale=1.5))
        assert_almost_equal(g.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(g.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Triangular distribution tests
# ===----------------------------------------------------------------------=== #


def test_triangular_pdf_cdf() raises:
    """Tests Triangular PDF/CDF at known values."""
    var t = Triangular(0.0, 1.0, 0.5)
    assert_almost_equal(t.pdf(-0.1), 0.0, atol=1e-15)
    assert_almost_equal(t.pdf(1.1), 0.0, atol=1e-15)
    # peak at mode: 2 / (b - a) = 2
    assert_almost_equal(t.pdf(0.5), 2.0, atol=1e-12)
    assert_almost_equal(t.cdf(0.0), 0.0, atol=1e-15)
    assert_almost_equal(t.cdf(1.0), 1.0, atol=1e-15)
    assert_almost_equal(t.cdf(0.5), 0.5, atol=1e-12)


def test_triangular_ppf_roundtrip() raises:
    """Tests Triangular PPF round-trip."""
    var t = Triangular(0.0, 10.0, 3.0)
    assert_almost_equal(t.cdf(t.ppf(0.2)), 0.2, atol=1e-10)
    assert_almost_equal(t.cdf(t.ppf(0.7)), 0.7, atol=1e-10)


def test_triangular_stats() raises:
    """Tests Triangular distribution statistics."""
    var t = Triangular(0.0, 1.0, 0.5)
    # mean = (a+b+c)/3
    assert_almost_equal(t.mean(), 0.5, atol=1e-12)


def test_triangular_scipy() raises:
    """Tests Triangular distribution against scipy.stats.triang."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_triangular_scipy skipped (scipy not available)")
        return

    var a = 0.0
    var b = 10.0
    var c = 3.0
    var t = Triangular(a, b, c)
    var xs: List[Float64] = [1.0, 3.0, 5.0, 9.0]

    for i in range(len(xs)):
        var x = xs[i]
        # scipy.stats.triang(c_frac, loc=a, scale=b-a), c_frac=(c-a)/(b-a).
        var sp_pdf = _py_f64(
            sp.triang.pdf(x, (c - a) / (b - a), loc=a, scale=b - a)
        )
        var sp_cdf = _py_f64(
            sp.triang.cdf(x, (c - a) / (b - a), loc=a, scale=b - a)
        )
        assert_almost_equal(t.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(t.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# VonMises distribution tests
# ===----------------------------------------------------------------------=== #


def test_vonmises_pdf_symmetry() raises:
    """Tests VonMises PDF is symmetric about mu."""
    var v = VonMises(0.0, 2.0)
    assert_almost_equal(v.pdf(0.5), v.pdf(-0.5), atol=1e-12)


def test_vonmises_cdf_bounds() raises:
    """Tests VonMises CDF at principal-branch boundaries."""
    var v = VonMises(0.0, 1.5)
    assert_almost_equal(v.cdf(0.0 - 3.1415926535897932), 0.0, atol=1e-8)
    assert_almost_equal(v.cdf(0.0 + 3.1415926535897932), 1.0, atol=1e-8)
    assert_almost_equal(v.cdf(0.0), 0.5, atol=1e-6)


def test_vonmises_ppf_roundtrip() raises:
    """Tests VonMises PPF round-trip."""
    var v = VonMises(0.3, 2.5)
    assert_almost_equal(v.cdf(v.ppf(0.25)), 0.25, atol=1e-6)
    assert_almost_equal(v.cdf(v.ppf(0.75)), 0.75, atol=1e-6)


def test_vonmises_scipy() raises:
    """Tests VonMises distribution against scipy.stats.vonmises."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_vonmises_scipy skipped (scipy not available)")
        return

    var v = VonMises(0.2, 3.0)
    var xs: List[Float64] = [-1.0, 0.0, 0.5, 1.5]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.vonmises.pdf(x, 3.0, loc=0.2))
        var sp_cdf = _py_f64(sp.vonmises.cdf(x, 3.0, loc=0.2))
        assert_almost_equal(v.pdf(x), sp_pdf, atol=1e-8)
        assert_almost_equal(v.cdf(x), sp_cdf, atol=1e-5)


# ===----------------------------------------------------------------------=== #
# TruncNorm distribution tests
# ===----------------------------------------------------------------------=== #


def test_truncnorm_pdf_cdf() raises:
    """Tests TruncNorm PDF/CDF at known values."""
    var t = TruncNorm(-1.0, 1.0, 0.0, 1.0)
    assert_almost_equal(t.pdf(-2.0), 0.0, atol=1e-15)
    assert_almost_equal(t.pdf(2.0), 0.0, atol=1e-15)
    assert_almost_equal(t.cdf(-1.0), 0.0, atol=1e-10)
    assert_almost_equal(t.cdf(1.0), 1.0, atol=1e-10)
    # Symmetric about 0.
    assert_almost_equal(t.pdf(0.5), t.pdf(-0.5), atol=1e-12)


def test_truncnorm_ppf_roundtrip() raises:
    """Tests TruncNorm PPF round-trip."""
    var t = TruncNorm(-2.0, 2.0, 1.0, 2.0)
    assert_almost_equal(t.cdf(t.ppf(0.3)), 0.3, atol=1e-8)
    assert_almost_equal(t.cdf(t.ppf(0.7)), 0.7, atol=1e-8)


def test_truncnorm_stats() raises:
    """Tests TruncNorm distribution statistics (symmetric case)."""
    var t = TruncNorm(-1.0, 1.0, 0.0, 1.0)
    # Symmetric truncation about loc => mean == loc.
    assert_almost_equal(t.mean(), 0.0, atol=1e-10)


def test_truncnorm_scipy() raises:
    """Tests TruncNorm distribution against scipy.stats.truncnorm."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("test_truncnorm_scipy skipped (scipy not available)")
        return

    var t = TruncNorm(-1.5, 2.0, 0.5, 1.5)
    var xs: List[Float64] = [-1.5, -0.5, 0.5, 2.0, 3.0]

    for i in range(len(xs)):
        var x = xs[i]
        var sp_pdf = _py_f64(sp.truncnorm.pdf(x, -1.5, 2.0, loc=0.5, scale=1.5))
        var sp_cdf = _py_f64(sp.truncnorm.cdf(x, -1.5, 2.0, loc=0.5, scale=1.5))
        assert_almost_equal(t.pdf(x), sp_pdf, atol=1e-10)
        assert_almost_equal(t.cdf(x), sp_cdf, atol=1e-10)


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
