# ===----------------------------------------------------------------------=== #
# StaMojo - Tests for hypothesis tests and correlation
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Tests for the stats.tests and stats.correlation submodules.

Covers:
  - One-sample, two-sample (Welch and pooled), paired t-tests
  - Chi-squared goodness-of-fit and independence tests
  - Kolmogorov-Smirnov test
  - Pearson, Spearman, Kendall correlation
  - One-way ANOVA
"""

from math import sqrt
from python import Python, PythonObject
from testing import assert_almost_equal

from stamojo.stats import (
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    chi2_gof,
    chi2_ind,
    ks_1samp,
    f_oneway,
    pearsonr,
    spearmanr,
    kendalltau,
)


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
# t-test tests
# ===----------------------------------------------------------------------=== #


fn test_ttest_1samp_basic() raises:
    """Test one-sample t-test with known data."""
    # Data with mean = 3.0; test H0: mu = 0
    var data: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]

    var result = ttest_1samp(data, 0.0)
    var t_stat = result[0]
    var p_val = result[1]

    # t = (3.0 - 0) / (sqrt(2.5) / sqrt(5)) = 3.0 / 0.7071... ≈ 4.2426
    assert_almost_equal(t_stat, 3.0 / sqrt(2.5 / 5.0), atol=1e-10)
    # p should be small (reject H0)
    if p_val > 0.05:
        raise Error("ttest_1samp: expected p < 0.05, got " + String(p_val))

    print("✓ test_ttest_1samp_basic passed")


fn test_ttest_1samp_no_effect() raises:
    """Test one-sample t-test when data mean ≈ mu0."""
    var data: List[Float64] = [-1.0, 0.0, 1.0]

    var result = ttest_1samp(data, 0.0)
    assert_almost_equal(result[0], 0.0, atol=1e-12)
    # p-value should be 1.0 (no evidence against H0)
    assert_almost_equal(result[1], 1.0, atol=1e-6)

    print("✓ test_ttest_1samp_no_effect passed")


fn test_ttest_1samp_scipy() raises:
    """Test one-sample t-test against scipy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_ttest_1samp_scipy skipped (scipy not available)")
        return

    var data: List[Float64] = [5.1, 4.8, 5.3, 5.0, 4.9, 5.2]

    var py_data = Python.evaluate("[5.1, 4.8, 5.3, 5.0, 4.9, 5.2]")
    var sp_result = sp.ttest_1samp(py_data, 5.0)
    var sp_t = _py_f64(sp_result[0])
    var sp_p = _py_f64(sp_result[1])

    var result = ttest_1samp(data, 5.0)
    assert_almost_equal(result[0], sp_t, atol=1e-6)
    assert_almost_equal(result[1], sp_p, atol=1e-4)

    print("✓ test_ttest_1samp_scipy passed")


fn test_ttest_ind_welch() raises:
    """Test Welch's two-sample t-test."""
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]

    var y: List[Float64] = [4.0, 5.0, 6.0, 7.0, 8.0]

    var result = ttest_ind(x, y, equal_var=False)
    # Means: 3.0 vs 6.0, should be significant
    if result[1] > 0.05:
        raise Error(
            "ttest_ind Welch: expected p < 0.05, got " + String(result[1])
        )

    print("✓ test_ttest_ind_welch passed")


fn test_ttest_ind_scipy() raises:
    """Test Welch's t-test against scipy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_ttest_ind_scipy skipped (scipy not available)")
        return

    var x: List[Float64] = [2.3, 3.1, 2.8, 3.5, 2.9]

    var y: List[Float64] = [4.2, 4.8, 3.9, 5.1, 4.5]

    var py_x = Python.evaluate("[2.3, 3.1, 2.8, 3.5, 2.9]")
    var py_y = Python.evaluate("[4.2, 4.8, 3.9, 5.1, 4.5]")
    var sp_result = sp.ttest_ind(py_x, py_y, equal_var=False)
    var sp_t = _py_f64(sp_result[0])
    var sp_p = _py_f64(sp_result[1])

    var result = ttest_ind(x, y, equal_var=False)
    assert_almost_equal(result[0], sp_t, atol=1e-4)
    assert_almost_equal(result[1], sp_p, atol=1e-3)

    print("✓ test_ttest_ind_scipy passed")


fn test_ttest_rel() raises:
    """Test paired t-test."""
    # Before and after treatment.
    var before: List[Float64] = [10.0, 12.0, 14.0, 11.0, 13.0]

    var after: List[Float64] = [12.0, 14.0, 16.0, 13.0, 15.0]

    var result = ttest_rel(before, after)
    # Differences are all 2.0, so t should be large and p very small.
    var diffs: List[Float64] = [-2.0, -2.0, -2.0, -2.0, -2.0]

    var result_1samp = ttest_1samp(diffs, 0.0)
    assert_almost_equal(result[0], result_1samp[0], atol=1e-12)
    assert_almost_equal(result[1], result_1samp[1], atol=1e-12)

    print("✓ test_ttest_rel passed")


# ===----------------------------------------------------------------------=== #
# Chi-squared tests
# ===----------------------------------------------------------------------=== #


fn test_chi2_gof_fair_die() raises:
    """Test chi-squared GoF for a fair die."""
    var observed: List[Float64] = [16.0, 18.0, 16.0, 14.0, 12.0, 14.0]

    var expected: List[Float64] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

    var result = chi2_gof(observed, expected)
    # chi2 = Σ(O-E)²/E = (1+9+1+1+9+1)/15 = 22/15 ≈ 1.467
    assert_almost_equal(result[0], 22.0 / 15.0, atol=1e-10)
    # p should not be significant (fair die is plausible)
    if result[1] < 0.05:
        raise Error("chi2_gof: expected non-significant, got p=" + String(result[1]))

    print("✓ test_chi2_gof_fair_die passed")


fn test_chi2_ind_basic() raises:
    """Test chi-squared independence test with 2×2 table."""
    # Example: [[10, 20], [20, 40]]
    # This table has perfect proportionality, chi2 ≈ 0.
    var row1: List[Float64] = [10.0, 20.0]
    var row2: List[Float64] = [20.0, 40.0]
    var table = List[List[Float64]]()
    table.append(row1^)
    table.append(row2^)

    var result = chi2_ind(table)
    assert_almost_equal(result[0], 0.0, atol=1e-10)
    assert_almost_equal(result[1], 1.0, atol=1e-4)

    print("✓ test_chi2_ind_basic passed")


fn test_chi2_ind_scipy() raises:
    """Test chi-squared independence test against scipy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_chi2_ind_scipy skipped (scipy not available)")
        return

    # A table with non-trivial dependence.
    var row1: List[Float64] = [20.0, 15.0]
    var row2: List[Float64] = [10.0, 30.0]
    var table = List[List[Float64]]()
    table.append(row1^)
    table.append(row2^)

    var np = Python.import_module("numpy")
    var py_table = np.array(
        Python.evaluate("[[20, 15], [10, 30]]")
    )
    var sp_result = sp.chi2_contingency(py_table, correction=False)
    var sp_chi2 = _py_f64(sp_result[0])
    var sp_p = _py_f64(sp_result[1])

    var result = chi2_ind(table)
    assert_almost_equal(result[0], sp_chi2, atol=1e-4)
    assert_almost_equal(result[1], sp_p, atol=1e-3)

    print("✓ test_chi2_ind_scipy passed")


# ===----------------------------------------------------------------------=== #
# Kolmogorov-Smirnov test
# ===----------------------------------------------------------------------=== #


fn test_ks_normal_data() raises:
    """Test KS test with data drawn from N(0,1) (should not reject)."""
    # Pre-computed standard normal quantiles (approx).
    var data: List[Float64] = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]

    var result = ks_1samp(data)
    # D should be small, p should be large.
    if result[1] < 0.05:
        raise Error("ks_1samp: expected p > 0.05 for normal data, got " + String(result[1]))

    print("✓ test_ks_normal_data passed")


fn test_ks_uniform_data() raises:
    """Test KS test with uniform data (should reject N(0,1))."""
    # Uniform [0, 10] data — definitely not N(0,1).
    var data = List[Float64]()
    for i in range(20):
        data.append(Float64(i) * 0.5)

    var result = ks_1samp(data)
    # D should be large, p should be very small.
    if result[1] > 0.05:
        raise Error(
            "ks_1samp: expected p < 0.05 for uniform data, got "
            + String(result[1])
        )

    print("✓ test_ks_uniform_data passed")


# ===----------------------------------------------------------------------=== #
# Correlation tests
# ===----------------------------------------------------------------------=== #


fn test_pearsonr_perfect() raises:
    """Test Pearson correlation with perfectly correlated data."""
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(10):
        x.append(Float64(i))
        y.append(Float64(i) * 2.0 + 1.0)

    var result = pearsonr(x, y)
    assert_almost_equal(result[0], 1.0, atol=1e-10)
    # p-value should be very small.
    if result[1] > 0.001:
        raise Error("pearsonr: expected p ≈ 0 for perfect correlation")

    print("✓ test_pearsonr_perfect passed")


fn test_pearsonr_negative() raises:
    """Test Pearson correlation for negative correlation."""
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(10):
        x.append(Float64(i))
        y.append(-Float64(i) * 3.0)

    var result = pearsonr(x, y)
    assert_almost_equal(result[0], -1.0, atol=1e-10)

    print("✓ test_pearsonr_negative passed")


fn test_pearsonr_scipy() raises:
    """Test Pearson correlation against scipy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_pearsonr_scipy skipped (scipy not available)")
        return

    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    var y: List[Float64] = [2.1, 3.8, 5.2, 6.9, 8.1, 10.5]

    var py_x = Python.evaluate("[1., 2., 3., 4., 5., 6.]")
    var py_y = Python.evaluate("[2.1, 3.8, 5.2, 6.9, 8.1, 10.5]")
    var sp_result = sp.pearsonr(py_x, py_y)
    var sp_r = _py_f64(sp_result[0])
    var sp_p = _py_f64(sp_result[1])

    var result = pearsonr(x, y)
    assert_almost_equal(result[0], sp_r, atol=1e-6)
    assert_almost_equal(result[1], sp_p, atol=1e-3)

    print("✓ test_pearsonr_scipy passed")


fn test_spearmanr_perfect_monotone() raises:
    """Test Spearman correlation with perfect monotone data."""
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(10):
        x.append(Float64(i))
        y.append(Float64(i) ** 2)  # Monotone but not linear

    var result = spearmanr(x, y)
    assert_almost_equal(result[0], 1.0, atol=1e-10)

    print("✓ test_spearmanr_perfect_monotone passed")


fn test_spearmanr_scipy() raises:
    """Test Spearman correlation against scipy."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_spearmanr_scipy skipped (scipy not available)")
        return

    var x: List[Float64] = [1.0, 3.0, 2.0, 5.0, 4.0]

    var y: List[Float64] = [5.0, 6.0, 7.0, 8.0, 7.0]

    var py_x = Python.evaluate("[1., 3., 2., 5., 4.]")
    var py_y = Python.evaluate("[5., 6., 7., 8., 7.]")
    var sp_result = sp.spearmanr(py_x, py_y)
    var sp_rho = _py_f64(sp_result[0])

    var result = spearmanr(x, y)
    assert_almost_equal(result[0], sp_rho, atol=1e-4)

    print("✓ test_spearmanr_scipy passed")


fn test_kendalltau_concordant() raises:
    """Test Kendall's tau with perfectly concordant data."""
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(10):
        x.append(Float64(i))
        y.append(Float64(i))

    var result = kendalltau(x, y)
    assert_almost_equal(result[0], 1.0, atol=1e-10)

    print("✓ test_kendalltau_concordant passed")


fn test_kendalltau_discordant() raises:
    """Test Kendall's tau with perfectly discordant data."""
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(10):
        x.append(Float64(i))
        y.append(Float64(9 - i))

    var result = kendalltau(x, y)
    assert_almost_equal(result[0], -1.0, atol=1e-10)

    print("✓ test_kendalltau_discordant passed")


# ===----------------------------------------------------------------------=== #
# One-way ANOVA
# ===----------------------------------------------------------------------=== #


fn test_f_oneway_identical() raises:
    """Test ANOVA with identical group means (should not reject)."""
    var g1: List[Float64] = [1.0, 2.0, 3.0]
    var g2: List[Float64] = [1.0, 2.0, 3.0]
    var g3: List[Float64] = [1.0, 2.0, 3.0]

    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)

    var result = f_oneway(groups)
    assert_almost_equal(result[0], 0.0, atol=1e-10)
    assert_almost_equal(result[1], 1.0, atol=1e-4)

    print("✓ test_f_oneway_identical passed")


fn test_f_oneway_different() raises:
    """Test ANOVA with clearly different group means."""
    var g1: List[Float64] = [1.0, 2.0, 3.0]
    var g2: List[Float64] = [10.0, 11.0, 12.0]
    var g3: List[Float64] = [20.0, 21.0, 22.0]

    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)

    var result = f_oneway(groups)
    # Should be highly significant.
    if result[1] > 0.001:
        raise Error(
            "f_oneway: expected p < 0.001 for different means, got "
            + String(result[1])
        )

    print("✓ test_f_oneway_different passed")


fn test_f_oneway_scipy() raises:
    """Test ANOVA against scipy.stats.f_oneway."""
    var sp = _load_scipy_stats()
    if sp is None:
        print("⊘ test_f_oneway_scipy skipped (scipy not available)")
        return

    var g1: List[Float64] = [3.0, 4.0, 5.0, 6.0]
    var g2: List[Float64] = [5.0, 6.0, 7.0, 8.0]
    var g3: List[Float64] = [7.0, 8.0, 9.0, 10.0]

    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)

    var py_g1 = Python.evaluate("[3., 4., 5., 6.]")
    var py_g2 = Python.evaluate("[5., 6., 7., 8.]")
    var py_g3 = Python.evaluate("[7., 8., 9., 10.]")
    var sp_result = sp.f_oneway(py_g1, py_g2, py_g3)
    var sp_f = _py_f64(sp_result[0])
    var sp_p = _py_f64(sp_result[1])

    var result = f_oneway(groups)
    assert_almost_equal(result[0], sp_f, atol=1e-4)
    assert_almost_equal(result[1], sp_p, atol=1e-3)

    print("✓ test_f_oneway_scipy passed")


# ===----------------------------------------------------------------------=== #
# Main test runner
# ===----------------------------------------------------------------------=== #


fn main() raises:
    print("=== StaMojo: Testing hypothesis tests & correlation ===")
    print()

    # t-tests
    test_ttest_1samp_basic()
    test_ttest_1samp_no_effect()
    test_ttest_1samp_scipy()
    test_ttest_ind_welch()
    test_ttest_ind_scipy()
    test_ttest_rel()
    print()

    # Chi-squared tests
    test_chi2_gof_fair_die()
    test_chi2_ind_basic()
    test_chi2_ind_scipy()
    print()

    # KS test
    test_ks_normal_data()
    test_ks_uniform_data()
    print()

    # Correlation
    test_pearsonr_perfect()
    test_pearsonr_negative()
    test_pearsonr_scipy()
    test_spearmanr_perfect_monotone()
    test_spearmanr_scipy()
    test_kendalltau_concordant()
    test_kendalltau_discordant()
    print()

    # ANOVA
    test_f_oneway_identical()
    test_f_oneway_different()
    test_f_oneway_scipy()

    print()
    print("=== All hypothesis test & correlation tests passed ===")
