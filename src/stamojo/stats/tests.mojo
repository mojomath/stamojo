# ===----------------------------------------------------------------------=== #
# Stamojo - Stats - Hypothesis tests
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Hypothesis testing functions.

Provides:
- ``ttest_1samp(data, mu0)``       — One-sample t-test
- ``ttest_ind(x, y)``             — Independent two-sample t-test (Welch's)
- ``ttest_rel(x, y)``             — Paired (related) samples t-test
- ``chi2_gof(observed, expected)`` — Chi-squared goodness-of-fit test
- ``chi2_ind(observed)``           — Chi-squared test of independence
- ``ks_1samp(data)``              — One-sample Kolmogorov-Smirnov test (vs N(0,1))
- ``f_oneway(groups)``            — One-way ANOVA F-test

Each function returns a ``Tuple[Float64, Float64]`` of (test statistic, p-value)
unless otherwise noted.
"""

from math import sqrt, nan, inf

from stamojo.distributions import Normal, StudentT, ChiSquared, FDist
from stamojo.stats.descriptive import mean, variance


# ===----------------------------------------------------------------------=== #
# Helper: sorted copy (for KS test)
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
# t-tests
# ===----------------------------------------------------------------------=== #


fn ttest_1samp(
    data: List[Float64], mu0: Float64 = 0.0
) -> Tuple[Float64, Float64]:
    """One-sample t-test.

    Tests H₀: μ = mu0 against H₁: μ ≠ mu0 (two-sided).

    Args:
        data: Sample data.
        mu0: Hypothesized population mean.  Default is 0.

    Returns:
        A tuple (t-statistic, p-value).
    """
    var n = len(data)
    if n < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    var m = mean(data)
    var s = sqrt(variance(data, ddof=1))

    if s == 0.0:
        if m == mu0:
            return (0.0, 1.0)
        else:
            return (inf[DType.float64](), 0.0)

    var t = (m - mu0) / (s / sqrt(Float64(n)))
    var df = Float64(n - 1)
    var tdist = StudentT(df)
    var p = 2.0 * tdist.sf(abs(t))

    return (t, p)


fn ttest_ind(
    x: List[Float64],
    y: List[Float64],
    equal_var: Bool = False,
) -> Tuple[Float64, Float64]:
    """Independent two-sample t-test (Welch's by default).

    Tests H₀: μ₁ = μ₂ against H₁: μ₁ ≠ μ₂ (two-sided).

    When `equal_var` is False (default), performs Welch's t-test which
    does not assume equal population variances.

    Args:
        x: First sample.
        y: Second sample.
        equal_var: If True, assume equal variances (pooled t-test).
                   If False (default), use Welch's t-test.

    Returns:
        A tuple (t-statistic, p-value).
    """
    var n1 = len(x)
    var n2 = len(y)
    if n1 < 2 or n2 < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    var m1 = mean(x)
    var m2 = mean(y)
    var v1 = variance(x, ddof=1)
    var v2 = variance(y, ddof=1)
    var fn1 = Float64(n1)
    var fn2 = Float64(n2)

    var t: Float64
    var df: Float64

    if equal_var:
        # Pooled variance t-test.
        var sp2 = ((fn1 - 1.0) * v1 + (fn2 - 1.0) * v2) / (fn1 + fn2 - 2.0)
        var se = sqrt(sp2 * (1.0 / fn1 + 1.0 / fn2))
        if se == 0.0:
            return (0.0, 1.0) if m1 == m2 else (
                inf[DType.float64](),
                0.0,
            )
        t = (m1 - m2) / se
        df = fn1 + fn2 - 2.0
    else:
        # Welch's t-test.
        var se2 = v1 / fn1 + v2 / fn2
        var se = sqrt(se2)
        if se == 0.0:
            return (0.0, 1.0) if m1 == m2 else (
                inf[DType.float64](),
                0.0,
            )
        t = (m1 - m2) / se
        # Welch-Satterthwaite degrees of freedom.
        var num = se2 * se2
        var denom = (
            (v1 / fn1) * (v1 / fn1) / (fn1 - 1.0)
            + (v2 / fn2) * (v2 / fn2) / (fn2 - 1.0)
        )
        df = num / denom

    var tdist = StudentT(df)
    var p = 2.0 * tdist.sf(abs(t))

    return (t, p)


fn ttest_rel(
    x: List[Float64], y: List[Float64]
) -> Tuple[Float64, Float64]:
    """Paired (related) samples t-test.

    Tests H₀: μ_d = 0 against H₁: μ_d ≠ 0, where d = x − y.

    Args:
        x: First sample.
        y: Second sample.  Must have the same length as *x*.

    Returns:
        A tuple (t-statistic, p-value).
    """
    var n = len(x)
    if n != len(y) or n < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    var diffs = List[Float64](capacity=n)
    for i in range(n):
        diffs.append(x[i] - y[i])

    return ttest_1samp(diffs, 0.0)


# ===----------------------------------------------------------------------=== #
# Chi-squared tests
# ===----------------------------------------------------------------------=== #


fn chi2_gof(
    observed: List[Float64], expected: List[Float64]
) -> Tuple[Float64, Float64]:
    """Chi-squared goodness-of-fit test.

    Tests H₀: the data follow the expected distribution.

    Args:
        observed: Observed frequencies.
        expected: Expected frequencies.  Must have the same length.

    Returns:
        A tuple (chi2 statistic, p-value).
    """
    var k = len(observed)
    if k != len(expected) or k < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    var chi2 = 0.0
    for i in range(k):
        if expected[i] <= 0.0:
            return (nan[DType.float64](), nan[DType.float64]())
        var diff = observed[i] - expected[i]
        chi2 += diff * diff / expected[i]

    var df = Float64(k - 1)
    var chi2dist = ChiSquared(df)
    var p = chi2dist.sf(chi2)

    return (chi2, p)


fn chi2_ind(
    observed: List[List[Float64]],
) -> Tuple[Float64, Float64]:
    """Chi-squared test of independence for a contingency table.

    Tests H₀: the row and column variables are independent.

    Args:
        observed: A 2D contingency table as a list of rows.
                  Each row must have the same number of columns.

    Returns:
        A tuple (chi2 statistic, p-value).
    """
    var nrows = len(observed)
    if nrows < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    var ncols = len(observed[0])
    if ncols < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    # Compute row sums, column sums, and grand total.
    var row_sums = List[Float64](capacity=nrows)
    var col_sums = List[Float64](capacity=ncols)
    for _ in range(ncols):
        col_sums.append(0.0)

    var total = 0.0
    for i in range(nrows):
        if len(observed[i]) != ncols:
            return (nan[DType.float64](), nan[DType.float64]())
        var rs = 0.0
        for j in range(ncols):
            rs += observed[i][j]
            col_sums[j] += observed[i][j]
        row_sums.append(rs)
        total += rs

    if total == 0.0:
        return (nan[DType.float64](), nan[DType.float64]())

    # Compute chi-squared statistic.
    var chi2 = 0.0
    for i in range(nrows):
        for j in range(ncols):
            var expected = row_sums[i] * col_sums[j] / total
            if expected <= 0.0:
                return (nan[DType.float64](), nan[DType.float64]())
            var diff = observed[i][j] - expected
            chi2 += diff * diff / expected

    var df = Float64((nrows - 1) * (ncols - 1))
    var chi2dist = ChiSquared(df)
    var p = chi2dist.sf(chi2)

    return (chi2, p)


# ===----------------------------------------------------------------------=== #
# Kolmogorov-Smirnov test
# ===----------------------------------------------------------------------=== #


fn ks_1samp(data: List[Float64]) -> Tuple[Float64, Float64]:
    """One-sample Kolmogorov-Smirnov test against the standard normal N(0,1).

    Tests H₀: the data come from a standard normal distribution.

    The p-value is computed using the asymptotic Kolmogorov distribution::

        P(D > d) ≈ 2 Σ_{k=1}^{∞} (-1)^{k+1} exp(-2k²n d²)

    Args:
        data: Sample data.

    Returns:
        A tuple (D-statistic, p-value).
    """
    var n = len(data)
    if n < 1:
        return (nan[DType.float64](), nan[DType.float64]())

    var sorted_data = _sorted_copy(data)
    var normal = Normal(0.0, 1.0)
    var fn_ = Float64(n)

    var d_max = 0.0
    for i in range(n):
        var cdf_val = normal.cdf(sorted_data[i])
        # D+ = max(i/n - F(x_i))
        var d_plus = Float64(i + 1) / fn_ - cdf_val
        # D- = max(F(x_i) - (i-1)/n)
        var d_minus = cdf_val - Float64(i) / fn_
        if d_plus > d_max:
            d_max = d_plus
        if d_minus > d_max:
            d_max = d_minus

    # Asymptotic p-value using Kolmogorov distribution.
    var p = _ks_pvalue(d_max, n)

    return (d_max, p)


fn _ks_pvalue(d: Float64, n: Int) -> Float64:
    """Compute the two-sided KS test p-value using the asymptotic formula.

    P(D_n > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2 k² (√n d)²)
    """
    if d <= 0.0:
        return 1.0
    if d >= 1.0:
        return 0.0

    var z = (sqrt(Float64(n)) + 0.12 + 0.11 / sqrt(Float64(n))) * d
    var z2 = z * z

    # Sum the series.
    var p = 0.0
    for k in range(1, 101):
        var fk = Float64(k)
        # Use exp on the exponent.
        var exp_term = (-1.0) ** (fk + 1.0)
        var exponent = -2.0 * fk * fk * z2
        if exponent < -700.0:
            break  # Negligible contribution.
        p += exp_term * _exp_safe(exponent)

    p *= 2.0

    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


fn _exp_safe(x: Float64) -> Float64:
    """Safe exponential that avoids underflow."""
    if x < -700.0:
        return 0.0
    from math import exp

    return exp(x)


# ===----------------------------------------------------------------------=== #
# One-way ANOVA
# ===----------------------------------------------------------------------=== #


fn f_oneway(
    groups: List[List[Float64]],
) -> Tuple[Float64, Float64]:
    """One-way ANOVA F-test.

    Tests H₀: all group means are equal against H₁: at least one differs.

    Args:
        groups: A list of groups, where each group is a list of observations.

    Returns:
        A tuple (F-statistic, p-value).
    """
    var k = len(groups)
    if k < 2:
        return (nan[DType.float64](), nan[DType.float64]())

    # Compute group means, sizes, and overall mean.
    var n_total = 0
    var grand_sum = 0.0

    var group_means = List[Float64](capacity=k)
    var group_sizes = List[Int](capacity=k)

    for i in range(k):
        var ni = len(groups[i])
        if ni < 1:
            return (nan[DType.float64](), nan[DType.float64]())
        group_sizes.append(ni)
        n_total += ni
        var s = 0.0
        for j in range(ni):
            s += groups[i][j]
        grand_sum += s
        group_means.append(s / Float64(ni))

    if n_total <= k:
        return (nan[DType.float64](), nan[DType.float64]())

    var grand_mean = grand_sum / Float64(n_total)

    # Between-group sum of squares.
    var ss_between = 0.0
    for i in range(k):
        var diff = group_means[i] - grand_mean
        ss_between += Float64(group_sizes[i]) * diff * diff

    # Within-group sum of squares.
    var ss_within = 0.0
    for i in range(k):
        for j in range(group_sizes[i]):
            var diff = groups[i][j] - group_means[i]
            ss_within += diff * diff

    var df_between = Float64(k - 1)
    var df_within = Float64(n_total - k)

    if df_within <= 0.0 or ss_within == 0.0:
        return (inf[DType.float64](), 0.0)

    var ms_between = ss_between / df_between
    var ms_within = ss_within / df_within
    var f_stat = ms_between / ms_within

    var fdist = FDist(df_between, df_within)
    var p = fdist.sf(f_stat)

    return (f_stat, p)
