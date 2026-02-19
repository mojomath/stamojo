# ===----------------------------------------------------------------------=== #
# Stamojo - Stats - Correlation coefficients
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Correlation coefficients with p-values.

Provides:
- ``pearsonr(x, y)``  — Pearson product-moment correlation coefficient
- ``spearmanr(x, y)`` — Spearman rank-order correlation coefficient
- ``kendalltau(x, y)``— Kendall's tau-b rank correlation coefficient

Each function returns a ``Tuple[Float64, Float64]`` of (correlation, p-value).
The two-sided p-value tests the null hypothesis that the true correlation is
zero.
"""

from math import sqrt, nan

from stamojo.distributions import Normal, StudentT
from stamojo.stats.descriptive import mean


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _rank_data(data: List[Float64]) -> List[Float64]:
    """Assign ranks to data, handling ties by averaging.

    Returns a list of ranks (1-based) with the same length as *data*.
    """
    var n = len(data)

    # Build (value, original_index) pairs and sort by value (insertion sort).
    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)

    # Sort indices by data[index] (insertion sort to keep it simple).
    for i in range(1, n):
        var key_idx = indices[i]
        var key_val = data[key_idx]
        var j = i - 1
        while j >= 0 and data[indices[j]] > key_val:
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = key_idx

    # Assign ranks with tie averaging.
    var ranks = List[Float64](capacity=n)
    for _ in range(n):
        ranks.append(0.0)

    var i = 0
    while i < n:
        var j = i
        # Find the end of the tie group.
        while j < n - 1 and data[indices[j + 1]] == data[indices[j]]:
            j += 1
        # Average rank for this tie group.
        var avg_rank = Float64(i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indices[k]] = avg_rank
        i = j + 1

    return ranks^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn pearsonr(
    x: List[Float64], y: List[Float64]
) -> Tuple[Float64, Float64]:
    """Pearson product-moment correlation coefficient and p-value.

    The p-value is two-sided and tests H₀: ρ = 0 using the t-distribution
    with n − 2 degrees of freedom.

    Args:
        x: First data vector.
        y: Second data vector.  Must have the same length as *x*.

    Returns:
        A tuple (r, p-value).
    """
    var n = len(x)
    if n != len(y) or n < 3:
        return (nan[DType.float64](), nan[DType.float64]())

    var mx = mean(x)
    var my = mean(y)

    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for i in range(n):
        var dx = x[i] - mx
        var dy = y[i] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy

    if sxx == 0.0 or syy == 0.0:
        return (nan[DType.float64](), nan[DType.float64]())

    var r = sxy / sqrt(sxx * syy)

    # Clamp to [-1, 1] for numerical safety.
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0

    # t-statistic: t = r * sqrt((n-2) / (1-r²))
    var df = Float64(n - 2)
    var t_stat = r * sqrt(df / (1.0 - r * r + 1.0e-300))
    var tdist = StudentT(df)
    var p_value = 2.0 * tdist.sf(abs(t_stat))

    return (r, p_value)


fn spearmanr(
    x: List[Float64], y: List[Float64]
) -> Tuple[Float64, Float64]:
    """Spearman rank-order correlation coefficient and p-value.

    Computes the Pearson correlation of the rank-transformed data.
    The p-value is two-sided using the t-distribution approximation.

    Args:
        x: First data vector.
        y: Second data vector.  Must have the same length as *x*.

    Returns:
        A tuple (rho, p-value).
    """
    var n = len(x)
    if n != len(y) or n < 3:
        return (nan[DType.float64](), nan[DType.float64]())

    var rx = _rank_data(x)
    var ry = _rank_data(y)

    return pearsonr(rx, ry)


fn kendalltau(
    x: List[Float64], y: List[Float64]
) -> Tuple[Float64, Float64]:
    """Kendall's tau-b rank correlation coefficient and p-value.

    Tau-b adjusts for ties. The p-value is two-sided based on the
    normal approximation for large samples.

    Args:
        x: First data vector.
        y: Second data vector.  Must have the same length as *x*.

    Returns:
        A tuple (tau, p-value).
    """
    var n = len(x)
    if n != len(y) or n < 3:
        return (nan[DType.float64](), nan[DType.float64]())

    var concordant = 0
    var discordant = 0
    var ties_x = 0
    var ties_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            var dx = x[i] - x[j]
            var dy = y[i] - y[j]

            if dx > 0.0 and dy > 0.0:
                concordant += 1
            elif dx < 0.0 and dy < 0.0:
                concordant += 1
            elif dx > 0.0 and dy < 0.0:
                discordant += 1
            elif dx < 0.0 and dy > 0.0:
                discordant += 1
            else:
                if dx == 0.0:
                    ties_x += 1
                if dy == 0.0:
                    ties_y += 1

    var n_pairs = n * (n - 1) // 2
    var denom = sqrt(
        Float64(n_pairs - ties_x) * Float64(n_pairs - ties_y)
    )

    if denom == 0.0:
        return (nan[DType.float64](), nan[DType.float64]())

    var tau = Float64(concordant - discordant) / denom

    # Normal approximation for p-value.
    # Var(S) ≈ n(n-1)(2n+5)/18 for no ties, but we use the simpler formula.
    var fn_ = Float64(n)
    var var_s = fn_ * (fn_ - 1.0) * (2.0 * fn_ + 5.0) / 18.0
    var z = Float64(concordant - discordant) / sqrt(var_s)
    var normal = Normal(0.0, 1.0)
    var p_value = 2.0 * normal.sf(abs(z))

    return (tau, p_value)
