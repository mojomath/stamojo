# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Binomial distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Binomial distribution.

Provides the `Binomial` distribution struct with PMF, log-PMF, CDF,
survival function, and percent-point function (PPF / quantile).

The binomial distribution with parameters n and p has PMF:

    P(X = k) = C(n, k) * p^k * (1-p)^(n-k),  k = 0, 1, ..., n

where C(n, k) is the binomial coefficient.
"""

from math import log, log1p, exp, lgamma, nan, inf, floor, sqrt

from stamojo.distributions.traits import DiscretelyDistributed


# `DiscretelyDistributed` trait contains `Copyable` and `Movable` traits
struct Binomial(DiscretelyDistributed):
    """Binomial distribution.

    Represents the binomial distribution, a discrete probability distribution
    that models the number of successes in a fixed number of independent
    Bernoulli trials, each with the same probability of success.

    The probability mass function (PMF) for the binomial distribution is:

        P(X = k) = C(n, k) * p^k * (1-p)^(n-k)

    where C(n, k) is the binomial coefficient.
    """

    var n: UInt
    """Number of trials (must be >= 0)."""

    var p: Float64
    """Probability of success in each trial (must be in [0, 1])."""

    # --- Initialization -------------------------------------------------------

    fn __init__(out self, n: UInt, p: Float64):
        self.n = n
        self.p = p

    # --- Probability functions ------------------------------------------------

    fn pmf(self, k: Int) -> Float64:
        """Probability mass function at *k*.

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            Returns 0.0 if k > n.
        """
        return exp(self.logpmf(k))

    fn logpmf(self, k: Int) -> Float64:
        """Natural logarithm of the PMF at *k*.

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            Returns -∞ if k > n.
        """
        if k > Int(self.n):
            return -inf[DType.float64]()

        if self.p == 0.0:
            return 0.0 if k == 0 else -inf[DType.float64]()

        if self.p == 1.0:
            return 0.0 if k == Int(self.n) else -inf[DType.float64]()

        var kf = Float64(k)
        var nf = Float64(self.n)
        var logc = _log_binomial_coefficient(self.n, k)
        return logc + kf * log(self.p) + (nf - kf) * log1p(-self.p)

    fn cdf(self, k: Int) -> Float64:
        """Cumulative distribution function P(X ≤ k).

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            CDF value at *k*. Returns 1.0 for k >= n.
        """
        if k >= Int(self.n):
            return 1.0

        if self.p == 0.0:
            return 1.0
        if self.p == 1.0:
            return 0.0 if k < Int(self.n) else 1.0

        var nf = Float64(self.n)
        var q = 1.0 - self.p

        var pmf_k = exp(nf * log(q))  # k = 0
        var total = pmf_k

        for i in range(0, k):
            pmf_k *= (nf - Float64(i)) / Float64(i + 1) * (self.p / q)
            total += pmf_k

        return total

    fn logcdf(self, k: Int) -> Float64:
        """Natural logarithm of the CDF at *k*.

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            Log-CDF value at *k*. Returns 0.0 for k >= n.
        """
        var c = self.cdf(k)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    fn sf(self, k: Int) -> Float64:
        """Survival function (1 − CDF) at *k*.

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            Survival function value at *k*. Returns 0.0 for k >= n.
        """
        return 1.0 - self.cdf(k)

    fn logsf(self, k: Int) -> Float64:
        """Natural logarithm of the survival function at *k*.

        Args:
            k: Number of successes (must be in [0, n]).

        Returns:
            Log-survival function value at *k*.
        """
        if self.p < 0.0 or self.p > 1.0:
            return nan[DType.float64]()
        return log1p(-self.cdf(k))

    fn ppf(self, q: Float64) -> Int:
        """Percent point function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            Smallest integer k such that CDF(k) ≥ q. Returns 0 for q=0, n for q=1.
        """
        if q == 0.0:
            return 0
        if q == 1.0:
            return Int(self.n)

        var cumulative: Float64 = 0.0
        for k in range(self.n + 1):
            cumulative += self.pmf(Int(k))
            if cumulative >= q:
                return Int(k)

        return Int(self.n)

    fn isf(self, q: Float64) -> Int:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            Smallest integer k such that SF(k) ≤ q. Returns n for q=0, 0 for q=1.
        """
        if q == 0.0:
            return Int(self.n)
        if q == 1.0:
            return 0

        var cumulative = 0.0
        for k in range(self.n + 1):
            cumulative += self.pmf(Int(k))
            if 1.0 - cumulative <= q:
                return Int(k)

        return Int(self.n)

    # --- Summary statistics --------------------------------------------------
    fn median(self) -> UInt:
        """Median of the distribution.

        Returns:
            The median of the distribution.
        """
        return UInt(floor(Float64(self.n) * self.p + 0.5))

    fn mean(self) -> Float64:
        """Distribution mean: n * p.

        Returns:
            The mean of the distribution.
        """
        return Float64(self.n) * self.p

    fn variance(self) -> Float64:
        """Distribution variance: n * p * (1 - p).

        Returns:
            The variance of the distribution.
        """
        var np = Float64(self.n) * self.p
        return np * (1.0 - self.p)

    fn std(self) -> Float64:
        """Distribution standard deviation.

        Returns:
            The standard deviation of the distribution.
        """
        return sqrt(self.variance())


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


fn _log_binomial_coefficient(n: UInt, k: Int) -> Float64:
    """Log of the binomial coefficient C(n, k).

    Args:
        n: Number of trials.
        k: Number of successes.

    Returns:
        log(C(n, k)).
    """
    var nf = Float64(n)
    var kf = Float64(k)
    var fnk = Float64(n - k)
    return lgamma(nf + 1.0) - lgamma(kf + 1.0) - lgamma(fnk + 1.0)
