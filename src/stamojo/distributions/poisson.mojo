# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Poisson distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Poisson distribution.

Provides the `Poisson` distribution struct with PMF, log-PMF, CDF,
survival function, and percent-point function (PPF / quantile).

The Poisson distribution with rate parameter *μ* has PMF::

    P(X = k; μ) = μ^k exp(−μ) / k!,  k = 0, 1, 2, ...
"""

from std.math import log, exp, lgamma, nan, inf, floor, sqrt

from stamojo.distributions.traits import DiscretelyDistributed
from stamojo.special import gammaincc


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _MAX_K = 10000
"""Maximum k value for PPF search to prevent infinite loops in extreme cases."""


# ===----------------------------------------------------------------------=== #
# Poisson distribution
# ===----------------------------------------------------------------------=== #


struct Poisson(DiscretelyDistributed):
    """Poisson distribution with rate parameter `mu`.

    Represents the Poisson distribution, a discrete probability distribution
    that expresses the probability of a given number of events occurring in a
    fixed interval of time or space, if these events occur with a known constant
    mean rate and independently of the time since the last event.

    The probability mass function (PMF) for the Poisson distribution is:

        P(X = k; μ) = μ^k * exp(-μ) / k!

    Fields:
        mu: Rate parameter (mean number of events). Must be positive.
    """

    var mu: Float64
    """Rate parameter (mean number of events). Must be positive."""

    def __init__(out self, mu: Float64):
        """Constructs a Poisson distribution with the given rate.

        Args:
            mu: Rate parameter (mean number of events). Must be positive.
        """
        self.mu = mu

    # --- Probability functions ------------------------------------------------

    def pmf(self, k: Int) -> Float64:
        """Computes the probability mass function at *k*.

        Args:
            k: Number of events (must be >= 0).

        Returns:
            PMF value at *k*. Returns 0.0 for k < 0.
        """
        return exp(self.logpmf(k))

    def logpmf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the PMF at *k*.

        Args:
            k: Number of events (must be >= 0).

        Returns:
            Log-PMF value at *k*. Returns -∞ for k < 0.
        """
        if k < 0:
            return -inf[DType.float64]()
        if self.mu == 0.0:
            return 0.0 if k == 0 else -inf[DType.float64]()
        return Float64(k) * log(self.mu) - self.mu - lgamma(Float64(k) + 1.0)

    def cdf(self, k: Int) -> Float64:
        """Computes the cumulative distribution function P(X ≤ k).

        Args:
            k: Number of events.

        Returns:
            CDF value at *k*. Returns 1.0 for large k.
        """
        if k < 0:
            return 0.0
        if self.mu == 0.0:
            return 1.0
        # CDF of Poisson(μ) at k = Q(k+1, μ) = gammaincc(k+1, μ)
        return gammaincc(Float64(k + 1), self.mu)

    def logcdf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the CDF at *k*.

        Args:
            k: Number of events.

        Returns:
            Log-CDF value at *k*.
        """
        var c = self.cdf(k)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    def sf(self, k: Int) -> Float64:
        """Computes the survival function (1 − CDF) at *k*.

        Args:
            k: Number of events.

        Returns:
            Survival function value at *k*.
        """
        if k < 0:
            return 1.0
        if self.mu == 0.0:
            return 0.0
        return 1.0 - self.cdf(k)

    def logsf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the survival function at *k*.

        Args:
            k: Number of events.

        Returns:
            Log-survival function value at *k*.
        """
        var s = self.sf(k)
        if s <= 0.0:
            return -inf[DType.float64]()
        return log(s)

    def ppf(self, q: Float64) -> Int:
        """Finds the smallest integer k such that CDF(k) ≥ q
        (percent point function).

        Uses binary search with the CDF to avoid numerical underflow
        that would occur with incremental PMF summation for large mu.

        Args:
            q: Probability in [0, 1].

        Returns:
            The smallest integer k such that CDF(k) ≥ q.
        """
        if q <= 0.0:
            return 0
        if q >= 1.0:
            return _MAX_K

        # Binary search: find smallest k where CDF(k) >= q.
        var lo = 0
        var hi = Int(self.mu + 10.0 * sqrt(self.mu) + 20.0)
        if hi > _MAX_K:
            hi = _MAX_K
        # Ensure hi is large enough.
        while self.cdf(hi) < q:
            hi *= 2
            if hi > _MAX_K:
                return _MAX_K

        while lo < hi:
            var mid = (lo + hi) // 2
            if self.cdf(mid) < q:
                lo = mid + 1
            else:
                hi = mid

        return lo

    def isf(self, q: Float64) -> Int:
        """Computes the inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The smallest integer k such that SF(k) ≤ q.
        """
        return self.ppf(1.0 - q)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> UInt:
        """Computes the median of the distribution (approximation).

        Uses the approximation: floor(μ + 1/3 - 0.02/μ).

        Returns:
            The median of the distribution.
        """
        if self.mu == 0.0:
            return 0
        return UInt(floor(self.mu + 1.0 / 3.0 - 0.02 / self.mu))

    def mean(self) -> Float64:
        """Computes the distribution mean = μ.

        Returns:
            The mean of the distribution.
        """
        return self.mu

    def variance(self) -> Float64:
        """Computes the distribution variance = μ.

        Returns:
            The variance of the distribution.
        """
        return self.mu

    def std(self) -> Float64:
        """Computes the distribution standard deviation = √μ.

        Returns:
            The standard deviation of the distribution.
        """
        return sqrt(self.mu)
