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

from math import log, exp, lgamma, nan, inf, floor, sqrt

from stamojo.distributions.traits import DiscretelyDistributed
from stamojo.special import gammaincc


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _MAX_K = 10000


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

    def __init__(out self, mu: Float64):
        self.mu = mu

    # --- Probability functions ------------------------------------------------

    def pmf(self, k: Int) -> Float64:
        """Probability mass function at *k*.

        Args:
            k: Number of events (must be >= 0).

        Returns:
            PMF value at *k*. Returns 0.0 for k < 0.
        """
        return exp(self.logpmf(k))

    def logpmf(self, k: Int) -> Float64:
        """Natural logarithm of the PMF at *k*.

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
        """Cumulative distribution function P(X ≤ k).

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
        """Natural logarithm of the CDF at *k*.

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
        """Survival function (1 − CDF) at *k*.

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
        """Natural logarithm of the survival function at *k*.

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
        """Percent point function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            Smallest integer k such that CDF(k) ≥ q.
        """
        if q <= 0.0:
            return 0
        if q >= 1.0:
            return _MAX_K

        var pk = exp(-self.mu)
        var cumulative: Float64 = pk

        if cumulative >= q:
            return 0

        for k in range(1, _MAX_K):
            pk *= self.mu / Float64(k)
            cumulative += pk
            if cumulative >= q:
                return k

        return _MAX_K

    def isf(self, q: Float64) -> Int:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            Smallest integer k such that SF(k) ≤ q.
        """
        if q <= 0.0:
            return _MAX_K
        if q >= 1.0:
            return 0

        var pk: Float64 = exp(-self.mu)
        var cumulative: Float64 = pk

        if 1.0 - cumulative <= q:
            return 0

        for k in range(1, _MAX_K):
            pk *= self.mu / Float64(k)
            cumulative += pk
            if 1.0 - cumulative <= q:
                return k

        return _MAX_K

    # --- Summary statistics --------------------------------------------------

    def median(self) -> UInt:
        """Median of the distribution (approximation).

        Uses the approximation: floor(μ + 1/3 - 0.02/μ).
        """
        if self.mu == 0.0:
            return 0
        return UInt(floor(self.mu + 1.0 / 3.0 - 0.02 / self.mu))

    def mean(self) -> Float64:
        """Distribution mean = μ."""
        return self.mu

    def variance(self) -> Float64:
        """Distribution variance = μ."""
        return self.mu

    def std(self) -> Float64:
        """Distribution standard deviation = √μ."""
        return sqrt(self.mu)
