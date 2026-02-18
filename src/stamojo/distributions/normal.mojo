# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Normal distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Normal (Gaussian) distribution.

Provides the `Normal` distribution struct with PDF, log-PDF, CDF, survival
function, percent-point function (PPF / quantile), and random variate
generation.

Examples::

    from stamojo.distributions import Normal

    var n = Normal(0.0, 1.0)         # standard normal
    n.pdf(0.0)                       # ≈ 0.3989
    n.cdf(1.96)                      # ≈ 0.975
    n.ppf(0.975)                     # ≈ 1.96
"""

from math import sqrt, log, cos, exp, erf, erfc, nan, inf
from random import random_float64

from stamojo.special import ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _INV_SQRT_2PI = 0.3989422804014326779399460599343819
comptime _LN_SQRT_2PI = 0.9189385332046727417803297364056176
comptime _SQRT2 = 1.4142135623730950488016887242096981
comptime _2PI = 6.283185307179586476925286766559006


# ===----------------------------------------------------------------------=== #
# Normal distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Normal(Copyable, Movable):
    """Normal (Gaussian) distribution.

    The normal distribution with mean `mu` and standard deviation `sigma`
    has probability density function:

        f(x) = (1 / (σ√(2π))) exp(-(x - μ)² / (2σ²))

    Fields:
        mu: Mean (location parameter).
        sigma: Standard deviation (scale parameter). Must be positive.
    """

    var mu: Float64
    var sigma: Float64

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        var z = (x - self.mu) / self.sigma
        return _INV_SQRT_2PI / self.sigma * exp(-0.5 * z * z)

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        var z = (x - self.mu) / self.sigma
        return -_LN_SQRT_2PI - log(self.sigma) - 0.5 * z * z

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x)."""
        return 0.5 * erfc(-(x - self.mu) / (self.sigma * _SQRT2))

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        return 0.5 * erfc((x - self.mu) / (self.sigma * _SQRT2))

    fn ppf(self, p: Float64) -> Float64:
        """Percent-point function (quantile / inverse CDF).

        Returns the value *x* such that P(X ≤ x) = p.

        Args:
            p: Probability value in [0, 1].

        Returns:
            The quantile corresponding to *p*.
        """
        if p < 0.0 or p > 1.0:
            return nan[DType.float64]()
        if p == 0.0:
            return -inf[DType.float64]()
        if p == 1.0:
            return inf[DType.float64]()
        return self.mu + self.sigma * ndtri(p)

    # --- Summary statistics --------------------------------------------------

    fn mean(self) -> Float64:
        """Distribution mean."""
        return self.mu

    fn variance(self) -> Float64:
        """Distribution variance σ²."""
        return self.sigma * self.sigma

    fn std(self) -> Float64:
        """Distribution standard deviation σ."""
        return self.sigma

    fn entropy(self) -> Float64:
        """Differential entropy of the distribution."""
        # H = 0.5 * ln(2πeσ²) = ln(σ√(2πe)) = ln(σ) + 0.5*ln(2π) + 0.5
        return _LN_SQRT_2PI + log(self.sigma) + 0.5

    # --- Random variate generation -------------------------------------------

    fn rvs(self) -> Float64:
        """Generate a single random variate (Box-Muller transform)."""
        var u1 = random_float64()
        while u1 == 0.0:
            u1 = random_float64()
        var u2 = random_float64()
        var z = sqrt(-2.0 * log(u1)) * cos(_2PI * u2)
        return self.mu + self.sigma * z

    fn rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        for _ in range(n):
            var u1 = random_float64()
            while u1 == 0.0:
                u1 = random_float64()
            var u2 = random_float64()
            var z = sqrt(-2.0 * log(u1)) * cos(_2PI * u2)
            result.append(self.mu + self.sigma * z)
        return result^
