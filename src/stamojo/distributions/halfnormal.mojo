# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Half-normal distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Half-normal distribution.

Provides the `HalfNormal` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The half-normal distribution with location *μ* and scale *σ* has PDF::

    f(x; μ, σ) = √(2/π) / σ · exp(−(x−μ)² / (2σ²)),  x ≥ μ
"""

from std.math import sqrt, log, exp, erf, erfc, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.special import ndtri
from stamojo.distributions._constants import (
    PI as _PI,
    SQRT2 as _SQRT2,
    SQRT_2_OVER_PI as _SQRT_2_OVER_PI,
)


# ===----------------------------------------------------------------------=== #
# Half-normal distribution
# ===----------------------------------------------------------------------=== #


struct HalfNormal(ContinuouslyDistributed):
    """Half-normal distribution with location `loc` and scale `scale`.

    The half-normal distribution is the distribution of |X| where
    X ~ Normal(loc, scale). It is supported on [loc, ∞).

    The probability density function (PDF) for the half-normal distribution is:

        f(x; μ, σ) = √(2/π) / σ * exp(-(x-μ)² / (2σ²)),  x ≥ μ

    Fields:
        loc: Location parameter (μ), the lower bound of support. Defaults to 0.0.
        scale: Scale parameter (σ). Must be positive. Defaults to 1.0.
    """

    var loc: Float64
    var scale: Float64

    def __init__(out self, loc: Float64 = 0.0, scale: Float64 = 1.0):
        self.loc = loc
        self.scale = scale

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*. Returns 0.0 for x < loc.
        """
        if x < self.loc:
            return 0.0
        var z = (x - self.loc) / self.scale
        return _SQRT_2_OVER_PI / self.scale * exp(-0.5 * z * z)

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x < loc.
        """
        if x < self.loc:
            return -inf[DType.float64]()
        var z = (x - self.loc) / self.scale
        return log(_SQRT_2_OVER_PI) - log(self.scale) - 0.5 * z * z

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*. Returns 0.0 for x < loc.
        """
        if x < self.loc:
            return 0.0
        var z = (x - self.loc) / self.scale
        return erf(z / _SQRT2)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*. Returns -∞ for x < loc.
        """
        if x < self.loc:
            return -inf[DType.float64]()
        var c = self.cdf(x)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*. Returns 1.0 for x < loc.
        """
        if x < self.loc:
            return 1.0
        var z = (x - self.loc) / self.scale
        return erfc(z / _SQRT2)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        if x < self.loc:
            return 0.0
        var s = self.sf(x)
        if s <= 0.0:
            return -inf[DType.float64]()
        return log(s)

    def ppf(self, q: Float64) -> Float64:
        """Percent-point (quantile) function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 1.0:
            return inf[DType.float64]()
        # X = loc + sigma * |Z| where CDF(z) = 2*Phi(z) - 1 = q for z >= 0.
        # => Phi(z) = (1+q)/2 => z = ndtri((1+q)/2).
        return self.loc + self.scale * ndtri((1.0 + q) / 2.0)

    def isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The value *x* such that SF(x) = *q*.
        """
        return self.ppf(1.0 - q)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution."""
        return self.ppf(0.5)

    def mean(self) -> Float64:
        """Distribution mean: loc + σ√(2/π)."""
        return self.loc + self.scale * _SQRT_2_OVER_PI

    def variance(self) -> Float64:
        """Distribution variance: σ² (1 − 2/π)."""
        return self.scale * self.scale * (1.0 - 2.0 / _PI)

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: 0.5·ln(π σ²/2) + 0.5."""
        return 0.5 * log(_PI * self.scale * self.scale / 2.0) + 0.5

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0 or u == 1.0:
            u = random_float64()
        return self.loc + self.scale * ndtri((1.0 + u) / 2.0)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        for _ in range(n):
            var u = random_float64()
            while u == 0.0 or u == 1.0:
                u = random_float64()
            result.append(self.loc + self.scale * ndtri((1.0 + u) / 2.0))
        return result^
