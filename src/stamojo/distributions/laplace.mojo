# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Laplace distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Laplace distribution.

Provides the `Laplace` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Laplace distribution with location *μ* and scale *b* has PDF::

    f(x; μ, b) = (1 / (2b)) exp(−|x − μ| / b)
"""

from std.math import sqrt, log, exp, nan, inf, abs
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Laplace distribution
# ===----------------------------------------------------------------------=== #


struct Laplace(ContinuouslyDistributed):
    """Laplace (double exponential) distribution with location `loc` and scale `scale`.

    The Laplace distribution is the distribution of the difference between two
    independent identically distributed exponential random variables.

    The probability density function (PDF) for the Laplace distribution is:

        f(x; μ, b) = (1 / (2b)) * exp(-|x - μ| / b)

    Fields:
        loc: Location parameter (μ). Defaults to 0.0.
        scale: Scale parameter (b). Must be positive. Defaults to 1.0.
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
            PDF value at *x*.
        """
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*.
        """
        return -log(2.0 * self.scale) - abs(x - self.loc) / self.scale

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*.
        """
        var d = (x - self.loc) / self.scale
        if d < 0.0:
            return 0.5 * exp(d)
        else:
            return 1.0 - 0.5 * exp(-d)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*.
        """
        var d = (x - self.loc) / self.scale
        if d < 0.0:
            return log(0.5) + d
        else:
            return log(1.0 - 0.5 * exp(-d))

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*.
        """
        var d = (x - self.loc) / self.scale
        if d < 0.0:
            return 1.0 - 0.5 * exp(d)
        else:
            return 0.5 * exp(-d)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        var d = (x - self.loc) / self.scale
        if d < 0.0:
            return log(1.0 - 0.5 * exp(d))
        else:
            return log(0.5) - d

    def ppf(self, q: Float64) -> Float64:
        """Percent-point (quantile) function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return -inf[DType.float64]()
        if q == 1.0:
            return inf[DType.float64]()
        if q < 0.5:
            return self.loc + self.scale * log(2.0 * q)
        else:
            return self.loc - self.scale * log(2.0 * (1.0 - q))

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
        """Median of the distribution: loc."""
        return self.loc

    def mean(self) -> Float64:
        """Distribution mean: loc."""
        return self.loc

    def variance(self) -> Float64:
        """Distribution variance: 2 * scale²."""
        return 2.0 * self.scale * self.scale

    def std(self) -> Float64:
        """Distribution standard deviation: scale * √2."""
        return self.scale * 1.4142135623730950488016887242096981

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: 1 + ln(2b)."""
        return 1.0 + log(2.0 * self.scale)

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64() - 0.5
        return self.loc - self.scale * _sgn(u) * log(1.0 - 2.0 * abs(u))

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        for _ in range(n):
            var u = random_float64() - 0.5
            result.append(
                self.loc - self.scale * _sgn(u) * log(1.0 - 2.0 * abs(u))
            )
        return result^


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _sgn(x: Float64) -> Float64:
    """Sign function: returns 1 if x >= 0, -1 otherwise."""
    if x >= 0.0:
        return 1.0
    return -1.0
