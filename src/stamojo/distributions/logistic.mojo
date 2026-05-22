# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Logistic distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Logistic distribution.

Provides the `Logistic` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The logistic distribution with location *μ* and scale *s* has PDF::

    f(x; μ, s) = exp(−(x−μ)/s) / (s (1 + exp(−(x−μ)/s))²)
"""

from std.math import sqrt, log, exp, nan, inf, pow, log1p, expm1
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _PI = 3.1415926535897932384626433832795029
comptime _SQRT_3 = 1.7320508075688772935274463415058724


# ===----------------------------------------------------------------------=== #
# Logistic distribution
# ===----------------------------------------------------------------------=== #


struct Logistic(ContinuouslyDistributed):
    """Logistic distribution with location `loc` and scale `scale`.

    The logistic distribution is similar to the normal distribution but has
    heavier tails. It is used in logistic regression and neural networks.

    The probability density function (PDF) for the logistic distribution is:

        f(x; μ, s) = exp(-(x-μ)/s) / (s * (1 + exp(-(x-μ)/s))²)

    Fields:
        loc: Location parameter (μ). Defaults to 0.0.
        scale: Scale parameter (s). Must be positive. Defaults to 1.0.
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
        var y = (x - self.loc) / self.scale
        var ey = exp(-y)
        return ey / (self.scale * (1.0 + ey) * (1.0 + ey))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return -y - log(self.scale) - 2.0 * log1p(exp(-y))

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return 1.0 / (1.0 + exp(-y))

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return -log1p(exp(-y))

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return 1.0 / (1.0 + exp(y))

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return -log1p(exp(y))

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
        return self.loc + self.scale * log(q / (1.0 - q))

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
        """Distribution variance: π² * scale² / 3."""
        return _PI * _PI * self.scale * self.scale / 3.0

    def std(self) -> Float64:
        """Distribution standard deviation: π * scale / √3."""
        return _PI * self.scale / _SQRT_3

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: ln(s) + 2."""
        return log(self.scale) + 2.0

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0 or u == 1.0:
            u = random_float64()
        return self.loc + self.scale * log(u / (1.0 - u))

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
            result.append(self.loc + self.scale * log(u / (1.0 - u)))
        return result^
