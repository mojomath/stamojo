# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Cauchy distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Cauchy distribution.

Provides the `Cauchy` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Cauchy distribution with location *x₀* and scale *γ* has PDF::

    f(x; x₀, γ) = 1 / (π γ [1 + ((x − x₀) / γ)²])
"""

from std.math import log, tan, atan, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.distributions._constants import PI as _PI


# ===----------------------------------------------------------------------=== #
# Cauchy distribution
# ===----------------------------------------------------------------------=== #


struct Cauchy(ContinuouslyDistributed):
    """Cauchy distribution with location `loc` and scale `scale`.

    The Cauchy distribution is a heavy-tailed distribution whose mean,
    variance, and higher moments are undefined (the integrals diverge).

    The probability density function (PDF) for the Cauchy distribution is:

        f(x; x₀, γ) = 1 / (π γ [1 + ((x - x₀) / γ)²])

    Fields:
        loc: Location parameter (x₀), the peak/median of the distribution. Defaults to 0.0.
        scale: Scale parameter (γ). Must be positive. Defaults to 1.0.
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
        return 1.0 / (_PI * self.scale * (1.0 + y * y))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return -log(_PI * self.scale * (1.0 + y * y))

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return 0.5 + atan(y) / _PI

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*.
        """
        return log(self.cdf(x))

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*.
        """
        var y = (x - self.loc) / self.scale
        return 0.5 - atan(y) / _PI

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        return log(self.sf(x))

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
        return self.loc + self.scale * tan(_PI * (q - 0.5))

    def isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The value *x* such that SF(x) = *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return inf[DType.float64]()
        if q == 1.0:
            return -inf[DType.float64]()
        return self.loc + self.scale * tan(_PI * (0.5 - q))

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: loc."""
        return self.loc

    def mean(self) -> Float64:
        """Distribution mean: undefined (returns NaN)."""
        return nan[DType.float64]()

    def variance(self) -> Float64:
        """Distribution variance: undefined (returns NaN)."""
        return nan[DType.float64]()

    def std(self) -> Float64:
        """Distribution standard deviation: undefined (returns NaN)."""
        return nan[DType.float64]()

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: ln(4π γ)."""
        return log(4.0 * _PI * self.scale)

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0:
            u = random_float64()
        return self.loc + self.scale * tan(_PI * (u - 0.5))

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
            while u == 0.0:
                u = random_float64()
            result.append(self.loc + self.scale * tan(_PI * (u - 0.5)))
        return result^
