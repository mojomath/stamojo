# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Half-Cauchy distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Half-Cauchy distribution.

Provides the `HalfCauchy` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The half-Cauchy distribution with location *μ* and scale *γ* has PDF::

    f(x; μ, γ) = 2 / (π γ [1 + ((x−μ)/γ)²]),  x ≥ μ
"""

from std.math import log, tan, atan, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.distributions._constants import PI as _PI


# ===----------------------------------------------------------------------=== #
# Half-Cauchy distribution
# ===----------------------------------------------------------------------=== #


struct HalfCauchy(ContinuouslyDistributed):
    """Half-Cauchy distribution with location `loc` and scale `scale`.

    The half-Cauchy distribution is the distribution of |X| where
    X ~ Cauchy(loc, scale). It is a common weakly-informative prior for
    scale parameters in Bayesian modeling. Its mean and variance are
    undefined, as with the full Cauchy distribution.

    The probability density function (PDF) for the half-Cauchy distribution is:

        f(x; μ, γ) = 2 / (π γ [1 + ((x-μ)/γ)²]),  x ≥ μ

    Fields:
        loc: Location parameter (μ), the lower bound of support. Defaults to 0.0.
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
            PDF value at *x*. Returns 0.0 for x < loc.
        """
        if x < self.loc:
            return 0.0
        var y = (x - self.loc) / self.scale
        return 2.0 / (_PI * self.scale * (1.0 + y * y))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x < loc.
        """
        if x < self.loc:
            return -inf[DType.float64]()
        var y = (x - self.loc) / self.scale
        return log(2.0) - log(_PI * self.scale * (1.0 + y * y))

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
        var y = (x - self.loc) / self.scale
        return 2.0 / _PI * atan(y)

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
        var y = (x - self.loc) / self.scale
        return 1.0 - 2.0 / _PI * atan(y)

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
        return self.loc + self.scale * tan(_PI * q / 2.0)

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
        return self.loc + self.scale * tan(_PI * (1.0 - q) / 2.0)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: loc + scale."""
        return self.loc + self.scale

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
        """Differential entropy of the distribution: ln(2π γ)."""
        return log(2.0 * _PI * self.scale)

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0:
            u = random_float64()
        return self.loc + self.scale * tan(_PI * u / 2.0)

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
            result.append(self.loc + self.scale * tan(_PI * u / 2.0))
        return result^
