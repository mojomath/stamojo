# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Pareto distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Pareto distribution.

Provides the `Pareto` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Pareto distribution with shape *α* and scale *x_m* has PDF::

    f(x; α, x_m) = α x_m^α / x^{α+1},  x ≥ x_m
"""

from std.math import sqrt, log, exp, nan, inf, pow
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Pareto distribution
# ===----------------------------------------------------------------------=== #


struct Pareto(ContinuouslyDistributed):
    """Pareto distribution with shape `alpha` and scale `x_m`.

    The Pareto distribution is a power-law probability distribution that
    describes many phenomena in nature and society.

    The probability density function (PDF) for the Pareto distribution is:

        f(x; α, x_m) = α * x_m^α / x^{α+1},  x ≥ x_m

    Fields:
        alpha: Shape parameter (α). Must be positive.
        x_m: Scale parameter (minimum value). Must be positive.
    """

    var alpha: Float64
    var x_m: Float64

    def __init__(out self, alpha: Float64 = 1.0, x_m: Float64 = 1.0):
        self.alpha = alpha
        self.x_m = x_m

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*. Returns 0.0 for x < x_m.
        """
        if x < self.x_m:
            return 0.0
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x < x_m.
        """
        if x < self.x_m:
            return -inf[DType.float64]()
        return (
            log(self.alpha)
            + self.alpha * log(self.x_m)
            - (self.alpha + 1.0) * log(x)
        )

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*. Returns 0.0 for x < x_m.
        """
        if x < self.x_m:
            return 0.0
        return 1.0 - pow(self.x_m / x, self.alpha)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*. Returns -∞ for x < x_m.
        """
        if x < self.x_m:
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
            Survival function value at *x*. Returns 1.0 for x < x_m.
        """
        if x < self.x_m:
            return 1.0
        return pow(self.x_m / x, self.alpha)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        if x < self.x_m:
            return 0.0
        return self.alpha * (log(self.x_m) - log(x))

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
            return self.x_m
        if q == 1.0:
            return inf[DType.float64]()
        return self.x_m / pow(1.0 - q, 1.0 / self.alpha)

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
            return self.x_m
        return self.x_m / pow(q, 1.0 / self.alpha)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: x_m * 2^{1/α}."""
        return self.x_m * pow(2.0, 1.0 / self.alpha)

    def mean(self) -> Float64:
        """Distribution mean: α * x_m / (α - 1) for α > 1."""
        if self.alpha > 1.0:
            return self.alpha * self.x_m / (self.alpha - 1.0)
        return inf[DType.float64]()

    def variance(self) -> Float64:
        """Distribution variance: x_m² * α / ((α-1)²(α-2)) for α > 2."""
        if self.alpha > 2.0:
            return (
                self.x_m
                * self.x_m
                * self.alpha
                / ((self.alpha - 1.0) * (self.alpha - 1.0) * (self.alpha - 2.0))
            )
        return inf[DType.float64]()

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = ln(x_m/α) + 1 + 1/α
        """
        return log(self.x_m / self.alpha) + 1.0 + 1.0 / self.alpha

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0:
            u = random_float64()
        return self.x_m / pow(1.0 - u, 1.0 / self.alpha)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        var inv_alpha = 1.0 / self.alpha
        for _ in range(n):
            var u = random_float64()
            while u == 0.0:
                u = random_float64()
            result.append(self.x_m / pow(1.0 - u, inv_alpha))
        return result^
