# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Uniform distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Uniform distribution.

Provides the `Uniform` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The continuous uniform distribution on [a, b] has PDF::

    f(x; a, b) = 1 / (b - a),  a ≤ x ≤ b
"""

from std.math import log, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Uniform distribution
# ===----------------------------------------------------------------------=== #


struct Uniform(ContinuouslyDistributed):
    """Continuous uniform distribution on [a, b].

    Represents the continuous uniform distribution, where all outcomes in the
    interval [a, b] are equally likely.

    The probability density function (PDF) for the uniform distribution is:

        f(x; a, b) = 1 / (b - a),  a ≤ x ≤ b

    Fields:
        a: Lower bound. Must be less than b.
        b: Upper bound. Must be greater than a.
    """

    var a: Float64
    var b: Float64

    def __init__(out self, a: Float64 = 0.0, b: Float64 = 1.0):
        self.a = a
        self.b = b

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            1/(b-a) for a ≤ x ≤ b; 0 otherwise.
        """
        if x >= self.a and x <= self.b:
            return 1.0 / (self.b - self.a)
        return 0.0

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            -log(b-a) for a ≤ x ≤ b; -∞ otherwise.
        """
        if x >= self.a and x <= self.b:
            return -log(self.b - self.a)
        return -inf[DType.float64]()

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            0 for x < a; (x-a)/(b-a) for a ≤ x ≤ b; 1 for x > b.
        """
        if x < self.a:
            return 0.0
        if x > self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            -∞ for x ≤ a; log((x-a)/(b-a)) for a < x ≤ b; 0 for x > b.
        """
        if x <= self.a:
            return -inf[DType.float64]()
        if x > self.b:
            return 0.0
        return log((x - self.a) / (self.b - self.a))

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            1 for x < a; (b-x)/(b-a) for a ≤ x ≤ b; 0 for x > b.
        """
        if x < self.a:
            return 1.0
        if x > self.b:
            return 0.0
        return (self.b - x) / (self.b - self.a)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            0 for x < a; log((b-x)/(b-a)) for a ≤ x < b; -∞ for x ≥ b.
        """
        if x < self.a:
            return 0.0
        if x >= self.b:
            return -inf[DType.float64]()
        return log((self.b - x) / (self.b - self.a))

    def ppf(self, q: Float64) -> Float64:
        """Percent-point (quantile) function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        return self.a + q * (self.b - self.a)

    def isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The value *x* such that SF(x) = *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        return self.b - q * (self.b - self.a)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: (a + b) / 2."""
        return (self.a + self.b) / 2.0

    def mean(self) -> Float64:
        """Distribution mean: (a + b) / 2."""
        return (self.a + self.b) / 2.0

    def variance(self) -> Float64:
        """Distribution variance: (b - a)² / 12."""
        var d = self.b - self.a
        return d * d / 12.0

    def std(self) -> Float64:
        """Distribution standard deviation: (b - a) / √12."""
        var d = self.b - self.a
        return d / 3.4641016151377545870548926830117447

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: ln(b - a)."""
        return log(self.b - self.a)

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate."""
        return self.a + random_float64() * (self.b - self.a)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        var d = self.b - self.a
        for _ in range(n):
            result.append(self.a + random_float64() * d)
        return result^
