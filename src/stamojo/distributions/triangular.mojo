# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Triangular distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Triangular distribution.

Provides the `Triangular` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The triangular distribution on [a, b] with mode *c* (a ≤ c ≤ b) has PDF::

    f(x; a, b, c) = 2(x−a) / ((b−a)(c−a)),  a ≤ x ≤ c
    f(x; a, b, c) = 2(b−x) / ((b−a)(b−c)),  c < x ≤ b
"""

from std.math import sqrt, log, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Triangular distribution
# ===----------------------------------------------------------------------=== #


struct Triangular(ContinuouslyDistributed):
    """Triangular distribution on [a, b] with mode `c`.

    A common choice in simulation and PERT/project-planning models when
    only a minimum, maximum, and most-likely value are known.

    The probability density function (PDF) for the triangular distribution is:

        f(x; a, b, c) = 2(x-a) / ((b-a)(c-a)),  a ≤ x ≤ c
        f(x; a, b, c) = 2(b-x) / ((b-a)(b-c)),  c < x ≤ b

    Fields:
        a: Lower bound. Must satisfy a ≤ c.
        b: Upper bound. Must satisfy b ≥ c.
        c: Mode (most likely value). Must satisfy a ≤ c ≤ b.
    """

    var a: Float64
    var b: Float64
    var c: Float64

    def __init__(
        out self, a: Float64 = 0.0, b: Float64 = 1.0, c: Float64 = 0.5
    ):
        self.a = a
        self.b = b
        self.c = c

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*. Returns 0.0 outside [a, b].
        """
        if x < self.a or x > self.b:
            return 0.0
        var span = self.b - self.a
        if x <= self.c:
            var denom = span * (self.c - self.a)
            if denom == 0.0:
                # c == a: density is a single spike handled via the b-branch.
                return 2.0 / span
            return 2.0 * (x - self.a) / denom
        else:
            var denom = span * (self.b - self.c)
            if denom == 0.0:
                return 2.0 / span
            return 2.0 * (self.b - x) / denom

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ outside [a, b].
        """
        var p = self.pdf(x)
        if p <= 0.0:
            return -inf[DType.float64]()
        return log(p)

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*.
        """
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        var span = self.b - self.a
        if x <= self.c:
            var denom = span * (self.c - self.a)
            if denom == 0.0:
                return 0.0
            var diff = x - self.a
            return diff * diff / denom
        else:
            var denom = span * (self.b - self.c)
            if denom == 0.0:
                return 1.0
            var diff = self.b - x
            return 1.0 - diff * diff / denom

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*.
        """
        var c = self.cdf(x)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*.
        """
        return 1.0 - self.cdf(x)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
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
        var span = self.b - self.a
        var fc = 0.0
        if span > 0.0:
            fc = (self.c - self.a) / span
        if q <= fc:
            if fc == 0.0:
                return self.a
            return self.a + sqrt(q * span * (self.c - self.a))
        else:
            if fc == 1.0:
                return self.b
            return self.b - sqrt((1.0 - q) * span * (self.b - self.c))

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
        """Distribution mean: (a + b + c) / 3."""
        return (self.a + self.b + self.c) / 3.0

    def variance(self) -> Float64:
        """Distribution variance: (a²+b²+c²−ab−ac−bc) / 18."""
        var a = self.a
        var b = self.b
        var c = self.c
        return (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: 0.5 + ln((b−a)/2)."""
        return 0.5 + log((self.b - self.a) / 2.0)

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        return self.ppf(u)

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
            result.append(self.ppf(u))
        return result^
