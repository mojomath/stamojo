# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Beta distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Beta distribution.

Provides the `Beta` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The beta distribution with shape parameters *a* and *b* has PDF::

    f(x; a, b) = x^{a-1} (1-x)^{b-1} / B(a, b),  0 < x < 1
"""

from math import sqrt, log, lgamma, exp, nan, inf

from stamojo.special import betainc, lbeta, ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# Beta distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Beta(Copyable, Movable):
    """Beta distribution with shape parameters `a` and `b`.

    Fields:
        a: First shape parameter. Must be positive.
        b: Second shape parameter. Must be positive.
    """

    var a: Float64
    var b: Float64

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        if x <= 0.0 or x >= 1.0:
            return 0.0
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        if x <= 0.0 or x >= 1.0:
            return -inf[DType.float64]()
        return (
            (self.a - 1.0) * log(x)
            + (self.b - 1.0) * log(1.0 - x)
            - lbeta(self.a, self.b)
        )

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        CDF(x; a, b) = I_x(a, b) (regularized incomplete beta).
        """
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return betainc(self.a, self.b, x)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*."""
        if x <= 0.0:
            return -inf[DType.float64]()
        if x >= 1.0:
            return 0.0
        var c = self.cdf(x)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        if x <= 0.0:
            return 1.0
        if x >= 1.0:
            return 0.0
        return 1.0 - self.cdf(x)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return -inf[DType.float64]()
        var s = self.sf(x)
        if s <= 0.0:
            return -inf[DType.float64]()
        return log(s)

    def ppf(self, p: Float64) -> Float64:
        """Percent-point function (quantile / inverse CDF).

        Uses Newton-Raphson with bisection fallback.

        Args:
            p: Probability value in [0, 1].

        Returns:
            The quantile corresponding to *p*.
        """
        if p < 0.0 or p > 1.0:
            return nan[DType.float64]()
        if p == 0.0:
            return 0.0
        if p == 1.0:
            return 1.0

        var mu = self.a / (self.a + self.b)
        var x: Float64
        if self.a > 1.0 and self.b > 1.0:
            var sigma = sqrt(
                self.a
                * self.b
                / ((self.a + self.b) ** 2 * (self.a + self.b + 1.0))
            )
            x = mu + sigma * ndtri(p)
            if x <= 0.0:
                x = 0.01
            if x >= 1.0:
                x = 0.99
        else:
            x = mu

        # Newton-Raphson with bisection fallback.
        var lo = 0.0
        var hi = 1.0

        for _ in range(_MAX_ITER):
            var f = self.cdf(x) - p
            if abs(f) < _EPS:
                return x

            var fp = self.pdf(x)
            if fp > 1.0e-300:
                var x_new = x - f / fp
                if f > 0.0:
                    hi = x
                else:
                    lo = x
                if x_new <= lo or x_new >= hi:
                    x = (lo + hi) / 2.0
                else:
                    x = x_new
            else:
                if f > 0.0:
                    hi = x
                else:
                    lo = x
                x = (lo + hi) / 2.0

        return x

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
        """Median of the distribution (approximation).

        Uses the approximation: (a - 1/3) / (a + b - 2/3) for a, b >= 1.
        """
        if self.a >= 1.0 and self.b >= 1.0:
            return (self.a - 1.0 / 3.0) / (self.a + self.b - 2.0 / 3.0)
        return self.a / (self.a + self.b)

    def mean(self) -> Float64:
        """Distribution mean = a / (a + b)."""
        return self.a / (self.a + self.b)

    def variance(self) -> Float64:
        """Distribution variance = ab / ((a+b)²(a+b+1))."""
        var ab = self.a + self.b
        return self.a * self.b / (ab * ab * (ab + 1.0))

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = ln(B(a,b)) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        Using digamma approximation: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²)
        """
        var digamma_a = (
            log(self.a) - 1.0 / (2.0 * self.a) - 1.0 / (12.0 * self.a * self.a)
        )
        var digamma_b = (
            log(self.b) - 1.0 / (2.0 * self.b) - 1.0 / (12.0 * self.b * self.b)
        )
        var ab = self.a + self.b
        var digamma_ab = log(ab) - 1.0 / (2.0 * ab) - 1.0 / (12.0 * ab * ab)
        return (
            lbeta(self.a, self.b)
            - (self.a - 1.0) * digamma_a
            - (self.b - 1.0) * digamma_b
            + (self.a + self.b - 2.0) * digamma_ab
        )
