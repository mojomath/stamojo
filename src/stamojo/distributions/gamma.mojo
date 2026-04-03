# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Gamma distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Gamma distribution.

Provides the `Gamma` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The gamma distribution with shape *a* and scale *θ* has PDF::

    f(x; a, θ) = x^{a-1} exp(−x/θ) / (θ^a Γ(a)),  x > 0
"""

from math import sqrt, log, lgamma, exp, nan, inf, floor, pow

from stamojo.special import gammainc, gammaincc, ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# Gamma distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Gamma(Copyable, Movable):
    """Gamma distribution with shape `a` and scale `scale`.

    Fields:
        a: Shape parameter. Must be positive.
        scale: Scale parameter (θ). Must be positive.
    """

    var a: Float64
    var scale: Float64

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        if x <= 0.0:
            return 0.0
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        if x <= 0.0:
            return -inf[DType.float64]()
        return (
            (self.a - 1.0) * log(x)
            - x / self.scale
            - self.a * log(self.scale)
            - lgamma(self.a)
        )

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        CDF(x; a, θ) = P(a, x/θ) (regularized lower incomplete gamma).
        """
        if x <= 0.0:
            return 0.0
        return gammainc(self.a, x / self.scale)

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*."""
        if x <= 0.0:
            return -inf[DType.float64]()
        var c = self.cdf(x)
        if c <= 0.0:
            return -inf[DType.float64]()
        return log(c)

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        if x <= 0.0:
            return 1.0
        return gammaincc(self.a, x / self.scale)

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*."""
        if x <= 0.0:
            return 0.0
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
            return inf[DType.float64]()

        var a = self.a

        # Initial guess using Wilson-Hilferty approximation.
        var z = ndtri(p)
        var wh = 1.0 / (9.0 * a)
        var cube = 1.0 - wh + z * sqrt(wh)
        var x: Float64
        if cube > 0.0:
            x = a * cube * cube * cube
        else:
            x = a * 0.1

        if x <= 0.0:
            x = 0.01
        x *= self.scale

        # TODO: Since many use Newton-Raphson, we could have a separate func for this.
        var lo = 0.0
        var hi = x * 4.0 + 10.0 * self.scale
        while self.cdf(hi) < p:
            hi *= 2.0

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

        Uses the approximation: scale * a * (1 - 1/(9a))^3 for a >= 1,
        and scale * a * 2^(-1/a) for a < 1.
        """
        if self.a >= 1.0:
            return self.scale * self.a * pow(1.0 - 1.0 / (9.0 * self.a), 3)
        else:
            return self.scale * self.a * pow(2.0, -1.0 / self.a)

    def mean(self) -> Float64:
        """Distribution mean = a * θ."""
        return self.a * self.scale

    def variance(self) -> Float64:
        """Distribution variance = a * θ²."""
        return self.a * self.scale * self.scale

    def std(self) -> Float64:
        """Distribution standard deviation = √(a) * θ."""
        return sqrt(self.a) * self.scale

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = a + ln(θ) + ln(Γ(a)) + (1 - a) * ψ(a)
        where ψ is the digamma function (approximated here).
        For simplicity: H ≈ a + ln(θ) + ln(Γ(a)) - (a-1)*ψ(a)
        Using approximation ψ(a) ≈ ln(a) - 1/(2a) for large a.
        """
        # H = a + ln(scale) + ln(Gamma(a)) + (1-a)*digamma(a)
        # Approximate digamma(a) ≈ ln(a) - 1/(2a) - 1/(12a²)
        var digamma = (
            log(self.a) - 1.0 / (2.0 * self.a) - 1.0 / (12.0 * self.a * self.a)
        )
        return (
            self.a + log(self.scale) + lgamma(self.a) + (1.0 - self.a) * digamma
        )
