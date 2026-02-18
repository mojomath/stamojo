# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Chi-squared distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Chi-squared distribution.

Provides the `ChiSquared` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The chi-squared distribution with *k* degrees of freedom has PDF::

    f(x; k) = x^{k/2−1} exp(−x/2) / (2^{k/2} Γ(k/2)),  x > 0
"""

from math import sqrt, log, lgamma, exp, nan, inf

from stamojo.special import gammainc, gammaincc, ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _LN2 = 0.6931471805599453094172321214581766
comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# Chi-squared distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ChiSquared(Copyable, Movable):
    """Chi-squared distribution with `df` degrees of freedom.

    Fields:
        df: Degrees of freedom. Must be positive.
    """

    var df: Float64

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        if x < 0.0:
            return 0.0
        if x == 0.0:
            if self.df < 2.0:
                return inf[DType.float64]()
            elif self.df == 2.0:
                return 0.5
            else:
                return 0.0
        return exp(self.logpdf(x))

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        if x <= 0.0:
            return -inf[DType.float64]()
        var k = self.df
        return (
            (k / 2.0 - 1.0) * log(x)
            - x / 2.0
            - (k / 2.0) * _LN2
            - lgamma(k / 2.0)
        )

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        CDF(x; k) = P(k/2, x/2) (regularized lower incomplete gamma).
        """
        if x <= 0.0:
            return 0.0
        return gammainc(self.df / 2.0, x / 2.0)

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        if x <= 0.0:
            return 1.0
        return gammaincc(self.df / 2.0, x / 2.0)

    fn ppf(self, p: Float64) -> Float64:
        """Percent-point function (quantile / inverse CDF).

        Uses the Wilson-Hilferty initial approximation refined by
        Newton-Raphson with bisection fallback.

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

        var k = self.df

        # Wilson-Hilferty approximation for initial guess.
        var z = ndtri(p)
        var wh = 2.0 / (9.0 * k)
        var cube = 1.0 - wh + z * sqrt(wh)
        var x: Float64
        if cube > 0.0:
            x = k * cube * cube * cube
        else:
            x = k * 0.1  # fallback for extreme tails

        if x <= 0.0:
            x = 0.01

        # Newton-Raphson with bisection fallback.
        var lo = 0.0
        var hi = x * 4.0 + 10.0
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

    # --- Summary statistics --------------------------------------------------

    fn mean(self) -> Float64:
        """Distribution mean = k."""
        return self.df

    fn variance(self) -> Float64:
        """Distribution variance = 2k."""
        return 2.0 * self.df

    fn std(self) -> Float64:
        """Distribution standard deviation = √(2k)."""
        return sqrt(2.0 * self.df)
