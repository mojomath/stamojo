# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - F-distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""F-distribution (Fisher-Snedecor).

Provides the `FDist` distribution struct with PDF, log-PDF, CDF, survival
function, and percent-point function (PPF / quantile).

The F-distribution with d₁ and d₂ degrees of freedom has PDF::

    f(x; d₁,d₂) = √((d₁x)^d₁ · d₂^d₂ / (d₁x+d₂)^{d₁+d₂})
                   / (x · B(d₁/2, d₂/2))
"""

from math import sqrt, log, exp, nan, inf

from stamojo.special import betainc, lbeta, ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# F-distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct FDist(Copyable, Movable):
    """F-distribution (Fisher-Snedecor) with `dfn` numerator and `dfd`
    denominator degrees of freedom.

    Fields:
        dfn: Numerator degrees of freedom (d₁). Must be positive.
        dfd: Denominator degrees of freedom (d₂). Must be positive.
    """

    var dfn: Float64
    var dfd: Float64

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        if x < 0.0:
            return 0.0
        if x == 0.0:
            if self.dfn < 2.0:
                return inf[DType.float64]()
            elif self.dfn == 2.0:
                return 1.0
            else:
                return 0.0
        return exp(self.logpdf(x))

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        if x <= 0.0:
            return -inf[DType.float64]()
        var d1 = self.dfn
        var d2 = self.dfd
        return (
            (d1 / 2.0) * log(d1 / d2)
            + (d1 / 2.0 - 1.0) * log(x)
            - ((d1 + d2) / 2.0) * log(1.0 + d1 * x / d2)
            - lbeta(d1 / 2.0, d2 / 2.0)
        )

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        CDF(x) = I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2).
        """
        if x <= 0.0:
            return 0.0
        var d1 = self.dfn
        var d2 = self.dfd
        var u = d1 * x / (d1 * x + d2)
        return betainc(d1 / 2.0, d2 / 2.0, u)

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        return 1.0 - self.cdf(x)

    fn ppf(self, p: Float64) -> Float64:
        """Percent-point function (quantile / inverse CDF).

        Computed via Newton-Raphson with bisection fallback.

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

        # Initial guess: use the mean if df2 > 2, else 1.
        var x: Float64
        if self.dfd > 2.0:
            x = self.dfd / (self.dfd - 2.0)
        else:
            x = 1.0

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
        """Distribution mean.  Defined for d₂ > 2."""
        if self.dfd > 2.0:
            return self.dfd / (self.dfd - 2.0)
        return nan[DType.float64]()

    fn variance(self) -> Float64:
        """Distribution variance.  Defined for d₂ > 4."""
        if self.dfd > 4.0:
            var d1 = self.dfn
            var d2 = self.dfd
            return (
                2.0 * d2 * d2 * (d1 + d2 - 2.0)
                / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
            )
        return nan[DType.float64]()

    fn std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())
