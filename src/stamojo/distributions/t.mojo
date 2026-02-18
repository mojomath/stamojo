# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Student's t distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Student's t-distribution.

Provides the `StudentT` distribution struct with PDF, log-PDF, CDF, survival
function, and percent-point function (PPF / quantile).

The Student's t-distribution with ν degrees of freedom has PDF::

    f(x; ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) (1 + x²/ν)^{-(ν+1)/2}
"""

from math import sqrt, log, lgamma, exp, nan, inf

from stamojo.special import betainc, ndtri


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _LN_PI = 1.1447298858494001741434273513530587
comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# Student's t distribution
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct StudentT(Copyable, Movable):
    """Student's t-distribution with `df` degrees of freedom.

    Fields:
        df: Degrees of freedom. Must be positive.
    """

    var df: Float64

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        return exp(self.logpdf(x))

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        var v = self.df
        return (
            lgamma((v + 1.0) / 2.0)
            - 0.5 * log(v) - 0.5 * _LN_PI
            - lgamma(v / 2.0)
            - ((v + 1.0) / 2.0) * log(1.0 + x * x / v)
        )

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Uses the regularized incomplete beta function:
            CDF(t) = 1 − 0.5·I_u(ν/2, 1/2)  for t ≥ 0
            CDF(t) = 0.5·I_u(ν/2, 1/2)      for t < 0
        where u = ν / (ν + t²).
        """
        var v = self.df
        var u = v / (v + x * x)
        var ib = betainc(v / 2.0, 0.5, u)
        if x >= 0.0:
            return 1.0 - 0.5 * ib
        else:
            return 0.5 * ib

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        var v = self.df
        var u = v / (v + x * x)
        var ib = betainc(v / 2.0, 0.5, u)
        if x >= 0.0:
            return 0.5 * ib
        else:
            return 1.0 - 0.5 * ib

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
            return -inf[DType.float64]()
        if p == 1.0:
            return inf[DType.float64]()
        if p == 0.5:
            return 0.0

        # Initial guess from normal approximation.
        var x = ndtri(p)

        # Newton-Raphson with bisection fallback.
        var lo = -1000.0
        var hi = 1000.0

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
        """Distribution mean.  Defined for df > 1."""
        if self.df > 1.0:
            return 0.0
        return nan[DType.float64]()

    fn variance(self) -> Float64:
        """Distribution variance.  Defined for df > 2; infinite for 1 < df ≤ 2."""
        if self.df > 2.0:
            return self.df / (self.df - 2.0)
        if self.df > 1.0:
            return inf[DType.float64]()
        return nan[DType.float64]()

    fn std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())
