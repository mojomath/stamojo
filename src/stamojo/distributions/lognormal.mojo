# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - LogNormal distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""LogNormal distribution.

Provides the `LogNormal` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The lognormal distribution with parameters *μ* and *σ* has PDF::

    f(x; μ, σ) = exp(−(ln x − μ)² / (2σ²)) / (x σ √(2π)),  x > 0
"""

from std.math import sqrt, log, cos, exp, erf, erfc, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _INV_SQRT_2PI = 0.3989422804014326779399460599343819
comptime _LN_SQRT_2PI = 0.9189385332046727417803297364056176
comptime _SQRT2 = 1.4142135623730950488016887242096981
comptime _2PI = 6.283185307179586476925286766559006


# ===----------------------------------------------------------------------=== #
# LogNormal distribution
# ===----------------------------------------------------------------------=== #


struct LogNormal(ContinuouslyDistributed):
    """LogNormal distribution with parameters `mu` and `sigma`.

    A random variable X is lognormally distributed if ln(X) is normally
    distributed. Equivalently, if Y ~ N(μ, σ²), then X = exp(Y) follows
    a lognormal distribution.

    The probability density function (PDF) for the lognormal distribution is:

        f(x; μ, σ) = exp(-(ln(x) - μ)² / (2σ²)) / (x σ √(2π)),  x > 0

    Fields:
        mu: Mean of the underlying normal distribution (location parameter).
        sigma: Standard deviation of the underlying normal distribution (scale parameter). Must be positive.
    """

    var mu: Float64
    var sigma: Float64

    def __init__(out self, mu: Float64 = 0.0, sigma: Float64 = 1.0):
        self.mu = mu
        self.sigma = sigma

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF (must be > 0).

        Returns:
            PDF value at *x*. Returns 0.0 for x ≤ 0.
        """
        if x <= 0.0:
            return 0.0
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x ≤ 0.
        """
        if x <= 0.0:
            return -inf[DType.float64]()
        var z = (log(x) - self.mu) / self.sigma
        return -_LN_SQRT_2PI - log(self.sigma) - log(x) - 0.5 * z * z

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*. Returns 0.0 for x ≤ 0.
        """
        if x <= 0.0:
            return 0.0
        return 0.5 * erfc(-(log(x) - self.mu) / (self.sigma * _SQRT2))

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*. Returns -∞ for x ≤ 0.
        """
        if x <= 0.0:
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
            Survival function value at *x*. Returns 1.0 for x ≤ 0.
        """
        if x <= 0.0:
            return 1.0
        return 0.5 * erfc((log(x) - self.mu) / (self.sigma * _SQRT2))

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        if x <= 0.0:
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
        if q == 0.0:
            return 0.0
        if q == 1.0:
            return inf[DType.float64]()
        return exp(self.mu + self.sigma * _ndtri(q))

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
            return 0.0
        return exp(self.mu + self.sigma * _ndtri(1.0 - q))

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: exp(μ)."""
        return exp(self.mu)

    def mean(self) -> Float64:
        """Distribution mean: exp(μ + σ²/2)."""
        return exp(self.mu + 0.5 * self.sigma * self.sigma)

    def variance(self) -> Float64:
        """Distribution variance: (exp(σ²) - 1) * exp(2μ + σ²)."""
        var es2 = exp(self.sigma * self.sigma)
        return (es2 - 1.0) * exp(2.0 * self.mu + self.sigma * self.sigma)

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = μ + 0.5 + ln(σ√(2πe)) = μ + 0.5*(1 + ln(2π)) + ln(σ) + 0.5
        """
        return self.mu + 0.5 + _LN_SQRT_2PI + log(self.sigma) + 0.5

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (Box-Muller transform)."""
        var u1 = random_float64()
        while u1 == 0.0:
            u1 = random_float64()
        var u2 = random_float64()
        var z = sqrt(-2.0 * log(u1)) * cos(_2PI * u2)
        return exp(self.mu + self.sigma * z)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        for _ in range(n):
            var u1 = random_float64()
            while u1 == 0.0:
                u1 = random_float64()
            var u2 = random_float64()
            var z = sqrt(-2.0 * log(u1)) * cos(_2PI * u2)
            result.append(exp(self.mu + self.sigma * z))
        return result^


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _ndtri(p: Float64) -> Float64:
    """Inverse standard normal CDF (probit function)."""
    if p <= 0.0:
        return -inf[DType.float64]()
    if p >= 1.0:
        return inf[DType.float64]()
    if p == 0.5:
        return 0.0

    var q = p - 0.5
    var r: Float64

    # NOTE: Sorry for this cursed brackets.
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        2509.0809287301226727e0 * r
                                        + 33430.575583588128105e0
                                    )
                                    * r
                                    + 67265.770927008700853e0
                                )
                                * r
                                + 45921.953931549871457e0
                            )
                            * r
                            + 13731.693765509461125e0
                        )
                        * r
                        + 1971.5909503065514427e0
                    )
                    * r
                    + 133.14166789178437745e0
                )
                * r
                + 3.387132872796366608e0
            )
            / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        5226.495278852854561e0 * r
                                        + 28729.085735721942674e0
                                    )
                                    * r
                                    + 39307.89580009271061e0
                                )
                                * r
                                + 21213.794301586595867e0
                            )
                            * r
                            + 5394.1960214247511077e0
                        )
                        * r
                        + 687.1870074920579083e0
                    )
                    * r
                    + 42.313330701600911252e0
                )
                * r
                + 1.0e0
            )
        )

    r = p
    if q > 0.0:
        r = 1.0 - p

    if r <= 0.0 or r >= 1.0:
        return 0.0

    r = sqrt(-log(r))
    var x: Float64

    if r <= 5.0:
        r = r - 1.6
        x = (
            (
                (
                    (
                        (
                            (
                                (
                                    7.7454501427834140764e-4 * r
                                    + 0.0227238449892691845833e0
                                )
                                * r
                                + 0.24178072517745061177e0
                            )
                            * r
                            + 1.2704582524523683825e0
                        )
                        * r
                        + 3.64784832476320460504e0
                    )
                    * r
                    + 5.7694972214606914055e0
                )
                * r
                + 4.6303378461565452959e0
            )
            * r
            + 1.42343711074968357734e0
        ) / (
            (
                (
                    (
                        (
                            (
                                (
                                    1.05075007164441684324e-9 * r
                                    + 5.475938084995344946e-4
                                )
                                * r
                                + 0.0151986665636164571966e0
                            )
                            * r
                            + 0.14810397642748007459e0
                        )
                        * r
                        + 0.68976733498510000455e0
                    )
                    * r
                    + 1.6763848301838038494e0
                )
                * r
                + 2.05319162663775882187e0
            )
            * r
            + 1.0e0
        )
    else:
        r = r - 5.0
        x = (
            (
                (
                    (
                        (
                            (
                                (
                                    2.01033439929228813265e-7 * r
                                    + 2.71155556874348757815e-5
                                )
                                * r
                                + 0.0012426609473880784386e0
                            )
                            * r
                            + 0.026532189526576123093e0
                        )
                        * r
                        + 0.29656057182850489123e0
                    )
                    * r
                    + 1.7848265399172913358e0
                )
                * r
                + 5.4637849111641143699e0
            )
            * r
            + 6.6579046435011037772e0
        ) / (
            (
                (
                    (
                        (
                            (
                                (
                                    2.04426310338993978564e-15 * r
                                    + 1.4215117583164458887e-7
                                )
                                * r
                                + 1.8463183175100546818e-5
                            )
                            * r
                            + 7.868691311456132591e-4
                        )
                        * r
                        + 0.0148753612908506148525e0
                    )
                    * r
                    + 0.13692988092273580531e0
                )
                * r
                + 0.59983220655588793769e0
            )
            * r
            + 1.0e0
        )

    if q < 0.0:
        return -x
    return x
