# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Weibull distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Weibull distribution.

Provides the `Weibull` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Weibull distribution with shape *k* and scale *λ* has PDF:

    f(x; k, λ) = (k/λ) (x/λ)^{k-1} exp(−(x/λ)^k),  x ≥ 0
"""

from std.math import sqrt, log, exp, nan, inf, pow, log1p, expm1
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EULER_MASCHERONI = 0.5772156649015328606065120900824024


# ===----------------------------------------------------------------------=== #
# Weibull distribution
# ===----------------------------------------------------------------------=== #


struct Weibull(ContinuouslyDistributed):
    """Weibull distribution with shape `k` and scale `lam`.

    The probability density function (PDF) for the Weibull distribution is:

        f(x; k, λ) = (k/λ) * (x/λ)^{k-1} * exp(-(x/λ)^k),  x ≥ 0

    Fields:
        k: Shape parameter. Must be positive.
        lam: Scale parameter (λ). Must be positive.
    """

    var k: Float64
    var lam: Float64

    def __init__(out self, k: Float64 = 1.0, lam: Float64 = 1.0):
        self.k = k
        self.lam = lam

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*. Returns 0.0 for x < 0.
        """
        if x < 0.0:
            return 0.0
        if x == 0.0:
            if self.k < 1.0:
                return inf[DType.float64]()
            elif self.k == 1.0:
                return 1.0 / self.lam
            else:
                return 0.0
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x < 0.
        """
        if x <= 0.0:
            return -inf[DType.float64]()
        var y = x / self.lam
        return log(self.k / self.lam) + (self.k - 1.0) * log(y) - pow(y, self.k)

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*. Returns 0.0 for x < 0.
        """
        if x < 0.0:
            return 0.0
        var y = x / self.lam
        return -expm1(-pow(y, self.k))

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*. Returns -∞ for x < 0.
        """
        if x < 0.0:
            return -inf[DType.float64]()
        var y = x / self.lam
        return log(-expm1(-pow(y, self.k)))

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            Survival function value at *x*. Returns 1.0 for x < 0.
        """
        if x < 0.0:
            return 1.0
        var y = x / self.lam
        return exp(-pow(y, self.k))

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        if x < 0.0:
            return 0.0
        var y = x / self.lam
        return -pow(y, self.k)

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
        return self.lam * pow(-log1p(-q), 1.0 / self.k)

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
        return self.lam * pow(-log(q), 1.0 / self.k)

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: λ * ln(2)^{1/k}."""
        return self.lam * pow(log(2.0), 1.0 / self.k)

    def mean(self) -> Float64:
        """Distribution mean: λ * Γ(1 + 1/k)."""
        return self.lam * _gamma_approx(1.0 + 1.0 / self.k)

    def variance(self) -> Float64:
        """Distribution variance: λ² * [Γ(1+2/k) - Γ(1+1/k)²]."""
        var g1 = _gamma_approx(1.0 + 1.0 / self.k)
        var g2 = _gamma_approx(1.0 + 2.0 / self.k)
        return self.lam * self.lam * (g2 - g1 * g1)

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = γ(1 - 1/k) + ln(λ/k) + 1, where γ is Euler's constant.
        """
        return (
            _EULER_MASCHERONI * (1.0 - 1.0 / self.k)
            + log(self.lam / self.k)
            + 1.0
        )

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0:
            u = random_float64()
        return self.lam * pow(-log(u), 1.0 / self.k)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        var inv_k = 1.0 / self.k
        for _ in range(n):
            var u = random_float64()
            while u == 0.0:
                u = random_float64()
            result.append(self.lam * pow(-log(u), inv_k))
        return result^


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _gamma_approx(x: Float64) -> Float64:
    """Approximation of the gamma function using Lanczos approximation.

    Args:
        x: Input value (must be positive).

    Returns:
        Approximation of Γ(x).
    """
    if x <= 0.0:
        return nan[DType.float64]()
    if x == 1.0:
        return 1.0
    if x == 0.5:
        return 1.7724538509055160272981674833411451

    var g = 7.0
    var c: List[Float64] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    var t = x - 1.0
    var y = c[0]
    for i in range(1, 9):
        y += c[i] / (t + Float64(i))

    var z = t + g + 0.5
    return sqrt(2.0 * 3.14159265358979323846) * pow(z, t + 0.5) * exp(-z) * y
