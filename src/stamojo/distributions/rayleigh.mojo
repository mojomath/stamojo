# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Rayleigh distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Rayleigh distribution.

Provides the `Rayleigh` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Rayleigh distribution with scale *σ* has PDF::

    f(x; σ) = (x / σ²) exp(−x² / (2σ²)),  x ≥ 0
"""

from std.math import sqrt, log, exp, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.distributions._constants import (
    PI as _PI,
    EULER_MASCHERONI as _EULER_MASCHERONI,
)


# ===----------------------------------------------------------------------=== #
# Rayleigh distribution
# ===----------------------------------------------------------------------=== #


struct Rayleigh(ContinuouslyDistributed):
    """Rayleigh distribution with scale `sigma`.

    The Rayleigh distribution arises when a 2-D vector's components are
    i.i.d. zero-mean Gaussians; the magnitude of that vector is Rayleigh
    distributed. It is a special case of the Weibull (k=2) and chi
    (df=2) distributions.

    The probability density function (PDF) for the Rayleigh distribution is:

        f(x; σ) = (x / σ²) * exp(-x² / (2σ²)),  x ≥ 0

    Fields:
        sigma: Scale parameter (σ). Must be positive. Defaults to 1.0.
    """

    var sigma: Float64

    def __init__(out self, sigma: Float64 = 1.0):
        self.sigma = sigma

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
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ for x < 0.
        """
        if x < 0.0:
            return -inf[DType.float64]()
        var s2 = self.sigma * self.sigma
        return log(x) - log(s2) - x * x / (2.0 * s2)

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
        return 1.0 - exp(-x * x / (2.0 * self.sigma * self.sigma))

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*. Returns -∞ for x < 0.
        """
        if x < 0.0:
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
            Survival function value at *x*. Returns 1.0 for x < 0.
        """
        if x < 0.0:
            return 1.0
        return exp(-x * x / (2.0 * self.sigma * self.sigma))

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            Log-survival function value at *x*.
        """
        if x < 0.0:
            return 0.0
        return -x * x / (2.0 * self.sigma * self.sigma)

    def ppf(self, q: Float64) -> Float64:
        """Percent-point (quantile) function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 1.0:
            return inf[DType.float64]()
        return self.sigma * sqrt(-2.0 * log(1.0 - q))

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
        return self.sigma * sqrt(-2.0 * log(q))

    # --- Summary statistics --------------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution: σ√(2 ln 2)."""
        return self.sigma * sqrt(2.0 * log(2.0))

    def mean(self) -> Float64:
        """Distribution mean: σ√(π/2)."""
        return self.sigma * sqrt(_PI / 2.0)

    def variance(self) -> Float64:
        """Distribution variance: (4 − π)/2 · σ²."""
        return (4.0 - _PI) / 2.0 * self.sigma * self.sigma

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: 1 + ln(σ/√2) + γ/2."""
        return 1.0 + log(self.sigma / sqrt(2.0)) + _EULER_MASCHERONI / 2.0

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0:
            u = random_float64()
        return self.sigma * sqrt(-2.0 * log(u))

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
            while u == 0.0:
                u = random_float64()
            result.append(self.sigma * sqrt(-2.0 * log(u)))
        return result^
