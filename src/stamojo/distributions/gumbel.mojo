# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Gumbel distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Gumbel distribution.

Provides the `Gumbel` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

This is the Gumbel (right-skewed / maximum) case, matching
`scipy.stats.gumbel_r`. The Gumbel distribution with location *μ* and
scale *β* has PDF::

    f(x; μ, β) = (1/β) exp(−z − exp(−z)),  z = (x − μ) / β
"""

from std.math import log, exp, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.distributions._constants import (
    PI as _PI,
    EULER_MASCHERONI as _EULER_MASCHERONI,
)


# ===----------------------------------------------------------------------=== #
# Gumbel distribution
# ===----------------------------------------------------------------------=== #


struct Gumbel(ContinuouslyDistributed):
    """Gumbel (right-skewed / maximum) distribution with location `loc`
    and scale `scale`.

    Used to model the distribution of the maximum (or minimum, via
    reflection) of a number of samples of various distributions. This
    struct implements the "gumbel_r" convention (matching
    `scipy.stats.gumbel_r`), appropriate for extreme *maxima*.

    The probability density function (PDF) is:

        f(x; μ, β) = (1/β) * exp(-z - exp(-z)),  z = (x - μ) / β

    Fields:
        loc: Location parameter (μ). Defaults to 0.0.
        scale: Scale parameter (β). Must be positive. Defaults to 1.0.
    """

    var loc: Float64
    var scale: Float64

    def __init__(out self, loc: Float64 = 0.0, scale: Float64 = 1.0):
        self.loc = loc
        self.scale = scale

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*.
        """
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*.
        """
        var z = (x - self.loc) / self.scale
        return -log(self.scale) - z - exp(-z)

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            CDF value at *x*.
        """
        var z = (x - self.loc) / self.scale
        return exp(-exp(-z))

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            Log-CDF value at *x*.
        """
        var z = (x - self.loc) / self.scale
        return -exp(-z)

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
        if q == 0.0:
            return -inf[DType.float64]()
        if q == 1.0:
            return inf[DType.float64]()
        return self.loc - self.scale * log(-log(q))

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
        """Median of the distribution: loc − β·ln(ln 2)."""
        return self.loc - self.scale * log(log(2.0))

    def mean(self) -> Float64:
        """Distribution mean: loc + β·γ (γ = Euler–Mascheroni constant)."""
        return self.loc + self.scale * _EULER_MASCHERONI

    def variance(self) -> Float64:
        """Distribution variance: π² β² / 6."""
        return _PI * _PI * self.scale * self.scale / 6.0

    def std(self) -> Float64:
        """Distribution standard deviation: π β / √6."""
        return _PI * self.scale / 2.4494897427831780981972840747058914

    def entropy(self) -> Float64:
        """Differential entropy of the distribution: ln(β) + γ + 1."""
        return log(self.scale) + _EULER_MASCHERONI + 1.0

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0 or u == 1.0:
            u = random_float64()
        return self.loc - self.scale * log(-log(u))

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
            while u == 0.0 or u == 1.0:
                u = random_float64()
            result.append(self.loc - self.scale * log(-log(u)))
        return result^
