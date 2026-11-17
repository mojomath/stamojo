# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Truncated normal distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Truncated normal distribution.

Provides the `TruncNorm` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The truncated normal distribution restricts a Normal(loc, scale) random
variable to the interval [a, b] (a, b given in *standardized* units, as in
`scipy.stats.truncnorm`), renormalizing the density so it integrates to 1
over that interval::

    f(x; a, b, μ, σ) = φ(z) / (σ [Φ(β) − Φ(α)]),  z = (x − μ)/σ
    α = a, β = b  (standardized bounds; support is x ∈ [μ + aσ, μ + bσ])
"""

from std.math import sqrt, log, exp, erf, erfc, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.special import ndtri
from stamojo.distributions._constants import (
    INV_SQRT_2PI as _INV_SQRT_2PI,
    SQRT2 as _SQRT2,
)


# ===----------------------------------------------------------------------=== #
# Truncated normal distribution
# ===----------------------------------------------------------------------=== #


struct TruncNorm(ContinuouslyDistributed):
    """Truncated normal distribution.

    Represents a Normal(loc, scale) distribution restricted to the interval
    [loc + a*scale, loc + b*scale], where *a* and *b* are given in
    standardized units (matching `scipy.stats.truncnorm`'s parameterization).

    The probability density function (PDF) is:

        f(x; a, b, μ, σ) = φ(z) / (σ [Φ(b) − Φ(a)]),  z = (x - μ)/σ

    for x in [μ + aσ, μ + bσ], and 0 elsewhere. φ and Φ are the standard
    normal PDF and CDF.

    Fields:
        a: Lower standardized bound. May be -inf.
        b: Upper standardized bound. Must satisfy b > a. May be +inf.
        loc: Location parameter (μ). Defaults to 0.0.
        scale: Scale parameter (σ). Must be positive. Defaults to 1.0.
    """

    var a: Float64
    var b: Float64
    var loc: Float64
    var scale: Float64

    def __init__(
        out self,
        a: Float64,
        b: Float64,
        loc: Float64 = 0.0,
        scale: Float64 = 1.0,
    ):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    # --- Internal helpers ------------------------------------------------

    def _std_pdf(self, z: Float64) -> Float64:
        """Standard normal PDF φ(z)."""
        return _INV_SQRT_2PI * exp(-0.5 * z * z)

    def _std_cdf(self, z: Float64) -> Float64:
        """Standard normal CDF Φ(z), handling ±inf bounds."""
        if z == inf[DType.float64]():
            return 1.0
        if z == -inf[DType.float64]():
            return 0.0
        return 0.5 * erfc(-z / _SQRT2)

    def _z(self, x: Float64) -> Float64:
        """Standardizes x."""
        return (x - self.loc) / self.scale

    def _norm_const(self) -> Float64:
        """Z = Φ(b) − Φ(a), the normalizing constant."""
        return self._std_cdf(self.b) - self._std_cdf(self.a)

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            PDF value at *x*. Returns 0.0 outside the truncated support.
        """
        var z = self._z(x)
        if z < self.a or z > self.b:
            return 0.0
        var denom = self._norm_const()
        if denom <= 0.0:
            return nan[DType.float64]()
        return self._std_pdf(z) / (self.scale * denom)

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            Log-PDF value at *x*. Returns -∞ outside the truncated support.
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
        var z = self._z(x)
        if z <= self.a:
            return 0.0
        if z >= self.b:
            return 1.0
        var denom = self._norm_const()
        if denom <= 0.0:
            return nan[DType.float64]()
        return (self._std_cdf(z) - self._std_cdf(self.a)) / denom

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

        var lo_z = self.loc + self.a * self.scale
        var hi_z = self.loc + self.b * self.scale
        if q == 0.0:
            return lo_z
        if q == 1.0:
            return hi_z

        var denom = self._norm_const()
        if denom <= 0.0:
            return nan[DType.float64]()

        # p_target is the CDF of the *untruncated* standard normal at the
        # answer: Phi(z) = Phi(a) + q * (Phi(b) - Phi(a)).
        var p_target = self._std_cdf(self.a) + q * denom
        if p_target <= 0.0:
            return lo_z
        if p_target >= 1.0:
            return hi_z
        var z = ndtri(p_target)
        return self.loc + self.scale * z

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
        """Distribution mean: μ + σ (φ(a) − φ(b)) / Z."""
        var denom = self._norm_const()
        if denom <= 0.0:
            return nan[DType.float64]()
        return (
            self.loc
            + self.scale
            * (self._std_pdf(self.a) - self._std_pdf(self.b))
            / denom
        )

    def variance(self) -> Float64:
        """Distribution variance.

        Var = σ² [1 + (a φ(a) − b φ(b))/Z − ((φ(a) − φ(b))/Z)²]
        """
        var denom = self._norm_const()
        if denom <= 0.0:
            return nan[DType.float64]()

        var phi_a = self._std_pdf(self.a)
        var phi_b = self._std_pdf(self.b)

        var a_term = 0.0
        if self.a != -inf[DType.float64]() and self.a != inf[DType.float64]():
            a_term = self.a * phi_a
        var b_term = 0.0
        if self.b != -inf[DType.float64]() and self.b != inf[DType.float64]():
            b_term = self.b * phi_b

        var ratio = (phi_a - phi_b) / denom
        var factor = 1.0 + (a_term - b_term) / denom - ratio * ratio
        return self.scale * self.scale * factor

    def std(self) -> Float64:
        """Distribution standard deviation."""
        return sqrt(self.variance())

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (inverse transform)."""
        var u = random_float64()
        while u == 0.0 or u == 1.0:
            u = random_float64()
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
            while u == 0.0 or u == 1.0:
                u = random_float64()
            result.append(self.ppf(u))
        return result^
