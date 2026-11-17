# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Von Mises distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Von Mises distribution.

Provides the `VonMises` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The Von Mises distribution with mean direction *μ* and concentration *κ*
has PDF (on the circle, conventionally taken as x ∈ [μ − π, μ + π])::

    f(x; μ, κ) = exp(κ cos(x − μ)) / (2π I₀(κ))

where I₀ is the modified Bessel function of the first kind, order 0.
"""

from std.math import sqrt, log, sin, cos, acos, exp, nan, inf
from std.random import random_float64

from stamojo.distributions.traits import ContinuouslyDistributed
from stamojo.special import i0, i0e, i1, i1e
from stamojo.distributions._constants import PI as _PI, TWO_PI as _2PI


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _EPS = 1.0e-12
comptime _MAX_ITER = 100


# ===----------------------------------------------------------------------=== #
# Von Mises distribution
# ===----------------------------------------------------------------------=== #


struct VonMises(ContinuouslyDistributed):
    """Von Mises distribution with mean direction `mu` and concentration `kappa`.

    The Von Mises distribution is the circular analogue of the normal
    distribution, used in directional/circular statistics. The CDF/PPF
    here are defined on the principal branch x ∈ [μ − π, μ + π], matching
    `scipy.stats.vonmises`.

    The probability density function (PDF) is:

        f(x; μ, κ) = exp(κ cos(x - μ)) / (2π I₀(κ))

    where I₀ is the modified Bessel function of the first kind, order 0.

    Fields:
        mu: Mean direction (μ), in radians. Defaults to 0.0.
        kappa: Concentration parameter (κ). Must be non-negative
            (κ=0 gives the circular uniform distribution). Defaults to 1.0.
    """

    var mu: Float64
    var kappa: Float64

    def __init__(out self, mu: Float64 = 0.0, kappa: Float64 = 1.0):
        self.mu = mu
        self.kappa = kappa

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF (radians).

        Returns:
            PDF value at *x*.
        """
        return exp(self.logpdf(x))

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Uses the exponentially-scaled Bessel function `i0e` for numerical
        stability at large κ: I₀(κ) = i0e(κ) · exp(κ).

        Args:
            x: Point at which to evaluate the log-PDF (radians).

        Returns:
            Log-PDF value at *x*.
        """
        return (
            self.kappa * (cos(x - self.mu) - 1.0)
            - log(_2PI)
            - log(i0e(self.kappa))
        )

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function on the principal branch.

        Computed by numerically integrating the PDF from μ − π to *x*
        (clamped to [μ − π, μ + π]) using Simpson's rule.

        Args:
            x: Value at which to evaluate the CDF (radians).

        Returns:
            CDF value at *x*, in [0, 1].
        """
        var lo = self.mu - _PI
        var hi = self.mu + _PI
        var xc = x
        if xc <= lo:
            return 0.0
        if xc >= hi:
            return 1.0
        return _simpson_integral(self, lo, xc)

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
        """Percent-point (quantile) function (inverse CDF) on the principal branch.

        Computed via Newton-Raphson (using the PDF as the derivative of the
        CDF) with bisection fallback.

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*, in [μ − π, μ + π].
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        var lo = self.mu - _PI
        var hi = self.mu + _PI
        if q == 0.0:
            return lo
        if q == 1.0:
            return hi

        var x = self.mu + (q - 0.5) * _2PI  # linear initial guess

        for _ in range(_MAX_ITER):
            var f = self.cdf(x) - q
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
        """Median of the distribution: mu."""
        return self.mu

    def mean(self) -> Float64:
        """Mean direction of the distribution: mu."""
        return self.mu

    def variance(self) -> Float64:
        """Circular variance: 1 − I₁(κ)/I₀(κ)."""
        if self.kappa == 0.0:
            return 1.0
        return 1.0 - i1e(self.kappa) / i0e(self.kappa)

    def std(self) -> Float64:
        """Circular standard deviation: √(−2 ln(I₁(κ)/I₀(κ)))."""
        if self.kappa == 0.0:
            return inf[DType.float64]()
        var r = i1e(self.kappa) / i0e(self.kappa)
        if r <= 0.0:
            return inf[DType.float64]()
        return sqrt(-2.0 * log(r))

    def entropy(self) -> Float64:
        """Differential entropy of the distribution.

        H = ln(2π I₀(κ)) − κ I₁(κ)/I₀(κ)
        """
        return (
            log(_2PI)
            + log(i0e(self.kappa))
            + self.kappa
            - self.kappa * i1e(self.kappa) / i0e(self.kappa)
        )

    # --- Random variate generation -------------------------------------------

    def rvs(self) -> Float64:
        """Generate a single random variate (Best-Fisher algorithm)."""
        return _vonmises_sample(self.mu, self.kappa)

    def rvs(self, n: Int) -> List[Float64]:
        """Generate *n* random variates.

        Args:
            n: Number of variates to generate.

        Returns:
            A list of *n* random variates from this distribution.
        """
        var result = List[Float64](capacity=n)
        for _ in range(n):
            result.append(_vonmises_sample(self.mu, self.kappa))
        return result^


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _simpson_integral(dist: VonMises, lo: Float64, hi: Float64) -> Float64:
    """Numerically integrates the Von Mises PDF over [lo, hi] via Simpson's rule.
    """
    comptime n = 200  # even number of subintervals
    var h = (hi - lo) / Float64(n)
    var total = dist.pdf(lo) + dist.pdf(hi)
    for i in range(1, n):
        var x = lo + Float64(i) * h
        if i % 2 == 0:
            total += 2.0 * dist.pdf(x)
        else:
            total += 4.0 * dist.pdf(x)
    var result = total * h / 3.0
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


def _vonmises_sample(mu: Float64, kappa: Float64) -> Float64:
    """Generates a single Von Mises variate using the Best-Fisher algorithm."""
    if kappa < 1.0e-8:
        # Degenerates to circular uniform.
        var u = random_float64()
        var x = mu - _PI + u * _2PI
        return _wrap(x, mu)

    var a = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa)
    var b = (a - sqrt(2.0 * a)) / (2.0 * kappa)
    var r = (1.0 + b * b) / (2.0 * b)

    var x: Float64
    while True:
        var u1 = random_float64()
        var z = cos(_PI * u1)
        var f = (1.0 + r * z) / (r + z)
        var c = kappa * (r - f)

        var u2 = random_float64()
        if c * (2.0 - c) - u2 > 0.0:
            var u3 = random_float64() - 0.5
            var sgn = 1.0
            if u3 < 0.0:
                sgn = -1.0
            x = mu + sgn * _acos_safe(f)
            break
        if log(c / u2) + 1.0 - c >= 0.0:
            var u3 = random_float64() - 0.5
            var sgn = 1.0
            if u3 < 0.0:
                sgn = -1.0
            x = mu + sgn * _acos_safe(f)
            break

    return _wrap(x, mu)


def _acos_safe(x: Float64) -> Float64:
    """Numerically-safe arccos, clamping x to [-1, 1]."""
    var y = x
    if y > 1.0:
        y = 1.0
    elif y < -1.0:
        y = -1.0
    return acos(y)


def _wrap(x: Float64, mu: Float64) -> Float64:
    """Wraps *x* into [mu - π, mu + π]."""
    var y = x
    while y > mu + _PI:
        y -= _2PI
    while y < mu - _PI:
        y += _2PI
    return y
