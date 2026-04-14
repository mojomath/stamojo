# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Exponential distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Exponential distribution.

Provides the `Exponential` distribution struct with PDF, log-PDF, CDF,
survival function, and percent-point function (PPF / quantile).

The exponential distribution with rate parameter λ has PDF:

    f(x; λ) = λ exp(−λx),  x ≥ 0
"""

from std.math import log, exp, nan, inf, log1p, expm1

from stamojo.distributions.traits import ContinuouslyDistributed


# `ContinuouslyDistributed` trait contains `Copyable` and `Movable` traits
struct Exponential(ContinuouslyDistributed):
    """Exponential distribution.

    Represents the exponential distribution, a continuous probability
    distribution commonly used to model the time between independent events
    that occur at a constant average rate.

    The probability density function (PDF) for the standardized exponential
    distribution is:

        f(x) = exp(-x),  x ≥ 0

    This implementation allows shifting and scaling via `loc` and `scale`::

        Exponential.pdf(x, loc, scale) = (1/scale) * exp(-(x - loc) / scale)

    The most common parameterization uses the rate parameter λ > 0, where::

        f(x; λ) = λ * exp(-λx),  x ≥ 0

    This is achieved by setting ``scale = 1/λ`` and ``loc = 0``.
    """

    var loc: Float64
    """Location (shift) parameter. Defaults to 0.0. The distribution is
    supported for x >= loc."""

    var scale: Float64
    """Scale parameter (must be > 0). Defaults to 1.0."""

    # --- Initialization -------------------------------------------------------

    fn __init__(out self, loc: Float64 = 0.0, scale: Float64 = 1.0):
        self.loc = loc
        self.scale = scale

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            0.0 for x < loc; (1/scale) * exp(-(x-loc)/scale) otherwise.
        """
        var y = (x - self.loc) / self.scale
        if y < 0.0:
            return 0.0
        return exp(-y) / self.scale

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            -∞ for x < loc; -((x - loc) / scale) - log(scale) otherwise.
        """
        var y = (x - self.loc) / self.scale
        if y < 0.0:
            return -inf[DType.float64]()
        return -y - log(self.scale)

    # --- Distribution functions ----------------------------------------------
    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x).

        Args:
            x: Value at which to evaluate the CDF.

        Returns:
            0.0 for x < loc; 1 - exp(-(x - loc)/scale) otherwise.
        """
        if x < self.loc:
            return 0.0
        var y = (x - self.loc) / self.scale
        return -expm1(-y)

    fn logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the CDF at *x*.

        Uses ``log(-expm1(-y))`` instead of ``log1p(-exp(-y))`` for better
        numerical stability when *x* is close to *loc*.

        Args:
            x: Value at which to evaluate the log-CDF.

        Returns:
            -∞ for x < loc; log(1 - exp(-(x - loc)/scale)) otherwise.
        """
        if x < self.loc:
            return -inf[DType.float64]()
        var y = (x - self.loc) / self.scale
        return log(-expm1(-y))

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*.

        Args:
            x: Value at which to evaluate the survival function.

        Returns:
            1.0 for x < loc; exp(-(x - loc)/scale) otherwise.
        """
        if x < self.loc:
            return 1.0
        var y = (x - self.loc) / self.scale
        return exp(-y)

    fn logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*.

        Args:
            x: Value at which to evaluate the log-SF.

        Returns:
            0.0 for x < loc; -(x - loc)/scale otherwise.
        """
        if x < self.loc:
            return 0.0
        var y = (x - self.loc) / self.scale
        return -y

    fn ppf(self, q: Float64) -> Float64:
        """Percent-point (quantile) function (inverse CDF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.

        Notes:

        PPF(q) = loc - scale * log(1 - q), 0 <= q < 1,
        PPF(0) = loc,
        PPF(1) = +∞.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return self.loc
        if q == 1.0:
            return inf[DType.float64]()
        return self.loc - self.scale * log1p(-q)

    fn isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse SF).

        Args:
            q: Probability in [0, 1].

        Returns:
            The value *x* such that SF(x) = *q*.

        Notes:

        ISF(q) = loc - scale * log(q), 0 < q <= 1,
        ISF(0) = +∞,
        ISF(1) = loc.
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return inf[DType.float64]()
        if q == 1.0:
            return self.loc
        return self.loc - self.scale * log(q)

    # --- Summary statistics --------------------------------------------------
    fn median(self) -> Float64:
        """Median of the distribution: loc + scale * ln(2).

        Returns:
            The median of the distribution.
        """
        return self.loc + self.scale * log(2.0)

    fn mean(self) -> Float64:
        """Distribution mean: loc + scale.

        Returns:
            The mean of the distribution.
        """
        return self.loc + self.scale

    fn variance(self) -> Float64:
        """Distribution variance: scale².

        Returns:
            The variance of the distribution.
        """
        return self.scale * self.scale

    fn std(self) -> Float64:
        """Distribution standard deviation: scale.

        Returns:
            The standard deviation of the distribution.
        """
        return self.scale
