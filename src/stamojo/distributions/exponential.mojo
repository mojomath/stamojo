# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Exponential distribution
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Exponential distribution.

Provides the `Exponential` distribution struct with PDF, log-PDF, CDF, survival function, percent-point function (PPF / quantile), and random variate
generation.

The exponential distribution with rate parameter λ has PDF:

    f(x; λ) = λ exp(−λx),  x ≥ 0
"""

from math import sqrt, log, lgamma, exp, nan, inf, log1p, expm1

from stamojo.distributions.traits import RVContinuousLike


struct Expon(Copyable, Movable, RVContinuousLike):
    """Exponential distribution.

    Represents the exponential distribution, a continuous probability distribution commonly
    used to model the time between independent events that occur at a constant average rate.

    The probability density function (PDF) for the standardized exponential distribution is:
        f(x) = exp(-x)
    for x >= 0.

    This implementation allows shifting and scaling of the distribution using the `loc` (location) and `scale` parameters:
        Expon.pdf(x, loc, scale) = (1/scale) * exp(-(x - loc) / scale)
    which is equivalent to `Expon.pdf((x - loc) / scale) / scale`.

    The most common parameterization uses the rate parameter λ > 0, where:
        f(x; λ) = λ * exp(-λx),  for x >= 0
    This is achieved by setting scale = 1/λ and loc = 0.
    """

    # --- Density functions ---------------------------------------------------

    fn pdf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Probability density function at x for Expon(loc, scale).

        Args:
            x: Point at which to evaluate the PDF.
            loc: Location (shift) parameter.
            scale: Scale parameter. Must be positive.

        Returns:
            0.0 for x < loc. For x >= loc returns (1/scale) * exp(-(x-loc)/scale).
        """
        var y = (x - loc) / scale
        if y < 0.0:
            return 0.0
        return exp(-y) / scale

    fn logpdf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Natural logarithm of the PDF at x for Expon(loc, scale).

        Args:
            x: Point at which to evaluate the log-PDF.
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            -∞ for x < loc. For x >= loc returns -((x - loc) / scale) - log(scale).
        """
        var y = (x - loc) / scale
        if y < 0.0:
            return -inf[DType.float64]()
        return -y - log(scale)

    # --- Distribution functions ----------------------------------------------
    fn cdf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Cumulative distribution function P(X <= x) for Expon(loc, scale).

        Args:
            x: Value at which to evaluate the CDF.
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            0.0 for x < loc. For x >= loc returns 1 - exp(-(x - loc)/scale).
        """
        if x < loc:
            return 0.0
        var y = (x - loc) / scale
        return -expm1(-y)

    fn logcdf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Natural logarithm of the CDF P(X <= x) for Expon(loc, scale).

        Args:
            x: Value at which to evaluate the log-CDF.
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            -∞ for x < loc. For x >= loc returns log(1 - exp(-(x - loc)/scale)).
        """
        if x < loc:
            return -inf[DType.float64]()
        var y = (x - loc) / scale
        return log1p(-exp(-y))

    fn sf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Survival function P(X > x) for Expon(loc, scale).

        Args:
            x: Value at which to evaluate the survival function.
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            1.0 for x < loc. For x >= loc returns exp(-(x - loc)/scale).
        """
        if x < loc:
            return 1.0
        var y = (x - loc) / scale
        return exp(-y)

    fn logsf(
        self, x: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Natural logarithm of the survival function for Expon(loc, scale).

        Args:
            x: Value at which to evaluate the log-SF.
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            0.0 for x < loc. For x >= loc returns -(x - loc)/scale.
        """
        if x < loc:
            return 0.0
        var y = (x - loc) / scale
        return -y

    fn ppf(
        self, q: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Percent-point (quantile) function for Expon(loc, scale).

        For 0 <= q < 1: PPF(q) = loc - scale * log(1 - q).
        PPF(0) = loc.
        PPF(1) = +∞.

        Args:
            q: Probability in [0, 1].
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return loc
        if q == 1.0:
            return inf[DType.float64]()
        return loc - scale * log1p(-q)

    fn isf(
        self, q: Float64, loc: Float64 = 0.0, scale: Float64 = 1.0
    ) -> Float64:
        """Inverse survival function for Expon(loc, scale).

        For 0 < q <= 1: ISF(q) = loc - scale * log(q).
        ISF(0) = +∞.
        ISF(1) = loc.

        Args:
            q: Probability in [0, 1].
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).
        """
        if q < 0.0 or q > 1.0:
            return nan[DType.float64]()
        if q == 0.0:
            return inf[DType.float64]()
        if q == 1.0:
            return loc
        return loc - scale * log(q)

    # --- Summary statistics --------------------------------------------------
    fn median(self, loc: Float64 = 0.0, scale: Float64 = 1.0) -> Float64:
        """
        Median of the Expon distribution.

        Args:
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            The median value, computed as loc + scale * log(2).
        """
        return loc + scale * log(2.0)

    fn mean(self, loc: Float64 = 0.0, scale: Float64 = 1.0) -> Float64:
        """
        Mean of the Expon distribution.

        Args:
            loc: Location (shift) parameter.
            scale: Scale parameter (must be > 0).

        Returns:
            The mean value, computed as loc + scale.
        """
        return loc + scale

    fn variance(self, loc: Float64 = 0.0, scale: Float64 = 1.0) -> Float64:
        """
        Variance of the Expon distribution.

        Args:
            loc: Location parameter (unused in variance).
            scale: Scale parameter (must be > 0).

        Returns:
            The variance value, computed as scale * scale.
        """
        return scale * scale

    fn std(self, loc: Float64 = 0.0, scale: Float64 = 1.0) -> Float64:
        """
        Standard deviation of the Expon distribution.

        Args:
            loc: Location parameter (unused in std).
            scale: Scale parameter (must be > 0).

        Returns:
            The standard deviation value, equal to scale.
        """
        return scale
