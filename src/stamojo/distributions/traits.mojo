# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Distribution traits
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Traits for probability distributions."""


trait ContinuouslyDistributed(Copyable, Movable):
    """Trait for continuous probability distributions."""

    # --- Density functions ---------------------------------------------------

    def pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        ...

    def logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        ...

    # --- Distribution functions ----------------------------------------------

    def cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x)."""
        ...

    def logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the cumulative distribution function at *x*."""
        ...

    def sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        ...

    def logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*."""
        ...

    def ppf(self, q: Float64) -> Float64:
        """Percent point function (inverse of CDF) at *q*."""
        ...

    def isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse of SF) at *q*."""
        ...

    # --- Statistical properties ----------------------------------------------

    def median(self) -> Float64:
        """Median of the distribution."""
        ...

    def mean(self) -> Float64:
        """Mean of the distribution."""
        ...

    def variance(self) -> Float64:
        """Variance of the distribution."""
        ...

    def std(self) -> Float64:
        """Standard deviation of the distribution."""
        ...


trait DiscretelyDistributed(Copyable, Movable):
    """Trait for discrete probability distributions."""

    # --- Probability mass functions ------------------------------------------

    def pmf(self, k: Int) -> Float64:
        """Probability mass function at *k*."""
        ...

    def logpmf(self, k: Int) -> Float64:
        """Natural logarithm of the probability mass function at *k*."""
        ...

    # --- Distribution functions ----------------------------------------------

    def cdf(self, k: Int) -> Float64:
        """Cumulative distribution function P(X ≤ k)."""
        ...

    def logcdf(self, k: Int) -> Float64:
        """Natural logarithm of the cumulative distribution function at *k*."""
        ...

    def sf(self, k: Int) -> Float64:
        """Survival function (1 − CDF) at *k*."""
        ...

    def logsf(self, k: Int) -> Float64:
        """Natural logarithm of the survival function at *k*."""
        ...

    def ppf(self, q: Float64) -> Int:
        """Percent point function (inverse of CDF) at *q*."""
        ...

    def isf(self, q: Float64) -> Int:
        """Inverse survival function (inverse of SF) at *q*."""
        ...

    # --- Statistical properties ----------------------------------------------

    def median(self) -> UInt:
        """Median of the distribution."""
        ...

    def mean(self) -> Float64:
        """Mean of the distribution."""
        ...

    def variance(self) -> Float64:
        """Variance of the distribution."""
        ...

    def std(self) -> Float64:
        """Standard deviation of the distribution."""
        ...
