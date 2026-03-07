# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Distribution traits
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Traits for probability distributions."""


trait ContinuouslyDistributed(Copyable, Movable):
    """Trait for continuous probability distributions."""

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Probability density function at *x*."""
        ...

    fn logpdf(self, x: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        ...

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x)."""
        ...

    fn logcdf(self, x: Float64) -> Float64:
        """Natural logarithm of the cumulative distribution function at *x*."""
        ...

    fn sf(self, x: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        ...

    fn logsf(self, x: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*."""
        ...

    fn ppf(self, q: Float64) -> Float64:
        """Percent point function (inverse of CDF) at *q*."""
        ...

    fn isf(self, q: Float64) -> Float64:
        """Inverse survival function (inverse of SF) at *q*."""
        ...

    # --- Statistical properties ----------------------------------------------

    fn median(self) -> Float64:
        """Median of the distribution."""
        ...

    fn mean(self) -> Float64:
        """Mean of the distribution."""
        ...

    fn variance(self) -> Float64:
        """Variance of the distribution."""
        ...

    fn std(self) -> Float64:
        """Standard deviation of the distribution."""
        ...


trait DiscretelyDistributed(Copyable, Movable):
    """Trait for discrete probability distributions."""

    # --- Probability mass functions ------------------------------------------

    fn pmf(self, k: Int) -> Float64:
        """Probability mass function at *k*."""
        ...

    fn logpmf(self, k: Int) -> Float64:
        """Natural logarithm of the probability mass function at *k*."""
        ...

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, k: Int) -> Float64:
        """Cumulative distribution function P(X ≤ k)."""
        ...

    fn logcdf(self, k: Int) -> Float64:
        """Natural logarithm of the cumulative distribution function at *k*."""
        ...

    fn sf(self, k: Int) -> Float64:
        """Survival function (1 − CDF) at *k*."""
        ...

    fn logsf(self, k: Int) -> Float64:
        """Natural logarithm of the survival function at *k*."""
        ...

    fn ppf(self, q: Float64) -> Int:
        """Percent point function (inverse of CDF) at *q*."""
        ...

    fn isf(self, q: Float64) -> Int:
        """Inverse survival function (inverse of SF) at *q*."""
        ...

    # --- Statistical properties ----------------------------------------------

    fn median(self) -> UInt:
        """Median of the distribution."""
        ...

    fn mean(self) -> Float64:
        """Mean of the distribution."""
        ...

    fn variance(self) -> Float64:
        """Variance of the distribution."""
        ...

    fn std(self) -> Float64:
        """Standard deviation of the distribution."""
        ...
