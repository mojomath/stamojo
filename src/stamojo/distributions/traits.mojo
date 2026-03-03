trait RVContinuousLike(Copyable, Movable):
    """Trait for continuous random variable distributions."""

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Probability density function at *x*."""
        ...

    fn logpdf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Natural logarithm of the probability density function at *x*."""
        ...

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Cumulative distribution function P(X ≤ x)."""
        ...

    fn logcdf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Natural logarithm of the cumulative distribution function at *x*."""
        ...

    fn sf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Survival function (1 − CDF) at *x*."""
        ...

    fn logsf(self, x: Float64, loc: Float64, scale: Float64) -> Float64:
        """Natural logarithm of the survival function at *x*."""
        ...

    fn ppf(self, q: Float64, loc: Float64, scale: Float64) -> Float64:
        """Percent point function (inverse of CDF) at *q*."""
        ...

    fn isf(self, q: Float64, loc: Float64, scale: Float64) -> Float64:
        """Inverse survival function (inverse of SF) at *q*."""
        ...

    # --- Statistical properties ------------------------------------------------
    fn median(self, loc: Float64, scale: Float64) -> Float64:
        """Median of the distribution."""
        ...

    fn mean(self, loc: Float64, scale: Float64) -> Float64:
        """Mean of the distribution."""
        ...

    fn var(self, loc: Float64, scale: Float64) -> Float64:
        """Variance of the distribution."""
        ...

    fn std(self, loc: Float64, scale: Float64) -> Float64:
        """Standard deviation of the distribution."""
        ...
