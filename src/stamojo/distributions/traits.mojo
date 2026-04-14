# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Distribution traits
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Traits for probability distributions."""


trait ContinuouslyDistributed(Copyable, Movable):
    """Trait for continuous probability distributions."""

    # --- Density functions ---------------------------------------------------

    fn pdf(self, x: Float64) -> Float64:
        """Computes the probability density function at *x*.

        Args:
            x: Point at which to evaluate the PDF.

        Returns:
            The PDF value at *x*.
        """
        ...

    fn logpdf(self, x: Float64) -> Float64:
        """Computes the natural logarithm of the PDF at *x*.

        Args:
            x: Point at which to evaluate the log-PDF.

        Returns:
            The log-PDF value at *x*.
        """
        ...

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, x: Float64) -> Float64:
        """Computes the cumulative distribution function P(X ≤ x).

        Args:
            x: Point at which to evaluate the CDF.

        Returns:
            The CDF value at *x*.
        """
        ...

    fn logcdf(self, x: Float64) -> Float64:
        """Computes the natural logarithm of the CDF at *x*.

        Args:
            x: Point at which to evaluate the log-CDF.

        Returns:
            The log-CDF value at *x*.
        """
        ...

    fn sf(self, x: Float64) -> Float64:
        """Computes the survival function (1 − CDF) at *x*.

        Args:
            x: Point at which to evaluate the survival function.

        Returns:
            The survival function value at *x*.
        """
        ...

    fn logsf(self, x: Float64) -> Float64:
        """Computes the natural logarithm of the survival function at *x*.

        Args:
            x: Point at which to evaluate the log-SF.

        Returns:
            The log-SF value at *x*.
        """
        ...

    fn ppf(self, q: Float64) -> Float64:
        """Computes the percent point function (inverse of CDF) at *q*.

        Args:
            q: Probability in [0, 1].

        Returns:
            The quantile corresponding to *q*.
        """
        ...

    fn isf(self, q: Float64) -> Float64:
        """Computes the inverse survival function (inverse of SF) at *q*.

        Args:
            q: Probability in [0, 1].

        Returns:
            The value *x* such that SF(x) = *q*.
        """
        ...

    # --- Statistical properties ----------------------------------------------

    fn median(self) -> Float64:
        """Computes the median of the distribution.

        Returns:
            The median value.
        """
        ...

    fn mean(self) -> Float64:
        """Computes the mean of the distribution.

        Returns:
            The mean value.
        """
        ...

    fn variance(self) -> Float64:
        """Computes the variance of the distribution.

        Returns:
            The variance value.
        """
        ...

    fn std(self) -> Float64:
        """Computes the standard deviation of the distribution.

        Returns:
            The standard deviation value.
        """
        ...


trait DiscretelyDistributed(Copyable, Movable):
    """Trait for discrete probability distributions."""

    # --- Probability mass functions ------------------------------------------

    fn pmf(self, k: Int) -> Float64:
        """Computes the probability mass function at *k*.

        Args:
            k: Point at which to evaluate the PMF.

        Returns:
            The PMF value at *k*.
        """
        ...

    fn logpmf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the PMF at *k*.

        Args:
            k: Point at which to evaluate the log-PMF.

        Returns:
            The log-PMF value at *k*.
        """
        ...

    # --- Distribution functions ----------------------------------------------

    fn cdf(self, k: Int) -> Float64:
        """Computes the cumulative distribution function P(X ≤ k).

        Args:
            k: Point at which to evaluate the CDF.

        Returns:
            The CDF value at *k*.
        """
        ...

    fn logcdf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the CDF at *k*.

        Args:
            k: Point at which to evaluate the log-CDF.

        Returns:
            The log-CDF value at *k*.
        """
        ...

    fn sf(self, k: Int) -> Float64:
        """Computes the survival function (1 − CDF) at *k*.

        Args:
            k: Point at which to evaluate the survival function.

        Returns:
            The survival function value at *k*.
        """
        ...

    fn logsf(self, k: Int) -> Float64:
        """Computes the natural logarithm of the survival function at *k*.

        Args:
            k: Point at which to evaluate the log-SF.

        Returns:
            The log-SF value at *k*.
        """
        ...

    fn ppf(self, q: Float64) -> Int:
        """Computes the percent point function (inverse of CDF) at *q*.

        Args:
            q: Probability in [0, 1].

        Returns:
            The smallest integer k such that CDF(k) ≥ q.
        """
        ...

    fn isf(self, q: Float64) -> Int:
        """Computes the inverse survival function (inverse of SF) at *q*.

        Args:
            q: Probability in [0, 1].

        Returns:
            The smallest integer k such that SF(k) ≤ q.
        """
        ...

    # --- Statistical properties ----------------------------------------------

    fn median(self) -> UInt:
        """Computes the median of the distribution.

        Returns:
            The median value.
        """
        ...

    fn mean(self) -> Float64:
        """Computes the mean of the distribution.

        Returns:
            The mean value.
        """
        ...

    fn variance(self) -> Float64:
        """Computes the variance of the distribution.

        Returns:
            The variance value.
        """
        ...

    fn std(self) -> Float64:
        """Computes the standard deviation of the distribution.

        Returns:
            The standard deviation value.
        """
        ...
