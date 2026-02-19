# ===----------------------------------------------------------------------=== #
# Stamojo - Stats subpackage
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Descriptive statistics, hypothesis tests, and correlation coefficients.

This subpackage provides:
- Descriptive statistics (mean, variance, skewness, kurtosis, etc.)
- Hypothesis tests (t-test, chi-squared, Kolmogorov-Smirnov, ANOVA)
- Correlation coefficients (Pearson, Spearman, Kendall) with p-values
"""

from .descriptive import (
    mean,
    variance,
    std,
    median,
    quantile,
    skewness,
    kurtosis,
    data_min,
    data_max,
)

from .correlation import (
    pearsonr,
    spearmanr,
    kendalltau,
)

from .tests import (
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    chi2_gof,
    chi2_ind,
    ks_1samp,
    f_oneway,
)
