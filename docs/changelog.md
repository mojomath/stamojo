# StaMojo changelog

This document tracks all notable changes to StaMojo, including new features,
API changes, bug fixes, and documentation updates.

## 20260622 (v0.3.0)

StaMojo v0.3.0 updates the codebase to Mojo v1.0.0b2.

## 20260512 (v0.2.0)

StaMojo v0.2.0 adds five new probability distributions
(`Exponential`, `Binomial`, `Gamma`, `Beta`, `Poisson`) and the
`DiscretelyDistributed` trait, a complete set of Bessel functions in
`stamojo.special`, and geometric / harmonic means in `stamojo.stats`. The
codebase is migrated to Mojo v1.0.0b1.

### ⭐️ New in v0.2.0

1. Add `Exponential` distribution with `pdf`, `logpdf`, `cdf`, `sf`, `ppf`,
   `isf`, `logcdf`, `logsf`, `mean`, `var`, `std`, `rvs`, and configurable
   `loc` / `scale` parameters (PR #1).
2. Add Bessel functions in `stamojo.special`: `j0`, `j1`, `jn[n: Int]` (Bessel
   functions of the first kind), `y0`, `y1` (Bessel functions of the second
   kind), `i0`, `i1` (modified Bessel functions of the first kind), and the
   exponentially-scaled variants `i0e`, `i1e` (PR #2).
3. Add `Binomial` distribution and a new `DiscretelyDistributed` trait that
   parallels `ContinuouslyDistributed` for discrete distributions, with
   `pmf`, `logpmf`, `cdf`, `sf`, `ppf`, `isf`, `mean`, `var`, `std`, and `rvs`
   (PR #3).
4. Add geometric mean (`gmean`) and harmonic mean (`hmean`) to
   `stamojo.stats` (PR #4).
5. Add `Gamma`, `Beta`, and `Poisson` distributions, completing the v0.2.0
   distribution set. Each provides the standard PDF/PMF, CDF, SF, PPF, ISF,
   and moment APIs (PR #5).

### 🦋 Changed in v0.2.0

1. Reorganise the `distributions` subpackage so that all continuous and
   discrete distribution structs are re-exported from
   `stamojo.distributions` (`Normal`, `StudentT`, `ChiSquared`, `FDist`,
   `Exponential`, `Binomial`, `Gamma`, `Beta`, `Poisson`) (PR #5).
2. Refresh docstrings across the library to follow the StaMojo convention
   (one-line summary ending in a period, parameter / returns / examples
   sections) (PR #5).

### 🔄 Mojo v0.26.2 migration (PR #5)

- Mojo v0.26.2 migration (PR #5).
- Mojo v1.0.0b1 migration (PR #7).

### 📚 Documentation and testing in v0.2.0

- Change `fn` to `def` and add `std.` to stdlib imports
  across all source files for the Mojo v0.26.2 convention (PR #6).
- Add unit tests for every new distribution and special function:
  Bessel coverage (`test_bessel_*`) in `tests/test_special.mojo`;
  exponential / binomial / gamma / beta / poisson coverage in
  `tests/test_distributions.mojo`; geometric / harmonic mean coverage in
  `tests/test_stats.mojo`. SciPy reference-value comparisons are run for
  every new function.
- Add a GitHub Actions workflow (`.github/workflows/run_tests.yaml`) that
  runs the full test suite on every push.
- Pin the Python feature dependency to `>=3.13,<3.14` to avoid a `mojo format`
  formatting failure observed on newer Python releases.

## 20260220 (v0.1.0)

Version 0.1.0 marks the initial release of StaMojo, providing a comprehensive suite of statistical functions and hypothesis tests. This release includes one-sample, two-sample, and paired t-tests, chi-squared tests for goodness-of-fit and independence, Kolmogorov-Smirnov tests, and one-way ANOVA. The codebase is designed for performance and accuracy, with careful handling of edge cases and robust error checking. Comprehensive documentation and test coverage ensure reliability and ease of use for statistical analysis in Mojo.
