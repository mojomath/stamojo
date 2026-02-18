# StaMojo <!-- omit in toc -->

A statistical computing library for [Mojo](https://www.modular.com/mojo), inspired by `scipy.stats` and `statsmodels` in Python.

**[Repository on GitHub»](https://github.com/mojomath/stamojo)**　|　**[Discord channel»](https://discord.gg/3rGH87uZTk)**

- [Overview](#overview)
  - [Why a separate library?](#why-a-separate-library)
- [Installation](#installation)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
  - [Phase 0 — Foundation (current)](#phase-0--foundation-current)
  - [Phase 1 — Core Distributions](#phase-1--core-distributions)
  - [Phase 2 — Hypothesis Testing](#phase-2--hypothesis-testing)
  - [Phase 3 — OLS Regression](#phase-3--ols-regression)
  - [Phase 4 — Generalized Linear Models](#phase-4--generalized-linear-models)
  - [Phase 5 — Extended Distributions \& Models](#phase-5--extended-distributions--models)
  - [Phase 6 — Advanced Topics](#phase-6--advanced-topics)
- [License](#license)

## Overview

StaMojo (Statistics + Mojo) brings comprehensive statistical computing to the Mojo ecosystem. The library covers three major areas:

1. **Probability distributions** — PDF/PMF, CDF, quantile functions (PPF), and random variate generation for continuous and discrete distributions.
2. **Statistical tests & descriptive statistics** — Hypothesis tests (t-test, chi-squared, K-S, etc.), summary statistics (mean, median, variance, skewness, kurtosis, correlation), and related utilities.
3. **Statistical models** — OLS, WLS, GLS, logistic regression, GLM, and model diagnostics (R², AIC, BIC, residual analysis, etc.).

StaMojo builds on top of [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) for n-dimensional array and linear algebra support.

### Why a separate library?

In the Python ecosystem, `scipy` bundles statistics, optimization, signal processing, integration, interpolation, and more into one giant package. For Mojo, a modular approach is more appropriate:

| Python package       | Mojo equivalent                     | Focus                             |
| -------------------- | ----------------------------------- | --------------------------------- |
| `numpy`              | **NuMojo**                          | N-dimensional arrays, basic math  |
| `decimal` / `mpmath` | **DeciMojo**                        | Arbitrary-precision arithmetic    |
| `scipy.stats`        | **StaMojo** (distributions + tests) | Statistical distributions & tests |
| `statsmodels`        | **StaMojo** (models)                | Statistical models & econometrics |

Placing `scipy.stats`-like functionality and `statsmodels`-like regression in **one library** is intentional: regression models inherently depend on distribution functions (for p-values, confidence intervals, etc.), so co-locating them avoids circular dependencies, simplifies versioning, and provides a cohesive API.

## Installation

Stamojo will be published to the `modular-community` channel once it reaches a functional first release. During development, clone the repo and build locally:

```bash
git clone https://github.com/mojomath/stamojo.git
cd stamojo
pixi install
pixi run package
```

## Architecture

```txt
src/stamojo/
├── __init__.mojo              # Package root
├── prelude.mojo               # Convenient re-exports
├── distributions/             # Probability distributions
│   ├── __init__.mojo
│   ├── normal.mojo            # Normal (Gaussian)
│   ├── t.mojo                 # Student's t
│   ├── chi2.mojo              # Chi-squared
│   ├── f.mojo                 # F-distribution
│   ├── beta.mojo              # Beta
│   ├── gamma.mojo             # Gamma
│   ├── binomial.mojo          # Binomial
│   ├── poisson.mojo           # Poisson
│   └── ...
├── stats/                     # Descriptive stats & hypothesis tests
│   ├── __init__.mojo
│   ├── descriptive.mojo       # mean, var, std, skewness, kurtosis, quantile
│   ├── correlation.mojo       # Pearson, Spearman, Kendall
│   ├── tests.mojo             # t-test, chi2-test, K-S test, etc.
│   └── ...
├── models/                    # Statistical models
│   ├── __init__.mojo
│   ├── ols.mojo               # Ordinary Least Squares
│   ├── gls.mojo               # Generalized Least Squares
│   ├── glm.mojo               # Generalized Linear Models
│   ├── logistic.mojo          # Logistic regression
│   ├── results.mojo           # ModelResults container
│   └── ...
tests/
├── test_all.sh
├── test_distributions.mojo
├── test_stats.mojo
└── test_models.mojo
```

## Roadmap

### Phase 0 — Foundation (current)

- Set up repository structure, pixi configuration, CI
- Establish coding conventions and testing framework
- Implement special mathematical functions needed as building blocks:
  - Gamma function, log-gamma, incomplete gamma
  - Beta function, incomplete beta (regularized)
  - Error function (erf, erfc)

### Phase 1 — Core Distributions

- **Normal distribution**: PDF, CDF, PPF, random sampling
- **Student's t distribution**: PDF, CDF, PPF
- **Chi-squared distribution**: PDF, CDF, PPF
- **F-distribution**: PDF, CDF, PPF
- Descriptive statistics: mean, variance, std, median, quantiles, skewness, kurtosis

### Phase 2 — Hypothesis Testing

- One-sample, two-sample, paired t-tests
- Chi-squared goodness-of-fit and independence tests
- Kolmogorov-Smirnov test
- Pearson, Spearman, Kendall correlation with p-values
- ANOVA (one-way)

### Phase 3 — OLS Regression

- OLS model fitting via normal equations and QR decomposition
- Coefficient standard errors, t-statistics, p-values
- R², adjusted R², F-statistic
- `ModelResults` summary output (similar to `statsmodels.summary()`)
- Prediction and confidence/prediction intervals

### Phase 4 — Generalized Linear Models

- GLM framework with link functions (identity, logit, log, probit)
- Logistic regression (binomial family)
- Poisson regression
- IRLS (Iteratively Reweighted Least Squares) fitting algorithm
- Deviance, AIC, BIC

### Phase 5 — Extended Distributions & Models

- More distributions: Beta, Gamma, Exponential, Uniform, Binomial, Poisson, Negative Binomial, Weibull, Log-normal, etc.
- Weighted Least Squares (WLS), Generalized Least Squares (GLS)
- Robust regression (M-estimators)
- Regularized regression (Ridge, Lasso, Elastic Net)

### Phase 6 — Advanced Topics

- Time series basics (ACF, PACF, stationarity tests)
- Survival analysis (Kaplan-Meier, Cox proportional hazards)
- Nonparametric methods (kernel density estimation, Mann-Whitney U test)
- Bootstrap and permutation tests

## License

This repository and its contributions are licensed under the Apache License v2.0.
