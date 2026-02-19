# StaMojo <!-- omit in toc -->

A statistical computing library for [Mojo](https://www.modular.com/mojo), inspired by `scipy.stats` and `statsmodels` in Python.

**[Repository on GitHub»](https://github.com/mojomath/stamojo)**　|　**[Discord channel»](https://discord.gg/3rGH87uZTk)**

- [Overview](#overview)
  - [Why a separate library?](#why-a-separate-library)
- [Background](#background)
- [Installation](#installation)
- [Examples](#examples)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
  - [Phase 0 — Foundation ✓](#phase-0--foundation-)
  - [Phase 1 — Core Distributions ✓](#phase-1--core-distributions-)
  - [Phase 2 — Hypothesis Testing ✓](#phase-2--hypothesis-testing-)
  - [Phase 3 — OLS Regression (current)](#phase-3--ols-regression-current)
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

## Background

Due to my academic and professional background, I work extensively with hypothesis testing and regression models on a daily basis, and have been a long-time user of Stata and `statsmodels`. It has been two years since Mojo first appeared, and [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) now has its core functionality in place. Driven by my enthusiasm for Mojo, I felt it was time to start migrating some of my personal research projects to the Mojo ecosystem — and that is precisely how StaMojo was born.

The library is designed around two pillars:

1. **Hard math foundations** — special functions, probability distributions, descriptive statistics, and hypothesis tests.
2. **Regression models** — OLS, GLM, logistic regression, and related diagnostics.

At the moment I am still building out the project scaffolding and solidifying the core functionality. Because Mojo has not yet reached v1.0, breaking changes are frequent across compiler releases, so **pull requests are not accepted at this time**. If you have any suggestions, questions, or feedback, please feel free to open an [issue](https://github.com/mojomath/stamojo/issues), start a [discussion](https://github.com/mojomath/stamojo/discussions), or reach out on our [Discord channel](https://discord.gg/3rGH87uZTk). Thank you for your understanding!

## Installation

Stamojo will be published to the `modular-community` channel once it reaches a functional first release. During development, clone the repo and build locally:

```bash
git clone https://github.com/mojomath/stamojo.git
cd stamojo
pixi install
pixi run package
```

## Examples

The file [`examples/examples.mojo`](examples/examples.mojo) demonstrates every function implemented so far (Phases 0–2). Run it with:

```bash
mojo run -I src examples/examples.mojo
```

```mojo
from stamojo.special import gammainc, gammaincc, beta, lbeta, betainc, erfinv, ndtri
from stamojo.distributions import Normal, StudentT, ChiSquared, FDist
from stamojo.stats import (
    mean, variance, std, median, quantile, skewness, kurtosis,
    pearsonr, spearmanr, kendalltau,
    ttest_1samp, ttest_ind, ttest_rel,
    chi2_gof, chi2_ind, ks_1samp, f_oneway,
)


fn main() raises:
    # --- Special functions ---------------------------------------------------
    print("gammainc(1, 2) =", gammainc(1.0, 2.0))           # 0.8646647167628346
    print("gammaincc(1, 2) =", gammaincc(1.0, 2.0))         # 0.13533528323716537
    print("beta(2, 3)     =", beta(2.0, 3.0))               # 0.08333333333323925
    print("betainc(2, 3, 0.5) =", betainc(2.0, 3.0, 0.5))   # 0.6875000000000885
    print("erfinv(0.5)    =", erfinv(0.5))                  # 0.4769362762044701
    print("ndtri(0.975)   =", ndtri(0.975))                 # 1.9599639845400543

    # --- Distributions -------------------------------------------------------
    var n = Normal(0.0, 1.0)
    print("Normal(0,1).pdf(0)   =", n.pdf(0.0))             # 0.3989422804014327
    print("Normal(0,1).cdf(1.96)=", n.cdf(1.96))            # 0.9750021048517795
    print("Normal(0,1).ppf(0.975)=", n.ppf(0.975))          # 1.9599639845400543
    print("Normal(0,1).sf(1.96) =", n.sf(1.96))             # 0.02499789514822043

    var t = StudentT(10.0)
    print("StudentT(10).cdf(2.0)=", t.cdf(2.0))             # 0.9633059826444078
    print("StudentT(10).ppf(0.975)=", t.ppf(0.975))         # 2.2281388540534057

    var c = ChiSquared(5.0)
    print("ChiSquared(5).cdf(11.07)=", c.cdf(11.07))        # 0.9499903814759155

    var f = FDist(5.0, 10.0)
    print("FDist(5,10).cdf(3.33)=", f.cdf(3.33))            # 0.9501687242532277

    # --- Descriptive statistics ----------------------------------------------
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print("mean    =", mean(data))              # 5.0
    print("variance=", variance(data, ddof=0))  # 4.0
    print("std     =", std(data, ddof=0))       # 2.0
    print("median  =", median(data))            # 4.5
    print("Q(0.25) =", quantile(data, 0.25))    # 4.0
    print("skewness=", skewness(data))          # 0.8184875533567997
    print("kurtosis=", kurtosis(data))          # 0.940625

    # --- Correlation ---------------------------------------------------------
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var y: List[Float64] = [2.1, 3.8, 6.0, 7.9, 10.1]
    var pr = pearsonr(x, y)
    print("pearsonr  r=", pr[0], " p=", pr[1])    # 0.9991718425080479, 2.8605484175113625e-05
    var sr = spearmanr(x, y)
    print("spearmanr ρ=", sr[0], " p=", sr[1])    # 1.0, 0.0
    var kt = kendalltau(x, y)
    print("kendalltau τ=", kt[0], " p=", kt[1])   # 1.0, 0.014305878435429659

    # --- Hypothesis tests ----------------------------------------------------
    var sample: List[Float64] = [5.1, 4.8, 5.3, 5.0, 4.9, 5.2]
    var res = ttest_1samp(sample, 5.0)
    print("ttest_1samp  t=", res[0], " p=", res[1])   # 0.654653670707975, 0.5416045608507769

    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var b: List[Float64] = [4.0, 5.0, 6.0, 7.0, 8.0]
    var res2 = ttest_ind(a, b)
    print("ttest_ind    t=", res2[0], " p=", res2[1])  # -3.0, 0.0170716812337895

    var obs: List[Float64] = [16.0, 18.0, 16.0, 14.0, 12.0, 14.0]
    var exp: List[Float64] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    var res3 = chi2_gof(obs, exp)
    print("chi2_gof   χ²=", res3[0], " p=", res3[1])  # 1.4666666666666666, 0.9168841203537823

    var g1: List[Float64] = [3.0, 4.0, 5.0]
    var g2: List[Float64] = [6.0, 7.0, 8.0]
    var g3: List[Float64] = [9.0, 10.0, 11.0]
    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)
    var res4 = f_oneway(groups)
    print("f_oneway   F=", res4[0], " p=", res4[1])   # 27.0, 0.0010000000005315757
```

## Architecture

```txt
src/stamojo/
├── __init__.mojo              # Package root (re-exports distributions & stats)
├── prelude.mojo               # Convenient re-exports
├── special/                   # Special mathematical functions (cf. scipy.special)
│   ├── __init__.mojo
│   ├── _gamma.mojo            # gammainc, gammaincc
│   ├── _beta.mojo             # beta, lbeta, betainc
│   └── _erf.mojo              # erfinv, ndtri
├── distributions/             # Probability distributions
│   ├── __init__.mojo
│   ├── normal.mojo            # Normal (Gaussian) — PDF, logPDF, CDF, SF, PPF, rvs
│   ├── t.mojo                 # Student's t
│   ├── chi2.mojo              # Chi-squared
│   └── f.mojo                 # F-distribution
├── stats/                     # Descriptive stats & hypothesis tests
│   ├── __init__.mojo
│   ├── descriptive.mojo       # mean, variance, std, median, quantile, skewness, kurtosis
│   ├── correlation.mojo       # pearsonr, spearmanr, kendalltau
│   └── tests.mojo             # ttest_1samp, ttest_ind, ttest_rel, chi2_gof, chi2_ind, ks_1samp, f_oneway
└── models/                    # Statistical models (planned)
    ├── __init__.mojo
    └── ols.mojo               # Ordinary Least Squares (stub)
tests/
├── test_all.sh                # Run all test suites
├── test_special.mojo          # 15 tests — special functions
├── test_distributions.mojo    # 20 tests — Normal, t, χ², F
├── test_stats.mojo            # 10 tests — descriptive statistics
└── test_hypothesis.mojo       # 22 tests — hypothesis tests, correlation, ANOVA
```

## Roadmap

### Phase 0 — Foundation ✓

- Set up repository structure, pixi configuration, CI
- Establish coding conventions and testing framework
- Implement special mathematical functions needed as building blocks:
  - Gamma function, log-gamma, regularized incomplete gamma (`gammainc`, `gammaincc`)
  - Beta function, log-beta, regularized incomplete beta (`beta`, `lbeta`, `betainc`)
  - Inverse error function, normal quantile (`erfinv`, `ndtri`)

### Phase 1 — Core Distributions ✓

- **Normal distribution**: PDF, log-PDF, CDF, SF, PPF, entropy, random sampling (Box-Muller)
- **Student's t distribution**: PDF, log-PDF, CDF, SF, PPF (Newton-Raphson + bisection)
- **Chi-squared distribution**: PDF, log-PDF, CDF, SF, PPF (Wilson-Hilferty + Newton-Raphson)
- **F-distribution**: PDF, log-PDF, CDF, SF, PPF
- Descriptive statistics: `mean`, `variance`, `std`, `median`, `quantile`, `skewness`, `kurtosis`, `data_min`, `data_max`

### Phase 2 — Hypothesis Testing ✓

- One-sample, two-sample (Welch's), and paired t-tests
- Chi-squared goodness-of-fit and test of independence
- Kolmogorov-Smirnov test (vs standard normal)
- Pearson, Spearman, Kendall correlation with p-values
- One-way ANOVA (F-test)

### Phase 3 — OLS Regression (current)

This phase relies on NuMojo and MatMojo for linear algebra operations.

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
