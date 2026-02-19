# StaMojo Roadmap

StaMojo is organized into two major parts:

| Part                                          | Scope                                                                                   | External dependencies                                                                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Part I — Statistical Computing Foundation** | Special functions, distributions, descriptive statistics, hypothesis tests, correlation | **None** (Mojo stdlib only)                                                                                                             |
| **Part II — Statistical Modeling**            | OLS, GLM, logistic regression, time series, survival analysis                           | [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo), [MatMojo](https://github.com/mojomath/matmojo) (linear algebra) |

---

## Part I — Statistical Computing Foundation

*Corresponds to `scipy.stats` in the Python ecosystem.*

All functions in Part I operate on scalar `Float64` values or `List[Float64]` and depend only on the Mojo standard library. No external packages are required.

### Phase 0 — Special Functions ✓

- Gamma function, log-gamma, regularized incomplete gamma (`gammainc`, `gammaincc`)
- Beta function, log-beta, regularized incomplete beta (`beta`, `lbeta`, `betainc`)
- Inverse error function, normal quantile (`erfinv`, `ndtri`)

### Phase 1 — Core Distributions & Descriptive Statistics ✓

**Distributions** (each provides PDF, log-PDF, CDF, SF, PPF, and random sampling):

- Normal (Gaussian)
- Student's t
- Chi-squared
- F-distribution

**Descriptive statistics:**

- `mean`, `variance`, `std`, `median`, `quantile`
- `skewness`, `kurtosis`
- `data_min`, `data_max`

### Phase 2 — Hypothesis Testing & Correlation ✓

**Hypothesis tests:**

- One-sample t-test (`ttest_1samp`)
- Independent two-sample t-test, Welch's (`ttest_ind`)
- Paired t-test (`ttest_rel`)
- Chi-squared goodness-of-fit (`chi2_gof`)
- Chi-squared test of independence (`chi2_ind`)
- Kolmogorov-Smirnov one-sample test (`ks_1samp`)
- One-way ANOVA F-test (`f_oneway`)

**Correlation:**

- Pearson (`pearsonr`)
- Spearman (`spearmanr`)
- Kendall's tau (`kendalltau`)

### Phase 3 — Extended Distributions

More continuous distributions:

- Exponential
- Gamma (generalizes Chi-squared and Exponential)
- Beta (bounded on \[0, 1\])
- Uniform
- Lognormal
- Weibull

Discrete distributions:

- Binomial
- Poisson

### Phase 4 — Extended Statistical Tests & Utilities

Additional hypothesis tests:

- Shapiro-Wilk normality test
- Mann-Whitney U test (nonparametric two-sample)
- Wilcoxon signed-rank test (nonparametric paired)
- Levene's test (homogeneity of variances)
- Kruskal-Wallis test (nonparametric one-way ANOVA)

Additional descriptive utilities:

- `mode`, `iqr`, `sem` (standard error of mean)
- `zscore` (standardization)
- `cov` (sample covariance)

---

## Part II — Statistical Modeling

*Corresponds to `statsmodels` in the Python ecosystem.*

Part II requires n-dimensional array and linear algebra support from [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) and [MatMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/MatMojo) (or equivalent). Development will begin once these upstream dependencies stabilize on a compatible Mojo release.

### Phase 5 — OLS Regression

- OLS model fitting via normal equations and QR decomposition
- Coefficient standard errors, t-statistics, p-values
- R², adjusted R², F-statistic
- `ModelResults` summary output (similar to `statsmodels.summary()`)
- Prediction and confidence/prediction intervals

### Phase 6 — Generalized Linear Models

- GLM framework with link functions (identity, logit, log, probit)
- Logistic regression (binomial family)
- Poisson regression
- IRLS (Iteratively Reweighted Least Squares) fitting algorithm
- Deviance, AIC, BIC

### Phase 7 — Extended Models

- Weighted Least Squares (WLS), Generalized Least Squares (GLS)
- Robust regression (M-estimators)
- Regularized regression (Ridge, Lasso, Elastic Net)

### Phase 8 — Advanced Topics

- Time series basics (ACF, PACF, stationarity tests)
- Survival analysis (Kaplan-Meier, Cox proportional hazards)
- Nonparametric methods (kernel density estimation)
- Bootstrap and permutation tests
