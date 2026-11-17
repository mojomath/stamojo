# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions subpackage
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Statistical probability distributions.

This subpackage provides continuous and discrete probability distributions,
including PDF/PMF, CDF, quantile (PPF), and random variate generation.

Distributions provided:
- `Normal`        — Normal (Gaussian) distribution
- `StudentT`      — Student's t-distribution
- `ChiSquared`    — Chi-squared distribution
- `FDist`         — F-distribution (Fisher-Snedecor)
- `Exponential`   — Exponential distribution
- `Binomial`      — Binomial distribution
- `Gamma`         — Gamma distribution
- `Beta`          — Beta distribution
- `Poisson`       — Poisson distribution
- `Uniform`       — Continuous uniform distribution
- `LogNormal`     — Lognormal distribution
- `Weibull`       — Weibull distribution
- `Cauchy`        — Cauchy distribution
- `Laplace`       — Laplace (double exponential) distribution
- `Logistic`      — Logistic distribution
- `Pareto`        — Pareto distribution
- `Rayleigh`      — Rayleigh distribution
- `HalfNormal`    — Half-normal distribution
- `HalfCauchy`    — Half-Cauchy distribution
- `Gumbel`        — Gumbel (extreme value, maximum) distribution
- `Triangular`    — Triangular distribution
- `VonMises`      — Von Mises (circular) distribution
- `TruncNorm`     — Truncated normal distribution
"""

from .normal import Normal
from .t import StudentT
from .chi2 import ChiSquared
from .f import FDist
from .exponential import Exponential
from .binomial import Binomial
from .gamma import Gamma
from .beta import Beta
from .poisson import Poisson
from .uniform import Uniform
from .lognormal import LogNormal
from .weibull import Weibull
from .cauchy import Cauchy
from .laplace import Laplace
from .logistic import Logistic
from .pareto import Pareto
from .rayleigh import Rayleigh
from .halfnormal import HalfNormal
from .halfcauchy import HalfCauchy
from .gumbel import Gumbel
from .triangular import Triangular
from .vonmises import VonMises
from .truncnorm import TruncNorm
