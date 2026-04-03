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
