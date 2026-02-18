# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions subpackage
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Statistical probability distributions.

This subpackage provides continuous and discrete probability distributions,
including PDF/PMF, CDF, quantile (PPF), and random variate generation.

Distributions provided:
- `Normal`      — Normal (Gaussian) distribution
- `StudentT`    — Student's t-distribution
- `ChiSquared`  — Chi-squared distribution
- `FDist`       — F-distribution (Fisher-Snedecor)
"""

from .normal import Normal
from .t import StudentT
from .chi2 import ChiSquared
from .f import FDist
