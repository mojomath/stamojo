# ===----------------------------------------------------------------------=== #
# Stamojo - A statistical computing library for Mojo
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""StaMojo: A statistical computing library for Mojo.

StaMojo provides statistical distributions, hypothesis testing, descriptive
statistics, and statistical modeling for the Mojo programming language.
It is inspired by scipy.stats and statsmodels in Python.

Part I (distributions, stats, special functions) is self-contained.
Part II (statistical models) will depend on NuMojo for linear algebra.
"""

from .distributions import *
from .stats import *
