# ===----------------------------------------------------------------------=== #
# StaMojo - Internal special mathematical functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Internal special mathematical functions.

This subpackage provides special functions that are not available in the Mojo
standard library but are required as building blocks for statistical
distributions. These include:

- Regularized incomplete gamma function (lower and upper)
- Regularized incomplete beta function
- Inverse error function (erfinv)
- Log-beta function
- Beta function

The Mojo standard library already provides erf, erfc, gamma, and lgamma,
so we do not reimplement those here.
"""

from ._gamma import gammainc, gammaincc
from ._beta import beta, lbeta, betainc
from ._erf import erfinv, ndtri
