# ===----------------------------------------------------------------------=== #
# StaMojo - Special mathematical functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Special mathematical functions.

This subpackage provides special functions that are not available in the Mojo
standard library but are needed for statistical distributions and may also be
useful to end users directly.  It mirrors the scope of `scipy.special`.

Functions provided:

- Regularized incomplete gamma function (lower and upper)
- Regularized incomplete beta function
- Inverse error function (erfinv)
- Log-beta function
- Beta function

The Mojo standard library already provides erf, erfc, gamma, and lgamma,
so we do not reimplement those here.

The modules of the subpackages are named with a leading underscore 
(e.g., `_gamma`) to avoid conflicts with the standard library functions.
"""

from ._gamma import gammainc, gammaincc
from ._beta import beta, lbeta, betainc
from ._erf import erfinv, ndtri
