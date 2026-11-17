# ===----------------------------------------------------------------------=== #
# Stamojo - Distributions - Shared mathematical constants
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Shared mathematical constants used across distribution implementations.

These are pure mathematical constants (π, √2, Euler-Mascheroni, ...) that
recur identically in many distributions' PDF/CDF/entropy formulas. Numerical
tolerances like `_EPS`/`_MAX_ITER` are intentionally *not* centralized here,
since their appropriate values differ per algorithm.
"""

comptime PI = 3.1415926535897932384626433832795029
comptime TWO_PI = 6.283185307179586476925286766559006
comptime SQRT2 = 1.4142135623730950488016887242096981
comptime SQRT3 = 1.7320508075688772935274463415058724
comptime INV_SQRT_2PI = 0.3989422804014326779399460599343819
comptime LN_SQRT_2PI = 0.9189385332046727417803297364056176
comptime SQRT_2_OVER_PI = 0.7978845608028653558798921198687637
comptime EULER_MASCHERONI = 0.5772156649015328606065120900824024
