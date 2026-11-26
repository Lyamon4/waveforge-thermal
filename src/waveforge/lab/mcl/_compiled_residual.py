"""Optional fused FP64 residual. No reassociation, FMA contraction or fastmath."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, inline="always")
def _two_sum(a, b):
    high = a + b
    virtual = high - a
    return high, (a - (high - virtual)) + (b - virtual)


@njit(cache=True, fastmath=False, nogil=True)
def real_residual(values, indices, a_high, a_low, x, rhs):
    result = np.empty_like(rhs)
    for row in range(rhs.shape[0]):
        for column in range(rhs.shape[1]):
            high = rhs[row, column]
            low = 0.0
            for slot in range(values.shape[0]):
                a = -values[slot, row]
                ah = -a_high[slot, row]
                al = -a_low[slot, row]
                b = x[indices[slot, row], column]
                split = b * 134217729.0
                bh = split - (split - b)
                bl = b - bh
                product = a * b
                error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
                high, carry = _two_sum(high, product)
                low, carry2 = _two_sum(low, error)
                high, carry = _two_sum(high, carry + low)
                high, low = _two_sum(high, carry + carry2)
            result[row, column] = high + low
    return result
