# pyright: reportUnusedCallResult=false, reportInvalidStubStatement=false
from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

import lmo

X: list[float]
XX: list[list[float]]

Half1D: TypeAlias = onp.Array1D[np.float16]
Float1D: TypeAlias = onp.Array1D[np.float64]
FloatND: TypeAlias = onp.Array[onp.AtLeast1D, np.float64]

# defaults
assert_type(lmo.l_stats(X), Float1D)
assert_type(lmo.l_stats(XX), Float1D)
assert_type(lmo.l_stats(np.asarray(X)), Float1D)
assert_type(lmo.l_stats(np.asarray(XX)), Float1D)
assert_type(lmo.l_stats(np.empty((1, 3, 3, 7), dtype=np.float32)), Float1D)
assert_type(lmo.l_stats(np.empty((1, 3, 3, 7), dtype=np.longlong)), Float1D)

# default + num
assert_type(lmo.l_stats(X, num=3), Float1D)
assert_type(lmo.l_stats(X, 0, 3), Float1D)

# defaults + axis
assert_type(lmo.l_stats(XX, axis=0), FloatND)
assert_type(lmo.l_stats(np.asarray(XX), axis=0), FloatND)

# defaults + dtype
assert_type(lmo.l_stats(X, dtype=np.float16), Half1D)
assert_type(lmo.l_stats(XX, dtype=np.float16), Half1D)
assert_type(lmo.l_stats(np.asarray(X, dtype=np.float32), dtype=np.float16), Half1D)
assert_type(lmo.l_stats(np.empty((4, 3), dtype=np.float32), dtype=np.float16), Half1D)
