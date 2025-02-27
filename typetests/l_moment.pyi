# pyright: reportUnusedCallResult=false, reportInvalidStubStatement=false
from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

import lmo

_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]

X: list[float]
XX: list[list[float]]

XX_np: onp.Array2D[np.float64]

# default
assert_type(lmo.l_moment(X, 2), float)
assert_type(lmo.l_moment(np.array(X), 2), float)
assert_type(lmo.l_moment(np.array(X, dtype=np.float32), 2), float)
assert_type(lmo.l_moment(np.array(X, dtype=np.int32), 2), float)
assert_type(lmo.l_moment(X, np.intp(2)), float)
assert_type(lmo.l_moment(X, np.uint8(2)), float)
assert_type(lmo.l_moment(XX, np.uint8(2)), float)
assert_type(lmo.l_moment(np.array(XX), np.uint8(2)), float)

# trim
assert_type(lmo.l_moment(X, 2, 0), float)
assert_type(lmo.l_moment(X, 2, 1), float)
assert_type(lmo.l_moment(X, 2, trim=1), float)
assert_type(lmo.l_moment(X, 2, trim=(1, 1)), float)
assert_type(lmo.l_moment(X, 2, trim=(0.5, 0.5)), float)
assert_type(lmo.l_moment(X, 2, trim=(1, 0.5)), float)
assert_type(lmo.l_moment(X, 2, trim=(0.5, 1)), float)

# vectorized r
assert_type(lmo.l_moment(X, [1, 2, 3, 4]), _Float1D)
assert_type(lmo.l_moment(X, (1, 2, 3, 4)), _Float1D)
assert_type(lmo.l_moment(X, np.arange(1, 5)), _Float1D)

# sctype
assert_type(lmo.l_moment(X, 2, dtype=np.float32), float)
assert_type(lmo.l_moment(X, 2, dtype=np.dtype(np.float16)), float)
assert_type(lmo.l_moment(X, [1, 2, 3, 4], dtype=np.float16), onp.Array1D[np.float16])

# axis
assert_type(lmo.l_moment(XX, 2, axis=0), _FloatND)
assert_type(lmo.l_moment(XX_np, 2, axis=0), _FloatND)
assert_type(lmo.l_moment(XX, 2, axis=0, dtype=np.float16), onp.ArrayND[np.float16])
