# pyright: reportUnusedCallResult=false
from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

import lmo


_ArrayF8: TypeAlias = npt.NDArray[np.float64]

X = [0.14543334, 2.17509751, 0.60844233, 1.47809552, -1.32510269, 1.0979731]
XX = [X, X]

# default
assert_type(lmo.l_moment(X, 2), np.float64)
assert_type(lmo.l_moment(np.array(X), 2), np.float64)
assert_type(lmo.l_moment(np.array(X, dtype=np.float32), 2), np.float64)
assert_type(lmo.l_moment(np.array(X, dtype=np.int32), 2), np.float64)
assert_type(lmo.l_moment(X, np.intp(2)), np.float64)
assert_type(lmo.l_moment(X, np.uint8(2)), np.float64)
assert_type(lmo.l_moment(XX, np.uint8(2)), np.float64)
assert_type(lmo.l_moment(np.array(XX), np.uint8(2)), np.float64)

# trim
assert_type(lmo.l_moment(X, 2, 0), np.float64)
assert_type(lmo.l_moment(X, 2, 1), np.float64)
assert_type(lmo.l_moment(X, 2, trim=1), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(1, 1)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(.5, .5)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(1, .5)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(.5, 1)), np.float64)

# vectorized r
assert_type(lmo.l_moment(X, [1, 2, 3, 4]), _ArrayF8)
assert_type(lmo.l_moment(X, (1, 2, 3, 4)), _ArrayF8)
assert_type(lmo.l_moment(X, np.arange(1, 5)), _ArrayF8)

# sctype
assert_type(lmo.l_moment(X, 2, dtype=np.float32), np.float32)
assert_type(lmo.l_moment(X, 2, dtype=np.longdouble), np.longdouble)
assert_type(lmo.l_moment(X, 2, dtype=np.dtype(np.float16)), np.float16)
assert_type(lmo.l_moment(X, [1, 2, 3, 4], dtype=np.half), npt.NDArray[np.half])

# axis
assert_type(lmo.l_moment(XX, 2, axis=0), _ArrayF8)
assert_type(lmo.l_moment(np.array(XX), 2, axis=0), _ArrayF8)
assert_type(lmo.l_moment(XX, 2, axis=0, dtype=np.half), npt.NDArray[np.half])
