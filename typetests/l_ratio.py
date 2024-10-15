# pyright: reportUnusedCallResult=false
from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

import lmo

X = [0.14543334, 2.17509751, 0.60844233, 1.47809552, -1.32510269, 1.0979731]

_ArrF8: TypeAlias = npt.NDArray[np.float64]

# default
assert_type(lmo.l_ratio(X, 4, 2), np.float64)
assert_type(lmo.l_ratio(np.array(X), 4, 2), np.float64)
assert_type(lmo.l_ratio(np.array(X, dtype=np.float32), 4, 2), np.float64)
assert_type(lmo.l_ratio(np.array(X, dtype=np.int32), 4, 2), np.float64)
assert_type(lmo.l_ratio(X, np.int16(4), 2), np.float64)
assert_type(lmo.l_ratio(X, 4, np.uint8(2)), np.float64)
assert_type(lmo.l_ratio(X, np.int16(4), np.uint8(2)), np.float64)

# trim
assert_type(lmo.l_ratio(X, 4, 2, 0), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, 1), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, trim=1), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, trim=(1, 1)), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, trim=(.5, .5)), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, trim=(1, .5)), np.float64)
assert_type(lmo.l_ratio(X, 4, 2, trim=(.5, 1)), np.float64)

# sctype
assert_type(lmo.l_ratio(X, 4, 2, dtype=np.float32), np.float32)
assert_type(lmo.l_ratio(X, 4, 2, dtype=np.longdouble), np.longdouble)
assert_type(lmo.l_ratio(X, 4, 2, dtype=np.dtype(np.float16)), np.float16)

# vectorized r
assert_type(lmo.l_ratio(X, [3, 4], 2), _ArrF8)
assert_type(lmo.l_ratio(X, np.array([3, 4]), 2), _ArrF8)
assert_type(lmo.l_ratio(X, [1, 2, 3, 4], [0, 0, 2, 2]), _ArrF8)
assert_type(lmo.l_ratio(X, np.array([1, 2, 3, 4]), [0, 0, 2, 2]), _ArrF8)
assert_type(lmo.l_ratio(X, [1, 2, 3, 4], np.array([0, 0, 2, 2])), _ArrF8)
assert_type(lmo.l_ratio(X, 3, [0, 2]), _ArrF8)
assert_type(lmo.l_ratio(X, 3, np.array([0, 2])), _ArrF8)
