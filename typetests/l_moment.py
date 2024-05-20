# pyright: reportUnusedCallResult=false
import numpy as np
import numpy.typing as npt

import lmo
from lmo.typing.compat import assert_type


X = [0.14543334, 2.17509751, 0.60844233, 1.47809552, -1.32510269, 1.0979731]

# default
assert_type(lmo.l_moment(X, 2), np.float64)
assert_type(lmo.l_moment(np.asarray(X), 2), np.float64)
assert_type(lmo.l_moment(np.asarray(X, dtype=np.float32), 2), np.float64)
assert_type(lmo.l_moment(X, np.intp(2)), np.float64)
assert_type(lmo.l_moment(X, np.uint8(2)), np.float64)

# trim
assert_type(lmo.l_moment(X, 2, 0), np.float64)
assert_type(lmo.l_moment(X, 2, 1), np.float64)
assert_type(lmo.l_moment(X, 2, trim=1), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(1, 1)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(.5, .5)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(1, .5)), np.float64)
assert_type(lmo.l_moment(X, 2, trim=(.5, 1)), np.float64)

# sctype
assert_type(lmo.l_moment(X, 2, dtype=np.float32), np.float32)
assert_type(lmo.l_moment(X, 2, dtype=np.longdouble), np.longdouble)
assert_type(lmo.l_moment(X, 2, dtype=np.dtype(np.float16)), np.float16)

# vectorized r
assert_type(lmo.l_moment(X, [2]), npt.NDArray[np.float64])
assert_type(lmo.l_moment(X, (2,)), npt.NDArray[np.float64])
assert_type(lmo.l_moment(X, [1, 2, 3, 4]), npt.NDArray[np.float64])
assert_type(lmo.l_moment(X, (1, 2, 3, 4)), npt.NDArray[np.float64])
assert_type(lmo.l_moment(X, np.arange(1, 5)), npt.NDArray[np.float64])
