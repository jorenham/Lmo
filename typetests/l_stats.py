# pyright: reportUnusedCallResult=false
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

import lmo


X = [0.14543334, 2.17509751, 0.60844233, 1.47809552, -1.32510269, 1.0979731]
XX = [X, X]

_ArrF8: TypeAlias = npt.NDArray[np.float64]

# default
assert_type(lmo.l_stats(X), _ArrF8)
assert_type(lmo.l_stats(np.array(X, dtype=np.float32)), _ArrF8)
assert_type(lmo.l_stats(np.array(X, dtype=np.int32)), _ArrF8)
assert_type(lmo.l_stats(XX), _ArrF8)
assert_type(lmo.l_stats(np.array(XX)), _ArrF8)

# num
assert_type(lmo.l_stats(X, num=3), _ArrF8)
assert_type(lmo.l_stats(X, 0, 3), _ArrF8)

# axis
assert_type(lmo.l_stats(XX, axis=0), _ArrF8)
assert_type(lmo.l_stats(np.array(XX), axis=0), _ArrF8)
