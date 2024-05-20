# pyright: reportUnusedCallResult=false
from typing import TypeAlias

import numpy as np

import lmo
from lmo.typing import np as lnpt
from lmo.typing.compat import assert_type


X = [0.14543334, 2.17509751, 0.60844233, 1.47809552, -1.32510269, 1.0979731]
XX = [X, X]

_VecF8: TypeAlias = lnpt.Array[tuple[int], np.float64]
_ArrayF8: TypeAlias = lnpt.Array[lnpt.AtLeast1D, np.float64]

# default
assert_type(lmo.l_stats(X), _VecF8)
assert_type(lmo.l_stats(np.array(X, dtype=np.float32)), _VecF8)
assert_type(lmo.l_stats(np.array(X, dtype=np.int32)), _VecF8)
assert_type(lmo.l_stats(XX), _VecF8)
assert_type(lmo.l_stats(np.array(XX)), _VecF8)

# axis
assert_type(lmo.l_stats(XX, axis=0), _ArrayF8)
assert_type(lmo.l_stats(np.array(XX), axis=0), _ArrayF8)
