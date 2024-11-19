# pyright: reportUnusedCallResult=false, reportInvalidStubStatement=false
from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

import lmo

X: list[float]
XX: list[list[float]]

FloatND: TypeAlias = onp.ArrayND[np.float64]

# default
assert_type(lmo.l_stats(X), FloatND)
assert_type(lmo.l_stats(np.array(X, dtype=np.float32)), FloatND)
assert_type(lmo.l_stats(np.array(X, dtype=np.int32)), FloatND)
assert_type(lmo.l_stats(XX), FloatND)
assert_type(lmo.l_stats(np.array(XX)), FloatND)

# num
assert_type(lmo.l_stats(X, num=3), FloatND)
assert_type(lmo.l_stats(X, 0, 3), FloatND)

# axis
assert_type(lmo.l_stats(XX, axis=0), FloatND)
assert_type(lmo.l_stats(np.array(XX), axis=0), FloatND)
