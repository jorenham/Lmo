# pyright: reportUnusedCallResult=false, reportInvalidStubStatement=false
from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

import lmo

_Float2D: TypeAlias = onp.Array2D[np.float64]

X: list[list[float]]

assert_type(lmo.l_comoment(X, 2), _Float2D)
assert_type(lmo.l_comoment(np.array(X), 2), _Float2D)
assert_type(lmo.l_comoment(np.array(X).T, 2, rowvar=False), _Float2D)
assert_type(lmo.l_comoment(X, 2, dtype=np.float16), onp.Array2D[np.float16])
