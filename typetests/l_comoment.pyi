# pyright: reportUnusedCallResult=false, reportInvalidStubStatement=false
from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

import lmo

_Float2ND: TypeAlias = onp.ArrayND[np.float64]

X: list[list[float]]

assert_type(lmo.l_comoment(X, 2), _Float2ND)
assert_type(lmo.l_comoment(np.array(X), 2), _Float2ND)
assert_type(lmo.l_comoment(np.array(X).T, 2, rowvar=False), _Float2ND)
assert_type(lmo.l_comoment(X, 2, dtype=np.float16), onp.ArrayND[np.float16])
