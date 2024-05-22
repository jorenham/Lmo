# pyright: reportUnusedCallResult=false
from typing import Any, TypeAlias

import numpy as np

import lmo
from lmo.typing import np as lnpt
from lmo.typing.compat import assert_type


_ArrayF8: TypeAlias = lnpt.Array[Any, np.float64]

X = [
    [1.9517689, -0.39353141, -0.46680832, -0.43176034, 0.03754792, -0.2559433],
    [-0.18679035, -0.30584785, -1.32954, 0.27871746, -0.19124341, -2.1717801],
]


assert_type(lmo.l_comoment(X, 2), _ArrayF8)
assert_type(lmo.l_comoment(np.array(X), 2), _ArrayF8)
assert_type(lmo.l_comoment(np.array(X).T, 2, rowvar=False), _ArrayF8)
assert_type(lmo.l_comoment(X, 2, dtype=np.half), lnpt.Array[Any, np.half])
