# pyright: reportUnusedCallResult=false
from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

import lmo


_ArrayF8: TypeAlias = npt.NDArray[np.float64]

X = [
    [1.9517689, -0.39353141, -0.46680832, -0.43176034, 0.03754792, -0.2559433],
    [-0.18679035, -0.30584785, -1.32954, 0.27871746, -0.19124341, -2.1717801],
]


assert_type(lmo.l_comoment(X, 2), _ArrayF8)
assert_type(lmo.l_comoment(np.array(X), 2), _ArrayF8)
assert_type(lmo.l_comoment(np.array(X).T, 2, rowvar=False), _ArrayF8)
assert_type(lmo.l_comoment(X, 2, dtype=np.float16), npt.NDArray[np.float16])
