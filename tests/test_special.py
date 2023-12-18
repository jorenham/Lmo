# pyright: reportUnknownVariableType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
import pytest

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.testing import assert_allclose
from lmo.special import fourier_jacobi
from scipy.special import eval_jacobi


X = np.linspace(-1, 1, num=21, dtype=np.float64)
C_EXAMPLES = [
    [0],
    [1],
    [-1],
    [512],
    [1, 0],
    [0, 1],
    [1, .5],
    [1, -.5],
    [0, 0, 1],
    [1, .5, .25],
    [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
]

@pytest.mark.parametrize('c', C_EXAMPLES)
def test_fourier_legendre(c: list[float]):
    y_expect = legval(X, c)
    y_true = fourier_jacobi(X, c, 0, 0)

    assert_allclose(y_true, y_expect)


@pytest.mark.parametrize('a', [0, 1, 1/2, -1/2, 42])
@pytest.mark.parametrize('b', [0, 1, 1/2, -1/2, 42])
@pytest.mark.parametrize('c', C_EXAMPLES)
def test_fourier_jacobi(a: float, b: float, c: list[float]):
    y_expect = np.sum([
        cn * eval_jacobi(n, a, b, X)
        for n, cn in enumerate(c)
    ], axis=0)
    y_true = fourier_jacobi(X, c, a, b)

    assert_allclose(y_true, y_expect, atol=1e-15)
