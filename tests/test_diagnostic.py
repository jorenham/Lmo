import numpy as np
from pytest import approx

from lmo.diagnostic import l_moment_bounds


def test_l_moment_bounds_00():
    assert np.isposinf(l_moment_bounds(1))
    assert l_moment_bounds(2) == approx(1 / np.sqrt(3))
    assert l_moment_bounds(3) == approx(1 / np.sqrt(5))
    assert l_moment_bounds(4) == approx(1 / np.sqrt(7))
    assert l_moment_bounds(42) == approx(1 / np.sqrt(83))


def test_l_moment_bounds_scale():
    assert l_moment_bounds(42, scale=69) == approx(l_moment_bounds(42) * 69)


def test_l_moment_bounds_vectorized():
    bounds = l_moment_bounds([1, 2, 42])
    assert np.isposinf(bounds[0])
    assert bounds[1] == approx(1 / np.sqrt(3))
    assert bounds[-1] == approx(1 / np.sqrt(83))

