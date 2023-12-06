import numpy as np

import pytest

from lmo.distributions import wakeby


@pytest.mark.parametrize('scale', [1, .5, 2])
@pytest.mark.parametrize('loc', [0, 1, -1])
@pytest.mark.parametrize('b, d, f', [
    (1, 0, 1),
    (0, 0, 1),
    (0, 0.9, 0),
    (0, 1.2, 0),
    (0.9, 0, 1),
    (1.2, 0, 1),
    (1, 1, .5),
    (1, 1, .9),
    (.8, 1.2, .4),
    (.3, .3, .7),
    (3, -2, .69),
    (1, -0.99, .5),
])
def test_wakeby(b: float, d: float, f: float, loc: float, scale: float):
    X = wakeby(b, d, f, loc, scale)

    assert X.cdf(X.support()[0]) == 0
    assert X.ppf(0) == X.support()[0]
    assert X.ppf(1) == X.support()[1]

    q = np.linspace(0, 1, 100, endpoint=False)
    x = X.ppf(q)
    q2 = X.cdf(x)
    assert np.allclose(q2, q)

    # quad_opts={} forces numerical evaluation
    l_stats_numerical = X.l_stats(quad_opts={})
    l_stats_exact = X.l_stats()
    assert np.allclose(l_stats_exact, l_stats_numerical, equal_nan=d >= 1)

    ll_stats_numerical = X.l_stats(quad_opts={}, trim=(0, 1))
    ll_stats_exact = X.l_stats(trim=(0, 1))
    assert np.allclose(ll_stats_exact, ll_stats_numerical)

    tl_stats_numerical = X.l_stats(quad_opts={}, trim=1)
    tl_stats_exact = X.l_stats(trim=1)
    assert np.allclose(tl_stats_exact, tl_stats_numerical)

    tll_stats_numerical = X.l_stats(quad_opts={}, trim=(1, 2))
    tll_stats_exact = X.l_stats(trim=(1, 2))
    assert np.allclose(tll_stats_exact, tll_stats_numerical)
