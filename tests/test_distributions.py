import numpy as np

import pytest

from lmo.distributions import wakeby


@pytest.mark.parametrize('scale', [1, 1/137, 42])
@pytest.mark.parametrize('loc', [0, 1, -1])
@pytest.mark.parametrize('b, d, f', [
    (1, 0, 1),
    (0, 0, 1),
    (0, 0.9, 0),
    (0, 3.14, 0),
    (0.9, 0, 1),
    (3.14, 0, 1),
    (1, 1, .5),
    (1, 1, .9),
    (.8, 1.2, .4),
    (.3, .3, .7),
    (4, -2, .69),
    (1, -0.99, .5),
    (-2, 3, .42),
])
def test_wakeby_cdf(b: float, d: float, f: float, loc: float, scale: float):
    X = wakeby(b, d, f, loc, scale)

    assert X.cdf(X.support()[0]) == 0
    assert X.ppf(0) == X.support()[0]
    assert X.ppf(1) == X.support()[1]

    q = np.linspace(0.005, 0.995, 100)
    x = X.ppf(q)
    q2 = X.cdf(x)
    assert np.allclose(q2, q)

