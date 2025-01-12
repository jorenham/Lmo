from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import distributions
from scipy.stats._distr_params import distcont

from lmo.contrib.scipy_stats import l_moments, l_rv_frozen

too_slow = {
    "dpareto_lognorm",
    "geninvgauss",
    "kappa4",
    "ksone",
    "kstwo",
    "levy_stable",
    "norminvgauss",
    "pearson3",
    "recipinvgauss",
    "skewnorm",
    "studentized_range",
    "vonmises",
}
rv_cont = [
    getattr(distributions, name)(*theta)
    for name, theta in distcont
    if name not in too_slow
]


@pytest.mark.parametrize("rv", rv_cont, ids=lambda rv: rv.dist.name)
def test_rv_cont(rv: distributions.rv_frozen[Any, Any]) -> None:
    a, b = rv.support()
    s = 2 * (not np.isfinite(a))
    t = 2 * (not np.isfinite(b))

    l_old = cast("l_rv_frozen", rv).l_stats(trim=(s, t))
    l_new = l_moments(rv, trim=(s, t))

    assert_allclose(l_old, l_new)
