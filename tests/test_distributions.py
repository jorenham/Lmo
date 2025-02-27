import functools
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose as _assert_allclose
from scipy.stats import distributions
from scipy.stats.distributions import tukeylambda, uniform

import lmo.typing as lmt
from lmo.contrib.scipy_stats import l_rv_generic
from lmo.distributions import genlambda, kumaraswamy, l_poly, wakeby

Q = np.linspace(1 / 100, 1, 99, endpoint=False)

assert_allclose = functools.partial(_assert_allclose, atol=1e-9)


@pytest.mark.parametrize(
    "trim",
    [0, 1, (0, 1), (1, 0), (13, 17), (2 / 3, 3 / 4)],
)
def test_l_poly_eq_uniform(trim: lmt.ToTrim):
    p0 = x0 = np.linspace(0, 1)

    X = cast("Any", uniform())
    X_hat = l_poly(X.l_moment([1, 2], trim=trim), trim=trim)

    t4 = X.l_stats(trim=trim)
    t4_hat = X_hat.l_stats(trim=trim)
    assert_allclose(t4_hat, t4)

    mvsk = X.stats(moments="mvsk")
    mvsk_hat = X_hat.stats(moments="mvsk")
    assert_allclose(mvsk_hat, mvsk)

    x = X.ppf(p0)
    x_hat = X_hat.ppf(p0)
    assert_allclose(x_hat, x)

    F = X.cdf(p0)
    F_hat = X_hat.cdf(p0)
    assert_allclose(F_hat, F)

    f = X.pdf(x0)
    f_hat = X_hat.pdf(x0)
    assert_allclose(f_hat, f)

    H = X.entropy()
    H_hat = X_hat.entropy()
    assert_allclose(H_hat, H)


@pytest.mark.parametrize(
    ("b", "d", "f"),
    [
        (1, 0, 1),
        (0, 0, 1),
        (0, 0.9, 0),
        (0, 1.2, 0),
        (0.9, 0, 1),
        (1.2, 0, 1),
        (1, 1, 0.5),
        (1, 1, 0.9),
        (0.8, 1.2, 0.4),
        (0.3, 0.3, 0.7),
        (3, -2, 0.69),
        (1, -0.9, 0.5),
    ],
)
def test_wakeby(b: float, d: float, f: float):
    X = cast("Any", wakeby(b, d, f))

    assert X.cdf(X.support()[0]) == 0
    assert X.ppf(0) == X.support()[0]
    assert X.ppf(1) == X.support()[1]

    x = X.ppf(Q)
    q2 = X.cdf(x)
    assert_allclose(q2, Q)

    # quad_opts={} forces numerical evaluation
    l_stats_quad = X.l_stats(quad_opts={})
    l_stats_theo = X.l_stats()
    assert_allclose(l_stats_theo, l_stats_quad, equal_nan=d >= 1)

    ll_stats_quad = X.l_stats(quad_opts={}, trim=(0, 1))
    ll_stats_theo = X.l_stats(trim=(0, 1))
    assert_allclose(ll_stats_theo, ll_stats_quad)

    tl_stats_quad = X.l_stats(quad_opts={}, trim=1)
    tl_stats_theo = X.l_stats(trim=1)
    assert_allclose(tl_stats_theo, tl_stats_quad)

    tll_stats_quad = X.l_stats(quad_opts={}, trim=(1, 2))
    tll_stats_theo = X.l_stats(trim=(1, 2))
    assert_allclose(tll_stats_theo, tll_stats_quad)


@pytest.mark.parametrize("lam", [0, 0.14, 1, -1])
def test_genlambda_tukeylamba(lam: float):
    X0 = cast("Any", tukeylambda(lam))
    X = cast("Any", genlambda(lam, lam, 0))

    x0 = X0.ppf(Q)
    x = X.ppf(Q)
    assert x[0] >= X.support()[0]
    assert x[-1] <= X.support()[1]
    assert_allclose(x, x0)

    pp = cast(
        "npt.NDArray[np.float64]",
        np.linspace(X0.ppf(0.05), X0.ppf(0.95), 100),
    )
    u0 = X0.cdf(pp)
    u = X.cdf(pp)
    assert_allclose(u, u0)

    # the `scipy.statstukeylambda` implementation kinda sucks,,,
    with np.errstate(divide="ignore"):
        du0 = X0.pdf(pp)

    du = X.pdf(pp)
    assert_allclose(du, du0)

    s0 = X0.var()
    s = X.var()
    assert_allclose(s, s0, equal_nan=True)

    h0 = X0.entropy()
    h = X.entropy()
    assert_allclose(h, h0)

    tl_tau0 = X0.l_stats(trim=1)
    tl_tau = X.l_stats(trim=1)
    assert_allclose(tl_tau, tl_tau0)


@pytest.mark.parametrize("f", [-0.95, 0, 0.95], ids="f={}".format)
@pytest.mark.parametrize("d", [-1.95, 0, 1.95], ids="d={}".format)
@pytest.mark.parametrize("b", [-1.95, 0, 1.95], ids="b={}".format)
def test_genlambda(b: float, d: float, f: float):
    X = cast("Any", genlambda(b, d, f))

    assert X.cdf(X.support()[0]) == 0
    assert X.ppf(0) == X.support()[0]
    assert X.ppf(1) == X.support()[1]

    x = X.ppf(Q)
    q2 = X.cdf(x)
    assert_allclose(q2, Q)

    # m_x1 = X.expect(lambda x: x) if min(b, d) > -1 else np.nan
    # mean = X.mean()
    # assert_allclose(mean, m_x1, equal_nan=True)

    # m_x2 = X.expect(lambda x: (x - m_x1)**2) if min(b, d) > -.5 else np.nan
    # var = X.var()
    # assert_allclose(var, m_x2, equal_nan=True)

    # quad_opts={} forces numerical evaluation
    if b > -1 and d > -1:
        l_tau_quad = X.l_stats(quad_opts={})
        assert_allclose(l_tau_quad[0], X.mean(), atol=1e-8)
        assert l_tau_quad[1] > 0 or np.isnan(l_tau_quad[1])
        l_tau_theo = X.l_stats()
        assert_allclose(l_tau_theo, l_tau_quad, atol=1e-8)

    if b > -1 and d > -2:
        ll_tau_quad = X.l_stats(quad_opts={}, trim=(0, 1))
        assert ll_tau_quad[1] > 0 or np.isnan(ll_tau_quad[1])
        ll_tau_theo = X.l_stats(trim=(0, 1))
        assert_allclose(ll_tau_theo, ll_tau_quad)

    if b > -2 and d > -1:
        lh_tau_quad = X.l_stats(quad_opts={}, trim=(1, 0))
        assert lh_tau_quad[1] > 0 or np.isnan(lh_tau_quad[1])
        lh_tau_theo = X.l_stats(trim=(1, 0))
        assert_allclose(lh_tau_theo, lh_tau_quad)

    tl_tau_quad = X.l_stats(quad_opts={}, trim=1)
    assert tl_tau_quad[1] > 0 or np.isnan(tl_tau_quad[1])
    tl_tau_theo = X.l_stats(trim=1)
    assert_allclose(tl_tau_theo, tl_tau_quad, atol=1e-7)


@pytest.mark.parametrize("trim", [(0, 0), (0, 1), (1, 1)], ids=str)
@pytest.mark.parametrize(
    "rv",
    [
        distributions.uniform(),
        distributions.logistic(),
        distributions.expon(),
        distributions.gumbel_r(),
        distributions.gumbel_l(),
        distributions.genextreme(-0.1),
        distributions.genpareto(0.1),
        kumaraswamy(2, 5),
        wakeby(5, 1, 0.6),
        genlambda(0.5, -1, -0.1),
    ],
    ids=lambda rv: rv.dist.name,
)
def test_exact_lm(rv: l_rv_generic, trim: tuple[int, int]) -> None:
    r = [1, 2, 3, 4, 8]
    # if `quad_opts` is not None, the numerical fallback is used
    lm_quad = rv.l_moment(r, trim=trim, quad_opts={})
    lm_exact = rv.l_moment(r, trim=trim)

    assert not np.all(lm_exact == lm_quad)
    assert_allclose(lm_exact, lm_quad)
