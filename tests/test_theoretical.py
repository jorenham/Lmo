import functools

import numpy as np
from hypothesis import (
    given,
    strategies as st,
)
from lmo.theoretical import (
    l_moment_from_cdf,
    l_moment_from_ppf,
    l_moment_cov_from_cdf,
    l_stats_cov_from_cdf,
)


def cauchy_cdf(x: float) -> float:
    return np.arctan(x) / np.pi + 1 / 2


def cauchy_ppf(p: float) -> float:
    return np.tan(np.pi * (p - 1 / 2))


def expon_cdf(x: float, a: float = 1) -> float:
    return 1 - np.exp(-x / a) if x >= 0 else 0.0


def expon_ppf(p: float, a: float = 1) -> float:
    return -a * np.log(1 - p)


@np.errstate(over='ignore', under='ignore')
def gumbel_cdf(x: float, loc: float = 0, scale: float = 1) -> float:
    return np.exp(-np.exp(-(x - loc) / scale))


@given(a=st.floats(0.1, 10))
def test_lm_expon(a: float):
    l_stats = np.array([a, a / 2, 1 / 3, 1 / 6])

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)

    l_ppf = l_moment_from_ppf(ppf, [0, 1, 2, 3, 4])
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = l_moment_from_cdf(cdf, [0, 1, 2, 3, 4])
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)


def test_lm_normal():
    # cdf and ppf of normal dist
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    l_stats = np.array(
        [
            mu,
            sigma / np.sqrt(np.pi),
            0,
            30 * np.arctan(np.sqrt(2)) / np.pi - 9,
        ]
    )

    l_ppf = l_moment_from_ppf(IQ.inv_cdf, [0, 1, 2, 3, 4])
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = l_moment_from_cdf(IQ.cdf, [0, 1, 2, 3, 4])
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)


def test_tlm_normal():
    # cdf and ppf of normal dist
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    l_stats = np.array([mu, 0.2970 * sigma, 0, 0.06248])

    l_ppf = l_moment_from_ppf(IQ.inv_cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats, rtol=1e-4)

    l_cdf = l_moment_from_cdf(IQ.cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats, rtol=1e-4)


def test_tlm_cauchy():
    l_stats = np.array([0, 0.698, 0, 0.343])

    l_ppf = l_moment_from_ppf(cauchy_ppf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats, rtol=1e-3)

    l_cdf = l_moment_from_cdf(cauchy_cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats, rtol=1e-3)


@given(a=st.floats(0.1, 10))
def test_lhm_expon(a: float):
    l_stats = np.array([a / 2, a / 4, 2 / 9, 1 / 12])

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)

    l_ppf = l_moment_from_ppf(ppf, [0, 1, 2, 3, 4], trim=(0, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = l_moment_from_cdf(cdf, [0, 1, 2, 3, 4], trim=(0, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)


def test_lm_cov_uniform():
    k4 = np.array([
        [1 / 2, 0, -1 / 10, 0],
        [0, 1 / 30, 0, -1 / 70],
        [-1 / 10, 0, 1 / 35, 0],
        [0, -1 / 70, 0, 1 / 105],
    ]) / 6
    k4_hat = l_moment_cov_from_cdf(lambda x: x, 4)

    assert np.allclose(k4, k4_hat)


def test_lm_cov_expon():
    k3 = np.array([
        [1, 1 / 2, 1 / 6],
        [1 / 2, 1 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 15],
    ])
    k3_hat = l_moment_cov_from_cdf(lambda x: 1-np.exp(-x), 3)

    assert np.allclose(k3, k3_hat)


def test_lhm_cov_expon():
    k3 = np.array([
        [1 / 3, 1 / 8, 0],
        [1 / 8, 3 / 40, 1 / 60],
        [0, 1 / 60, 16 / 945],
    ])
    k3_hat = l_moment_cov_from_cdf(expon_cdf, 3, trim=(0, 1))

    assert np.allclose(k3, k3_hat)


def test_lm_cov_loc_invariant():
    k4_hat = l_moment_cov_from_cdf(gumbel_cdf, 4)
    k4_hat_l = l_moment_cov_from_cdf(
        functools.partial(gumbel_cdf, loc=-1),
        4
    )
    k4_hat_r = l_moment_cov_from_cdf(
        functools.partial(gumbel_cdf, loc=1),
        4
    )

    assert np.allclose(k4_hat, k4_hat_l)
    assert np.allclose(k4_hat, k4_hat_r)


def test_lm_cov_scale_invariant():
    k4_hat = l_moment_cov_from_cdf(gumbel_cdf, 4)
    k4_hat_l = l_moment_cov_from_cdf(
        functools.partial(gumbel_cdf, scale=1/3),
        4
    )
    k4_hat_r = l_moment_cov_from_cdf(
        functools.partial(gumbel_cdf, scale=3),
        4
    )

    assert np.allclose(k4_hat, k4_hat_l * 9)
    assert np.allclose(k4_hat, k4_hat_r / 9)


def test_ls_cov_uniform():
    k4 = np.array([
        [1 / 12, 0, -1 / 10, 0],
        [0, 1 / 180, 0, -1 / 70],
        [-1 / 10, 0, 6 / 35, 0],
        [0, -1 / 70, 0, 2 / 35],
    ])
    k4_hat = l_stats_cov_from_cdf(lambda x: x)

    assert np.allclose(k4, k4_hat)
