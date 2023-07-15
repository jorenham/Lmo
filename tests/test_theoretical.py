import functools

from hypothesis import given, strategies as st
import numpy as np

from lmo.theoretical import l_moment_from_ppf, l_moment_from_cdf


def cauchy_cdf(x: float) -> float:
    return np.arctan(x) / np.pi + 1 / 2

def cauchy_ppf(p: float) -> float:
    return np.tan(np.pi * (p - 1 / 2))

def expon_cdf(x: float, a: float = 1) -> float:
    return 1 - np.exp(-x / a) if x >= 0 else 0.

def expon_ppf(p: float, a: float = 1) -> float:
    return -a * np.log(1 - p)


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

    l_stats = np.array([
        mu,
        sigma / np.sqrt(np.pi),
        0,
        30 * np.arctan(np.sqrt(2)) / np.pi - 9,
    ])

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
def test_tlm_expon(a: float):
    l_stats = np.array([a * 5 / 6, a / 4, 2 / 9, 1 / 12])

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)

    l_ppf = l_moment_from_ppf(ppf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = l_moment_from_cdf(cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)
