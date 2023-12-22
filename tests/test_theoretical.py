# pyright: reportUnknownVariableType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false

import functools
from typing import Callable, cast
from numpy.testing import assert_allclose

from hypothesis import (
    given,
    settings,
    strategies as st,
)

import numpy as np
from scipy.special import ndtr, ndtri, zeta

from lmo import constants
from lmo.theoretical import (
    l_moment_from_cdf,
    l_moment_from_ppf,
    l_moment_from_qdf,
    l_moment_cov_from_cdf,
    l_stats_cov_from_cdf,
    ppf_from_l_moments,
    qdf_from_l_moments,
)


norm_cdf = cast(Callable[[float], float], ndtr)
norm_ppf = cast(Callable[[float], float], ndtri)

@np.errstate(over='ignore', under='ignore')
def norm_qdf(x: float) -> float:
    # cool, eh?
    return np.sqrt(2 * np.pi) * np.exp(norm_ppf(x)**2 / 2)

def cauchy_cdf(x: float) -> float:
    return np.arctan(x) / np.pi + 1 / 2

def cauchy_ppf(p: float) -> float:
    return np.tan(np.pi * (p - 1 / 2))

def cauchy_qdf(p: float) -> float:
    return np.pi / np.sin(p * np.pi)**2

def expon_cdf(x: float, a: float = 1) -> float:
    return 1 - np.exp(-x / a) if x >= 0 else 0.0

def expon_ppf(p: float, a: float = 1) -> float:
    return -a * np.log1p(-p)

def expon_qdf(p: float, a: float = 1) -> float:
    return a  / (1 - p)

@np.errstate(over='ignore', under='ignore')
def gumbel_cdf(x: float, loc: float = 0, scale: float = 1) -> float:
    return np.exp(-np.exp(-(x - loc) / scale))

@np.errstate(over='ignore', under='ignore')
def gumbel_ppf(p: float, loc: float = 0, scale: float = 1) -> float:
    return loc - scale * np.log(-np.log(p))

@np.errstate(over='ignore', under='ignore', divide='ignore')
def gumbel_qdf(p: float, loc: float = 0, scale: float = 1) -> float:
    # return -scale / (p * np.log(p))
    return scale / np.log(np.exp(-p * np.log(p)))

def rayleigh_cdf(x: float) -> float:
    return -np.expm1(-x**2 / 2)

def rayleigh_ppf(p: float) -> float:
    return np.sqrt(-2 * np.log1p(-p))

def rayleigh_qdf(p: float) -> float:
    return 1 / ((1 - p) * rayleigh_ppf(p))

def uniform_cdf(x: float) -> float:
    return np.clip(x, 0, 1)

def uniform_ppf(p: float) -> float:
    return np.clip(p, 0, 1)

def uniform_qdf(p: float) -> float:
    return ((p > 0) & (p < 1)) * 1.

@given(a=st.floats(0.1, 10))
def test_lm_expon(a: float):
    l_stats = np.array([a, a / 2, 1 / 3, 1 / 6])

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)

    l_ppf = l_moment_from_ppf(ppf, [0, 1, 2, 3, 4])
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert_allclose(l_stats_ppf, l_stats)

    l_cdf = l_moment_from_cdf(cdf, [0, 1, 2, 3, 4])
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert_allclose(l_stats_cdf, l_stats)


def test_lm_normal():
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    r = [1, 2, 3, 4]

    l2 = sigma / np.sqrt(np.pi)
    l = np.array([mu, l2, 0, l2 * (30 * constants.theta_m / np.pi - 9)])

    l_ppf = l_moment_from_ppf(IQ.inv_cdf, r)
    assert_allclose(l_ppf, l)

    l_cdf = l_moment_from_cdf(IQ.cdf, r)
    assert_allclose(l_cdf, l)

    # QDF is shift-invariant, so it can't be used to find the L-loc
    l_qdf = l_moment_from_qdf(lambda u: 1 / IQ.pdf(IQ.inv_cdf(u)), r[1:])
    assert_allclose(l_qdf, l[1:])


def test_tlm_normal():
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    r = [1, 2, 3, 4]

    tl2 = 6 * sigma / np.sqrt(np.pi) * (1 - 3 * constants.theta_m / np.pi)
    tl = np.array([mu, tl2, 0, tl2 * 0.06247999167])

    tl_ppf = l_moment_from_ppf(IQ.inv_cdf, r, trim=1)
    assert_allclose(tl_ppf, tl)

    tl_cdf = l_moment_from_cdf(IQ.cdf, r, trim=1)
    assert_allclose(tl_cdf, tl)

    # QDF is shift-invariant, so it can't be used to find the L-loc
    tl_qdf = l_moment_from_qdf(
        lambda u: 1 / IQ.pdf(IQ.inv_cdf(u)), r[1:],
        trim=1,
    )
    assert_allclose(tl_qdf, tl[1:])


def test_tlm_cauchy():
    r = [1, 2, 3, 4]

    z3 = zeta(3)
    l2 = 18 * z3 / np.pi**3
    l = l2 * np.array([0, 1, 0, 25 / 6 - 175 * zeta(5) / (4 * np.pi**2 * z3)])

    l_ppf = l_moment_from_ppf(cauchy_ppf, r, trim=1)
    assert_allclose(l_ppf, l)

    l_cdf = l_moment_from_cdf(cauchy_cdf, r, trim=1)
    assert_allclose(l_cdf, l)

    l_qdf = l_moment_from_qdf(cauchy_qdf, r[1:], trim=1)
    assert_allclose(l_qdf, l[1:])


@given(a=st.floats(0.1, 10))
def test_lhm_expon(a: float):
    r = [1, 2, 3, 4]
    l = a * np.array([1, 1 / 2, 1 / 9, 1 / 24]) / 2

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)
    qdf = functools.partial(expon_qdf, a=a)

    l_ppf = l_moment_from_ppf(ppf, r, trim=(0, 1))
    assert_allclose(l_ppf, l)

    l_cdf = l_moment_from_cdf(cdf, r, trim=(0, 1))
    assert_allclose(l_cdf, l)

    l_qdf = l_moment_from_qdf(qdf, r[1:], trim=(0, 1))
    assert_allclose(l_qdf, l[1:])


def test_lm_cov_uniform():
    k4 = np.array([
        [1 / 2, 0, -1 / 10, 0],
        [0, 1 / 30, 0, -1 / 70],
        [-1 / 10, 0, 1 / 35, 0],
        [0, -1 / 70, 0, 1 / 105],
    ]) / 6
    k4_hat = l_moment_cov_from_cdf(lambda x: x, 4)

    assert_allclose(k4, k4_hat)


def test_lm_cov_expon():
    k3 = np.array([
        [1, 1 / 2, 1 / 6],
        [1 / 2, 1 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 15],
    ])
    k3_hat = l_moment_cov_from_cdf(lambda x: 1-np.exp(-x), 3)

    assert_allclose(k3, k3_hat)


def test_lhm_cov_expon():
    k3 = np.array([
        [1 / 3, 1 / 8, 0],
        [1 / 8, 3 / 40, 1 / 60],
        [0, 1 / 60, 16 / 945],
    ])
    k3_hat = l_moment_cov_from_cdf(expon_cdf, 3, trim=(0, 1))

    assert_allclose(k3, k3_hat)


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

    assert_allclose(k4_hat, k4_hat_l)
    assert_allclose(k4_hat, k4_hat_r)


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

    assert_allclose(k4_hat, k4_hat_l * 9)
    assert_allclose(k4_hat, k4_hat_r / 9)


def test_ls_cov_uniform():
    k4 = np.array([
        [1 / 12, 0, -1 / 10, 0],
        [0, 1 / 180, 0, -1 / 70],
        [-1 / 10, 0, 6 / 35, 0],
        [0, -1 / 70, 0, 2 / 35],
    ])
    k4_hat = l_stats_cov_from_cdf(lambda x: x)

    assert_allclose(k4, k4_hat)


@settings(deadline=1_000)
@given(
    ppf=st.one_of(*map(st.just, [
        uniform_ppf,
        norm_ppf,
        gumbel_ppf,
        rayleigh_ppf,
        expon_ppf,
    ])),
    rmax=st.integers(2, 8),
    trim=st.tuples(st.integers(0, 1), st.integers(0, 3))
)
def test_ppf_from_l_moments_identity(
    ppf: Callable[[float], float],
    rmax: int,
    trim: tuple[int, int] | int,
):
    r = np.mgrid[1: rmax + 1]
    l_r = l_moment_from_ppf(ppf, r, trim)

    ppf_hat = ppf_from_l_moments(l_r, trim=trim, validate=False)
    l_r_hat = l_moment_from_ppf(ppf_hat, r, trim)
    assert_allclose(l_r_hat, l_r)

    l_0 = np.zeros(4)
    l_0_hat = l_moment_from_ppf(ppf_hat, np.mgrid[rmax + 1: rmax + 5], trim)
    assert_allclose(l_0_hat, l_0)


@settings(deadline=1_000)
@given(
    qdf=st.one_of(*map(st.just, [
        uniform_qdf,
        norm_qdf,
        gumbel_qdf,
        rayleigh_qdf,
        expon_qdf,
    ])),
    rmax=st.integers(3, 8),
    trim=st.tuples(st.integers(0, 1), st.integers(0, 3))
)
def test_qdf_from_l_moments_identity(
    qdf: Callable[[float], float],
    rmax: int,
    trim: tuple[int, int] | int,
):
    r = np.mgrid[2: rmax + 1]
    l_r = l_moment_from_qdf(qdf, r, trim)

    qdf_hat = qdf_from_l_moments(np.r_[0, l_r], trim=trim, validate=False)
    l_r_hat = l_moment_from_qdf(qdf_hat, r, trim)
    assert_allclose(l_r_hat, l_r)

    l_0 = np.zeros(4)
    l_0_hat = l_moment_from_qdf(qdf_hat, np.mgrid[rmax + 1: rmax + 5], trim)
    assert_allclose(l_0_hat, l_0)
