import functools
from collections.abc import Callable, Sequence
from math import exp, factorial, gamma, lgamma, log
from typing import Concatenate, Final, ParamSpec, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi
import scipy.special as sps

import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt


__all__ = (
    'ALPHA',
    'QUAD_LIMIT',
    'l_coef_factor',
    'l_const',
    'tighten_cdf_support',
)


_Fn1: TypeAlias = Callable[[float], float | lnpt.Float]
_Tss = ParamSpec('_Tss')

ALPHA: Final[float] = 0.1
QUAD_LIMIT: Final[int] = 100


@functools.cache
def l_const(r: int, s: float, t: float, k: int = 0) -> float:
    assert k >= 0

    if r <= k:
        return 1.0
    if s == t == 0:
        if k == 0:
            return 1.0
        if k == 1:
            return 1 / (r - 1)

    # math.lgamma is faster (and has better type annotations) than
    # scipy.special.loggamma.
    if r + s + t <= 20:
        v = gamma(r + s + t + 1) / (gamma(r + s) * gamma(r + t))
    elif r + s + t <= 128:
        v = exp(lgamma(r + s + t + 1) - lgamma(r + s) - lgamma(r + t))
    else:
        return exp(
            +lgamma(r + s + t + 1)
            - lgamma(r + s)
            - lgamma(r + t)
            + lgamma(r - k)
            - log(r),
        )

    return factorial(r - 1 - k) / r * v


def l_coef_factor(
    r: int | np.intp | npt.NDArray[np.intp],
    s: float = 0,
    t: float = 0,
) -> npt.NDArray[np.float64]:
    if s == t == 0:
        return np.sqrt(2 * r - 1)

    assert s + t > -1

    _r = np.asarray(r)
    c = (
        np.sqrt(
            (2 * _r + (s + t - 1))
            * sps.beta(_r + s, _r + t)
            / sps.beta(_r, _r + s + t),
        )
        * _r
        / (_r + s + t)
    )
    return c[()] if _r.ndim == 0 and np.isscalar(r) else c


def tighten_cdf_support(
    cdf: _Fn1,
    support: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Attempt to tighten the support by checking some common bounds."""
    a, b = (-np.inf, np.inf) if support is None else map(float, support)

    # assert a < b, (a, b)
    # assert (u_a := cdf(a)) == 0, (a, u_a)
    # assert (u_b := cdf(b)) == 1, (b, u_b)

    # attempt to tighten the default support by checking some common bounds
    if cdf(0) == 0:
        # left-bounded at 0 (e.g. weibull)
        a = 0

        if (u1 := cdf(1)) == 0:
            # left-bounded at 1 (e.g. pareto)
            a = 1
        elif u1 == 1:
            # right-bounded at 1 (e.g. beta)
            b = 1

    return a, b


def nquad(
    integrand: Callable[Concatenate[float, float, _Tss], float],
    domains: Sequence[
        tuple[float, float] | Callable[..., tuple[float, float]],
    ],
    opts: lspt.QuadOptions | None = None,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float:
    # nquad only has an `args` param for some invalid reason
    fn = functools.partial(integrand, **kwds) if kwds else integrand

    return cast(
        tuple[float, float],
        spi.nquad(fn, domains[::-1], args, opts=opts),  # pyright: ignore[reportUnknownMemberType]
    )[0]
