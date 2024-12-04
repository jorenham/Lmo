from __future__ import annotations

import functools
from math import exp, factorial, gamma, lgamma, log
from typing import TYPE_CHECKING, Concatenate, Final, ParamSpec

import numpy as np
import optype.numpy as onp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import lmo.typing as lmt

__all__ = ("ALPHA", "QUAD_LIMIT", "l_coef_factor", "l_const", "tighten_cdf_support")


_Tss = ParamSpec("_Tss")


ALPHA: Final = 0.1
QUAD_LIMIT: Final = 100


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

    # math.lgamma is faster than scipy.special.loggamma.
    if r + s + t <= 20:
        v = gamma(r + s + t + 1) / (gamma(r + s) * gamma(r + t))
    elif r + s + t <= 128:
        v = exp(lgamma(r + s + t + 1) - lgamma(r + s) - lgamma(r + t))
    else:
        return exp(
            lgamma(r + s + t + 1)
            - lgamma(r + s)
            - lgamma(r + t)
            + lgamma(r - k)
            - log(r)
        )

    return factorial(r - 1 - k) / r * v


def l_coef_factor(
    r: int | lmt.Integer | onp.ArrayND[lmt.Integer],
    s: float = 0,
    t: float = 0,
) -> onp.ArrayND[np.float64]:
    if s == t == 0:
        return np.sqrt(2 * r - 1)

    assert s + t > -1

    import scipy.special as sps

    rst = r + s + t
    return np.sqrt((rst + r - 1) * sps.beta(r + s, r + t) / sps.beta(r, rst)) * r / rst


def tighten_cdf_support(
    cdf: Callable[[float], float],
    support: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Attempt to tighten the support by checking some common bounds."""
    a, b = (-np.inf, np.inf) if support is None else map(float, support)

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
    domains: Sequence[tuple[float, float] | Callable[..., tuple[float, float]],],
    opts: lmt.QuadOptions | None = None,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float:
    import scipy.integrate as spi

    if kwds:
        integrand = functools.partial(integrand, **kwds)

    return spi.nquad(integrand, domains[::-1], args=args, opts=opts)[0]
