# ruff: noqa: C901

"""
Known theoretical L-moments for specific distributions.
The names that are used here, match those in
[`scipy.stats.distributions`][scipy.stats.distributions].
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from math import gamma, log
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Final,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy.special as sps

from lmo.special import harmonic
from lmo.theoretical import l_moment_from_ppf


if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

if TYPE_CHECKING:
    import lmo.typing.np as lnpt


__all__ = [
    'lm_uniform',
    'lm_logistic',
    'lm_expon',
    'lm_gumbel_r',
    'lm_genextreme',
    'lm_genpareto',
    'lm_kumaraswamy',
    'lm_wakeby',
    'lm_genlambda',
]

DistributionName: TypeAlias = Literal[
    # 0 params
    'uniform',
    'logistic',
    'expon',
    'gumbel_r',
    # 1 param
    'genextreme',
    'genpareto',
    # 2 params
    'kumaraswamy',
    # 3 params
    'wakeby',
    'genlambda',
]
_LM_REGISTRY: Final[dict[DistributionName, _LmVFunc[...]]] = {}

_ArrF8: TypeAlias = onpt.Array[tuple[int, ...], np.float64]

_Tss = ParamSpec('_Tss')
# (r, s, t, *params) -> float
_LmFunc: TypeAlias = Callable[
    Concatenate[int, float, float, _Tss],
    float | np.float64,
]

_ShapeT = TypeVar('_ShapeT', bound=tuple[int, ...])


_LN2: Final = np.log(2)
_LN3: Final = np.log(3)
_LN5: Final = np.log(5)


# workaround for partial type annotations (i.e. missing and un-inferrable)
_binom = cast(
    Callable[
        [npt.NDArray[np.intp] | int, npt.NDArray[np.intp] | int],
        npt.NDArray[np.intp],
    ],
    sps.comb,
)


class _LmVFunc(Protocol[_Tss]):
    pyfunc: _LmFunc[_Tss]

    @overload
    def __call__(
        self,
        r: onpt.CanArray[_ShapeT, np.dtype[lnpt.Integer]],
        s: float,
        t: float,
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> onpt.Array[_ShapeT, np.float64]: ...
    @overload
    def __call__(
        self,
        r: int | lnpt.Integer,
        s: float,
        t: float,
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> onpt.Array[tuple[()], np.float64]: ...


def register_lm_func(
    name: DistributionName,
    /,
) -> Callable[[_LmFunc[_Tss]], _LmVFunc[_Tss]]:
    # TODO: vectorize decorator (only for `r`), with correct signature
    assert name not in _LM_REGISTRY

    def _wrapper(func: _LmFunc[_Tss], /) -> _LmVFunc[_Tss]:
        vfunc = np.vectorize(func, [float], excluded={1, 2})
        _LM_REGISTRY[name] = vfunc
        return vfunc

    return _wrapper


def has_lm_func(name: str, /) -> TypeIs[DistributionName]:
    return name in _LM_REGISTRY


def get_lm_func(name: DistributionName, /) -> _LmVFunc[...]:
    """
    Return the L-moment function for the given distribution, or raise a
    `KeyError` if not registered.
    """
    return _LM_REGISTRY[name]


@register_lm_func('uniform')
def lm_uniform(r: int, s: float, t: float, /) -> float:
    """
    Exact generalized* trimmed L-moments of the standard uniform distribution
    on the interval [0, 1].

    *: Only the `s` trim-length can be fractional.
    """
    if r == 0:
        return 1

    if not isinstance(t, int):
        msg = 't must be an integer'
        raise NotImplementedError(msg)

    if r == 1:
        return (1 + s) / (2 + s + t)
    if r == 2:
        return 1 / (2 * (3 + s + t))
    return 0


@register_lm_func('logistic')
def lm_logistic(r: int, s: float, t: float, /) -> float:
    """
    Exact generalized trimmed L-moments of the standard logistic
    distribution.
    """
    if r == 0:
        return 1

    if isinstance(t, float) and not t.is_integer():
        msg = 't must be an integer'
        raise NotImplementedError(msg)

    # symmetric trimming
    if s == t:
        if r % 2:
            return 0

        if s == 0:
            if r == 2:
                return 1
            if r == 4:
                return 1 / 6
        if s == 1:
            if r == 2:
                return 1 / 2
            if r == 4:
                return 1 / 24

        return 2 * sps.beta(r - 1, t + 1) / r

    # asymmetric trimming
    if r == 1:
        return harmonic(s) - harmonic(t)

    r1m = r - 1
    return ((-1) ** r * sps.beta(r1m, s + 1) + sps.beta(r1m, t + 1)) / r


@register_lm_func('expon')
def lm_expon(r: int, s: float, t: float, /) -> float:
    """
    Exact generalized trimmed L-moments of the standard exponential
    distribution.
    """
    if r == 0:
        return 1
    if r == 1:
        return 1 / (1 + t) if s == 0 else harmonic(s + t + 1) - harmonic(t)

    if t == 0:
        return 1 / (r * (r - 1))

    # it doesn't depend on `s` when `r > 1`, which is pretty crazy actually
    if r == 2:
        return 1 / (2 * (t + 1))
    if r == 3:
        return 1 / (3 * (t + 1) * (t + 2))
    if r == 4:
        # the 2 here isn't a typo
        return 1 / (2 * (t + 1) * (t + 2) * (t + 3))

    return sps.beta(r - 1, t + 1) / r


@register_lm_func('gumbel_r')
def lm_gumbel_r(r: int, s: float, t: float, /) -> np.float64 | float:
    """
    Exact trimmed L-moments of the Gumbel / Extreme Value (EV) distribution.

    Doesn't support fractional trimming.
    """
    if r == 0:
        return 1

    if not isinstance(s, int) or not isinstance(t, int):
        msg = 'fractional trimming'
        raise NotImplementedError(msg)

    match (r, s, t):
        case (1, 0, 0):
            return np.euler_gamma
        case (1, 0, 1):
            return np.euler_gamma - _LN2
        case (1, 1, 1):
            return np.euler_gamma + _LN2 * 3 - _LN3 * 2

        case (2, 0, 0):
            return _LN2
        case (2, 0, 1):
            return _LN2 * 3 - _LN3 * 3 / 2
        case (2, 1, 1):
            return _LN2 * -9 + _LN3 * 6

        case (3, 0, 0):
            return _LN2 * -3 + _LN3 * 2
        case (3, 0, 1):
            return _LN2 * -38 / 3 + _LN3 * 8
        case (3, 1, 1):
            return (_LN2 * 110 - _LN3 * 40 - _LN5 * 20) / 3

        case (4, 0, 0):
            return _LN2 * 16 - _LN3 * 10
        case (4, 0, 1):
            return _LN2 * 60 - _LN3 * 25 - _LN5 * 35 / 4
        case (4, 1, 1):
            return (_LN2 * -107 + _LN3 * 6 + _LN5 * 42) * 5 / 4

        case _:
            return lm_genextreme.pyfunc(r, s, t, 0)


@register_lm_func('genextreme')
def lm_genextreme(
    r: int,
    s: float,
    t: float,
    /,
    a: float,
) -> np.float64 | float:
    """
    Exact trimmed L-moments of the Generalized Extreme Value (GEV)
    distribution.

    Doesn't support fractional trimming.
    """
    if r == 0:
        return 1

    if not isinstance(s, int) or not isinstance(t, int):
        msg = 'fractional trimming'
        raise NotImplementedError(msg)

    if a < 0 and (isinstance(a, int) or a.is_integer()):
        msg = 'a cannot be a negative integer'
        raise ValueError(msg)

    if r + s + t < 8:
        # numerical accurate to within approx. < 1e-12 error
        k0 = s + 1
        kn = r + s + t
        k = np.arange(k0, kn + 1, dtype=np.intp)

        if a == 0:
            pwm = np.euler_gamma + np.log(k)
        else:
            pwm = (1 / a - sps.gamma(a) / k**a)

        return np.sum(
            (-1) ** k
            * _binom(kn, k)
            * _binom(r - 2 + k, r - 2 + k0)
            * pwm,
        ) * (-1) ** (r + s) / r

    # NOTE: some performance notes:
    # - `math.log` is faster for scalar input that `numpy.log`
    # - conditionals within the function are avoided through multiple functions
    if a == 0:
        def _ppf(q: float) -> float:
            if q <= 0:
                return -float('inf')
            if q >= 1:
                return float('inf')
            return -log(-log(q))
    elif a < 0:
        def _ppf(q: float) -> float:
            if q <= 0:
                return 1 / a
            if q >= 1:
                return float('inf')
            return (1 - (-log(q))**a) / a
    else:  # a > 0
        def _ppf(q: float) -> float:
            if q <= 0:
                return -float('inf')
            if q >= 1:
                return 1 / a
            return (1 - (-log(q))**a) / a

    return l_moment_from_ppf(_ppf, r, (s, t))


@register_lm_func('genpareto')
def lm_genpareto(
    r: int,
    s: float,
    t: float,
    /,
    a: float,
) -> np.float64 | float:
    """
    Exact trimmed L-moments of the Generalized Pareto distribution (GPD).
    """
    if r == 0:
        return 1
    if a == 0:
        return lm_expon.pyfunc(r, s, t)
    if a == 1:
        return lm_uniform.pyfunc(r, s, t)

    if not isinstance(t, int):
        msg = 'fractional trimming'
        raise NotImplementedError(msg)

    if a >= 1 + t:
        return float('nan')

    b = a - 1
    n = r + s + t + 1

    if r == 1 and s == 0:
        return 1 / (t - b)
    if r == 1:
        return (gamma(t - b) / gamma(r + t) * gamma(n) / gamma(n - a) - 1) / a

    return sps.beta(r + b, t - b) / (a * sps.beta(n - a, a)) / r


@register_lm_func('kumaraswamy')
def lm_kumaraswamy(
    r: int,
    s: float,
    t: float,
    /,
    a: float,
    b: float,
) -> np.float64 | float:
    """
    Exact trimmed L-moments of (the location-scale reparametrization of)
    Kumaraswamy's distribution [@kumaraswamy1980].

    Doesn't support fractional trimming.
    """
    if r == 0:
        return 1

    if not isinstance(s, int) or not isinstance(t, int):
        msg = 'fractional trimming not supported'
        raise NotImplementedError(msg)

    k = np.arange(t + 1, r + s + t + 1)
    return (
        np.sum(
            (-1) ** (k - t - 1)
            * cast(_ArrF8, sps.comb(r + k - 2, r + t - 1))  # pyright: ignore[reportUnknownMemberType]
            * cast(_ArrF8, sps.comb(r + s + t, k))  # pyright: ignore[reportUnknownMemberType]
            * cast(_ArrF8, sps.beta(1 / a, 1 + k * b))
            / a,
        )
        / r
    )


@register_lm_func('wakeby')
def lm_wakeby(
    r: int,
    s: float,
    t: float,
    /,
    b: float,
    d: float,
    f: float,
) -> float | np.float64:
    """
    Exact generalized trimmed L-moments of (the location-scale
    reparametrization of) Wakeby's distribution [@houghton1978].
    """
    if r == 0:
        return 1

    if d >= (b == 0) + 1 + t:
        return np.nan

    def _lmo0_partial(theta: float, scale: float, /) -> float | np.float64:
        if scale == 0:
            return 0
        if r == 1 and theta == 0:
            return harmonic(s + t + 1) - harmonic(t)

        return (
            scale
            * (
                sps.poch(r + t, s + 1)
                * sps.poch(1 - theta, r - 2)
                / sps.poch(1 + theta + t, r + s)
                + (1 / theta if r == 1 else 0)
            )
            / r
        )

    return float(_lmo0_partial(b, f) + _lmo0_partial(-d, 1 - f))


@register_lm_func('genlambda')
def lm_genlambda(
    r: int,
    s: float,
    t: float,
    /,
    b: float,
    d: float,
    f: float,
) -> float:
    """
    Exact generalized trimmed L-moments of the (location-scale
    reparametrization of the) Generalized Tukey-Lambda Distribution (GLD)
    [@ramberg1974].
    """
    if r == 0:
        return 1

    if b <= -1 - s and d <= -1 - t:
        return np.nan

    sgn = (-1) ** r

    def _lmo0_partial(trim: float, theta: float) -> float | np.float64:
        if r == 1 and theta == 0:
            return harmonic(trim) - harmonic(s + t + 1)

        return (
            sgn
            * sps.poch(r + trim, s + t - trim + 1)
            * sps.poch(1 - theta, r - 2)
            / sps.poch(1 + theta + trim, r + s + t - trim)
            - (1 / theta if r == 1 else 0)
        ) / r

    lhs = (1 + f) * _lmo0_partial(s, b)
    rhs = (1 - f) * _lmo0_partial(t, d)
    return lhs + sgn * rhs
