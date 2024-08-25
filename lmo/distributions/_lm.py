"""
Known theoretical L-moments for specific distributions.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
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
import optype.numpy as onpt
import scipy.special as sps

from lmo.special import harmonic


if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

if TYPE_CHECKING:
    import lmo.typing.np as lnpt


__all__ = [
    'lm_expon',
    'lm_kumaraswamy',
    'lm_wakeby',
    'lm_genlambda',
]

DistributionName: TypeAlias = Literal[
    'expon',
    'kumaraswamy',
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


class _LmVFunc(Protocol[_Tss]):
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

    Doesn't support fraction trimming.
    """
    if r == 0:
        return 1

    if not isinstance(s, int) or not isinstance(t, int):
        msg = 'fractional trimming not supported'
        raise NotImplementedError(msg)

    k = np.arange(t + 1, r + s + t + 1)
    return (
        np.sum(
            (-1) ** (k - 1)
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
