"""Typing utilities meant for internal usage."""  # noqa: A005

# pyright: reportPrivateUsage=false
# ruff: noqa: PLC2701

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    overload,
)

import numpy as np
import optype.numpy as onp
from scipy.stats._distn_infrastructure import (
    _ShapeInfo as ShapeInfo,
    rv_continuous,
    rv_frozen,
    rv_generic,
)

__all__ = [
    "Callable2",
    "Floating",
    "Integer",
    "LComomentOptions",
    "LMomentOptions",
    "OrderReshape",
    "QuadOptions",
    "ShapeInfo",
    "SortKind",
    "ToAWeights",
    "ToFWeights",
    "ToIntTrim",
    "ToOrder0D",
    "ToOrder1D",
    "ToOrderND",
    "ToTrim",
    "rv_continuous",
    "rv_frozen",
    "rv_generic",
]


def __dir__() -> list[str]:
    return __all__


Integer: TypeAlias = np.integer[Any]
Floating: TypeAlias = np.floating[Any]

OrderReshape: TypeAlias = Literal["C", "F", "A"]
"""Type of the `order` parameter of e.g. [`np.reshape`][numpy.array]."""

# matches `_SortKind` in `numpy/__init__.pyi` and `numpy >= 2.1`
SortKind: TypeAlias = Literal[
    "Q", "quick", "quicksort",
    "M", "merge", "mergesort",
    "H", "heap", "heapsort",
    "S", "stable", "stablesort",
]  # fmt: skip
"""
Type of the `kind` parameter of e.g. [`np.sort`][numpy.sort], as
allowed by numpy's own stubs.
Note that the actual implementation just looks at `kind[0].lower() == 'q'`.
This means that it's possible to select stable-sort by passing
`kind='SnailSort'` instead of `kind='stable'` (although your typechecker might
ruin the fun).
"""

RNG: TypeAlias = np.random.Generator | np.random.RandomState
Seed: TypeAlias = (
    int
    | Integer
    # | np.timedelta64
    # | onp.ArrayND[Integer | np.timedelta64 | np.flexible | np.object_]
    | onp.ArrayND[Integer]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | RNG
    | None
)

"""The accepted type of [`numpy.random.default_rng`][numpy.random.default_rng]."""


ToIntTrim: TypeAlias = int | tuple[int, int]
ToTrim: TypeAlias = float | tuple[float, float]

ToOrder0D: TypeAlias = int | Integer
ToOrder1D: TypeAlias = onp.CanArrayND[Integer] | Sequence[ToOrder0D]
ToOrderND: TypeAlias = (
    onp.CanArrayND[Integer]
    | onp.SequenceND[onp.CanArrayND[Integer]]
    | onp.SequenceND[ToOrder0D]
)
ToOrder: TypeAlias = ToOrder0D | ToOrderND

ToFWeights: TypeAlias = onp.ArrayND[Integer]
ToAWeights: TypeAlias = onp.ArrayND[Floating]


class UnivariateOptions(TypedDict, total=False):
    """Use as e.g. `**kwds: Unpack[UnivariateOptions]`."""

    sort: bool | SortKind
    fweights: ToFWeights
    aweights: ToAWeights


class LMomentOptions(UnivariateOptions, TypedDict, total=False):
    """Use as e.g. `**kwds: Unpack[LMomentOptions]`."""

    cache: bool | None


class LComomentOptions(TypedDict, total=False):
    """Use as e.g. `**kwds: Unpack[LComomentOptions]`."""

    sort: SortKind
    cache: bool | None
    rowvar: bool | None


_Tss = ParamSpec("_Tss")
_T = TypeVar("_T")
_F1_co = TypeVar("_F1_co", bound=Callable[..., object], covariant=True)
_F2_co = TypeVar("_F2_co", bound=Callable[..., object], covariant=True)


class Callable2(Protocol[_F1_co, _F2_co]):
    """
    The intersection of two callable types, i.e. a callable with two overloads, one
    for each of the callable type params.
    """

    @overload
    def __call__(
        self: Callable2[Callable[_Tss, _T], _F2_co],
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> _T: ...
    @overload
    def __call__(
        self: Callable2[_F1_co, Callable[_Tss, _T]],
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> _T: ...


# scipy stuff

_IntLike: TypeAlias = int | Integer
_FloatLike: TypeAlias = float | Floating


class QuadOptions(TypedDict, total=False):
    """
    Optional quadrature options to be passed to the integration routine, e.g.
    [`scipy.integrate.quad`][scipy.integrate.quad].
    """

    epsabs: _FloatLike
    epsrel: _FloatLike
    limit: _IntLike
    points: onp.ToFloat1D
    weight: Literal["cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy"]
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike]
    wopts: tuple[_IntLike, onp.ArrayND[np.float32 | np.float64]]
