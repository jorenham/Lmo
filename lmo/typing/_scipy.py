# pyright: reportPropertyTypeMismatch=false
# ruff: noqa: TD002, TD003
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from .np import (
    AnyScalarFloat,
    AnyVectorFloat,
    Array,
    AtLeast0D,
    AtLeast1D,
    AtLeast2D,
    CanArray,
    Integer,
    Real,
)


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        ItemsView,
        Iterator,
        KeysView,
        Sequence,
        ValuesView,
    )

    from .compat import LiteralString, Self


__all__ = (
    'QuadOptions',
    'OptimizeResult',
    'RV', 'RVFrozen',
    'RVDiscrete', 'RVContinuous',
    'RVDiscreteFrozen', 'RVContinuousFrozen',
)

V = TypeVar('V', bound=float | Array[Any, np.float64])


# scipy.integrate

QuadWeights: TypeAlias = Literal[
    'cos',
    'sin',
    'alg',
    'alg-loga',
    'alg-logb',
    'alg-log',
    'cauchy',
]


class QuadOptions(TypedDict, total=False):
    """
    Optional quadrature options to be passed to
    [`scipy.integrate.quad`][scipy.integrate.quad].
    """
    epsabs: float
    epsrel: float
    limit: int
    limlst: int
    points: Array[tuple[int], np.floating[Any]] | Sequence[float]
    weight: QuadWeights
    wvar: float | tuple[float, float]
    wopts: tuple[int, Array[tuple[Literal[25], int], np.floating[Any]]]
    maxp1: int


# scipy.optimize


@runtime_checkable
class OptimizeResult(Protocol):
    """
    Type stub for the most generally available attributes of
    [`scipy.optimize.OptimizeResult`][scipy.optimize.OptimizeResult].

    Note that other attributes might be present as well, e.g. `jac` or `hess`.
    But these are currently impossible to type, as there's no way to define
    optional attributes in protocols.

    Note that `OptimizeResult` is actually subclasses dict, whose attributes
    are keys in disguise. But because `collections.abc.(Mutable)Mapping`
    aren't pure protocols (which makes no sense from a theoretical standpoint),
    they cannot be used as a superclass to another protocol. Basically this
    means that nothing in `collections.abc` can be used when writing type
    stubs...
    """
    x: Array[tuple[int], np.float64]
    success: bool
    status: int
    fun: float
    message: LiteralString
    nfev: int
    nit: int

    def __getitem__(self, k: str, /) -> object: ...
    def __setitem__(self, k: str, v: object, /) -> None: ...
    def __delitem__(self, k: str, /) -> None: ...
    def __contains__(self, k: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __reversed__(self) -> Iterator[str]: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[object]: ...
    def items(self) -> ItemsView[str, object]: ...
    @overload
    def get(self, k: str, /) -> object: ...
    @overload
    def get(self, k: str, d: object, /) -> object: ...
    @overload
    def pop(self, k: str, /) -> object: ...
    @overload
    def pop(self, k: str, d: object, /) -> object: ...
    def copy(self) -> dict[str, object]: ...


# scipy.stats

_RNG: TypeAlias = np.random.Generator | np.random.RandomState
_Seed: TypeAlias = _RNG | int | None
_Moments1: TypeAlias = Literal['m', 'v', 's', 'k']
_Moments2: TypeAlias = Literal['mv', 'ms', 'mk', 'vs', 'vk', 'sk']
_Moments3: TypeAlias = Literal['mvs', 'mvk', 'msk', 'vsk']
_Moments4: TypeAlias = Literal['mvsk']

_F8: TypeAlias = np.float64
_Real0D: TypeAlias = AnyScalarFloat
_Real1D: TypeAlias = AnyVectorFloat

_ND0 = TypeVar('_ND0', bound=AtLeast0D)
_ND1 = TypeVar('_ND1', bound=AtLeast1D)


@runtime_checkable
class RV(Protocol):
    a: float
    b: float
    badvalue: float
    moment_type: Literal[0, 1]
    name: LiteralString
    numargs: int
    shapes: LiteralString | None
    xtol: float

    @property
    def random_state(self) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed) -> None: ...

    def freeze(self, *args: _Real0D, **kwds: _Real0D) -> RVFrozen[Self]: ...
    def __call__(self, *args: _Real0D, **kwds: _Real0D) -> RVFrozen[Self]: ...

    @overload
    def rvs(
        self,
        *args: _Real0D,
        size: None = ...,
        random_state: _Seed = ...,
        **kwds: _Real0D,
    ) -> _F8: ...
    @overload
    def rvs(
        self,
        *args: _Real0D,
        size: int,
        random_state: _Seed = ...,
        **kwds: _Real0D,
    ) -> Array[tuple[int], _F8]: ...
    @overload
    def rvs(
        self,
        *args: _Real0D,
        size: _ND0,
        random_state: _Seed = ...,
        **kwds: _Real0D,
    ) -> Array[_ND0, _F8]: ...

    @overload
    def stats(
        self,
        *args: _Real0D,
        moments: _Moments1,
        **kwds: _Real0D,
    ) -> _F8: ...
    @overload
    def stats(
        self,
        *args: _Real0D,
        moments: _Moments2 = ...,
        **kwds: _Real0D,
    ) -> tuple[_F8, _F8]: ...
    @overload
    def stats(
        self,
        *args: _Real0D,
        moments: _Moments3 = ...,
        **kwds: _Real0D,
    ) -> tuple[_F8, _F8, _F8]: ...
    @overload
    def stats(
        self,
        *args: _Real0D,
        moments: _Moments4 = ...,
        **kwds: _Real0D,
    ) -> tuple[_F8, _F8, _F8, _F8]: ...

    def moment(
        self,
        order: Integer,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> _F8: ...

    def entropy(self, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    def median(self, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    def mean(self, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    def var(self, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    def std(self, *args: _Real0D, **kwds: _Real0D) -> _F8: ...

    @overload
    def interval(
        self,
        confidence: _Real0D,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[_F8, _F8]: ...
    @overload
    def interval(
        self,
        confidence: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[CanArray[_ND1, _F8], CanArray[_ND1, _F8]]: ...
    @overload
    def interval(
        self,
        confidence: Sequence[npt.ArrayLike],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[CanArray[AtLeast1D, _F8], CanArray[AtLeast1D, _F8]]: ...

    def support(self, *args: _Real0D, **kwds: _Real0D) -> tuple[_F8, _F8]: ...

    @overload
    def nnlf(self, __theta: _Real1D, __x: _Real1D) -> _F8: ...
    @overload
    def nnlf(
        self,
        __theta: _Real1D,
        __x: CanArray[AtLeast2D, Real],
    ) -> Array[AtLeast1D, _F8]: ...


@runtime_checkable
class RVDiscrete(RV, Protocol):
    # TODO: __init__
    # TODO: expect

    @property
    def inc(self) -> int: ...

    @overload
    def pmf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def pmf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logpmf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logpmf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def cdf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def cdf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logcdf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logcdf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def sf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def sf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logsf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logsf(
        self,
        k: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def ppf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def ppf(
        self,
        q: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def isf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def isf(
        self,
        q: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...


@runtime_checkable
class RVContinuous(RV, Protocol):
    # TODO: __init__
    # TODO: expect

    @overload
    def pdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def pdf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logpdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logpdf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def cdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def cdf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logcdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logcdf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def sf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def sf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def logsf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logsf(
        self,
        x: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def ppf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def ppf(
        self,
        q: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    @overload
    def isf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def isf(
        self,
        q: CanArray[_ND1, Real],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> Array[_ND1, _F8]: ...

    def fit(
        self,
        data: npt.ArrayLike,
        *args: _Real0D,
        optimizer: Callable[..., _Real1D] = ...,
        method: Literal['MLE', 'MM'] = ...,
        floc: _Real0D = ...,
        fscale: _Real0D = ...,
        **kwds: _Real0D,
    ) -> tuple[float | Real, ...]: ...

    def fit_loc_scale(
        self,
        data: npt.ArrayLike,
        *args: _Real0D,
    ) -> tuple[_F8, _F8]: ...


_RV_co = TypeVar('_RV_co', bound=RV, covariant=True)


@runtime_checkable
class RVFrozen(Protocol[_RV_co]):
    """Currently limited to scalar arguments."""

    def __init__(
        self,
        __dist: _RV_co,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> None: ...

    @property
    def args(self) -> tuple[_Real0D, ...]: ...
    @property
    def kwds(self) -> dict[str, _Real0D]: ...
    @property
    def dist(self) -> _RV_co: ...
    @property
    def a(self) -> float: ...
    @property
    def b(self) -> float: ...

    @property
    def random_state(self) -> _RNG: ...
    @random_state.setter
    def random_state(self, __seed: _Seed) -> None: ...

    @overload
    def rvs(self, size: None = ..., random_state: _Seed = ...) -> _F8: ...
    @overload
    def rvs(
        self,
        size: int,
        random_state: _Seed = ...,
    ) -> Array[tuple[int], _F8]: ...
    @overload
    def rvs(
        self,
        size: _ND0,
        random_state: _Seed = ...,
    ) -> Array[_ND0, _F8]: ...

    @overload
    def cdf(self, __x: _Real0D) -> _F8: ...
    @overload
    def cdf(self, __x: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...

    @overload
    def logcdf(self, __x: _Real0D) -> _F8: ...
    @overload
    def logcdf(self, __x: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...

    @overload
    def sf(self, __x: _Real0D) -> _F8: ...
    @overload
    def sf(self, __x: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...

    @overload
    def ppf(self, __q: _Real0D) -> _F8: ...
    @overload
    def ppf(self, __q: CanArray[_ND1, Real]) -> CanArray[_ND1, _F8]: ...

    @overload
    def isf(self, __q: _Real0D) -> _F8: ...
    @overload
    def isf(self, __q: CanArray[_ND1, Real]) -> CanArray[_ND1, _F8]: ...

    @overload
    def stats(self, moments: _Moments1) -> _F8: ...
    @overload
    def stats(self, moments: _Moments2 = ...) -> tuple[_F8, _F8]: ...
    @overload
    def stats(self, moments: _Moments3) -> tuple[_F8, _F8, _F8]: ...
    @overload
    def stats(self, moments: _Moments4) -> tuple[_F8, _F8, _F8, _F8]: ...

    def moment(self, order: Integer) -> _F8: ...

    def median(self) -> _F8: ...
    def mean(self) -> _F8: ...
    def var(self) -> _F8: ...
    def std(self) -> _F8: ...
    def entropy(self) -> _F8: ...

    @overload
    def interval(self, confidence: _Real0D) -> tuple[_F8, _F8]: ...
    @overload
    def interval(
        self,
        confidence: CanArray[_ND1, Real],
    ) -> tuple[CanArray[_ND1, _F8], CanArray[_ND1, _F8]]: ...
    @overload
    def interval(
        self,
        confidence: Sequence[npt.ArrayLike],
    ) -> tuple[CanArray[AtLeast1D, _F8], CanArray[AtLeast1D, _F8]]: ...

    def support(self) -> tuple[_F8, _F8]: ...

    def expect(
        self,
        func: Callable[[_F8], float] | None = ...,
        lb: _Real0D | None = ...,
        ub: _Real0D | None = ...,
        conditional: bool = ...,
        **kwds: Any,
    ) -> _F8: ...


_RV_D_co = TypeVar('_RV_D_co', bound=RVDiscrete, covariant=True)


@runtime_checkable
class RVDiscreteFrozen(RVFrozen[_RV_D_co], Protocol[_RV_D_co]):
    @overload
    def pmf(self, k: _Real0D) -> _F8: ...
    @overload
    def pmf(self, k: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...

    @overload
    def logpmf(self, k: _Real0D) -> _F8: ...
    @overload
    def logpmf(self, k: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...


_RV_C_co = TypeVar('_RV_C_co', bound=RVContinuous, covariant=True)


@runtime_checkable
class RVContinuousFrozen(RVFrozen[_RV_C_co], Protocol[_RV_C_co]):
    @overload
    def pdf(self, x: _Real0D) -> _F8: ...
    @overload
    def pdf(self, x: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...

    @overload
    def logpdf(self, x: _Real0D) -> _F8: ...
    @overload
    def logpdf(self, x: CanArray[_ND1, Real]) -> Array[_ND1, _F8]: ...
