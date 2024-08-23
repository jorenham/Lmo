"""SciPy-related type aliases for internal use."""

# ruff: noqa: D102, D105, D107
# pyright: reportPropertyTypeMismatch=false, reportMissingSuperCall=false

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

import lmo.typing.np as lnpt


if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        Unpack,
        overload,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        LiteralString,
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        Unpack,
        overload,
        runtime_checkable,
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

    import optype.numpy as onpt

__all__ = (
    'RV',
    'FitResult',
    'OptimizeResult',
    'QuadWeights',
    'QuadOptions',
    'RVContinuous',
    'RVContinuousFrozen',
    'RVDiscrete',
    'RVDiscreteFrozen',
    'RVFrozen',
    'RVFunction',
)


_T = TypeVar('_T')
_Tss = ParamSpec('_Tss')

_ShapeT0 = TypeVar('_ShapeT0', bound=tuple[int, ...])
_ShapeT1 = TypeVar('_ShapeT1', bound='onpt.AtLeast1D')

_Tuple2: TypeAlias = tuple[_T, _T]

_RNG: TypeAlias = np.random.Generator | np.random.RandomState
_Seed: TypeAlias = _RNG | int | None

_Moments1: TypeAlias = Literal['m', 'v', 's', 'k']
_Moments2: TypeAlias = Literal['mv', 'ms', 'mk', 'vs', 'vk', 'sk']
_Moments3: TypeAlias = Literal['mvs', 'mvk', 'msk', 'vsk']
_Moments4: TypeAlias = Literal['mvsk']

_F8: TypeAlias = np.float64
_Real0D: TypeAlias = lnpt.AnyScalarInt | lnpt.AnyScalarFloat
_Real1D: TypeAlias = lnpt.AnyVectorInt | lnpt.AnyVectorFloat
_Real2D: TypeAlias = lnpt.AnyMatrixInt | lnpt.AnyMatrixFloat
_Real3ND: TypeAlias = lnpt.AnyTensorInt | lnpt.AnyTensorFloat


# scipy.integrate

QuadWeights: TypeAlias = Literal[
    'cos', 'sin', 'alg', 'alg-loga', 'alg-logb', 'alg-log', 'cauchy',
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
    points: lnpt.AnyVectorFloat
    weight: QuadWeights
    wvar: float | tuple[float, float]
    wopts: tuple[
        int,
        onpt.Array[tuple[Literal[25], int], np.floating[Any]],
    ]
    maxp1: int


# scipy.optimize


class _RichResult(Protocol):
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


@runtime_checkable
class OptimizeResult(_RichResult, Protocol):
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
    x: onpt.Array[tuple[int], np.float64]
    success: bool
    status: int
    fun: float
    message: LiteralString
    nfev: int
    nit: int


RootResultFlag: TypeAlias = Literal[
    'converged',
    'sign error',
    'convergence error',
    'value error',
    'no error',
]
RootScalarMethod: TypeAlias = Literal[
    'bisect',
    'brentq',
    'brenth',
    'ridder',
    'toms748',
    'newton',
    'secant',
    'halley',
]


@runtime_checkable
class RootResult(_RichResult, Protocol):
    """
    Runtime protocol for the
    [`scipy.optimize.RootResults`][scipy.optimize.RootResults] result class.

    Note:
        For naming consistency with `OptimizeResult`, the singular form was
        used, as opposed to the (plural) `scipy.optimize.RootResults`.
    """
    root: float
    iterations: int
    function_calls: int
    converged: bool
    flag: RootResultFlag | LiteralString
    method: RootScalarMethod


# scipy.stats

_T_params_co = TypeVar(
    '_T_params_co',
    bound=tuple[np.float64, ...],  # usually a namedtuple
    covariant=True,
    default=tuple[np.float64, np.float64],  # loc, scale
)

# placeholder for `matplotlib.axes.Axes`.
_PlotAxes: TypeAlias = Any
_PlotType: TypeAlias = Literal['hist', 'qq', 'pp', 'cdf']


@runtime_checkable
class FitResult(Protocol[_T_params_co]):
    """
    Type stub for the `scipy.stats.fit` result.

    Examples:
        Create a dummy fit result instance

        >>> import numpy as np
        >>> from scipy.stats import fit, norm, bernoulli
        >>> data = [0]
        >>> isinstance(fit(norm, data), FitResult)
        True
        >>> isinstance(fit(bernoulli, data, [(0, 1)]), FitResult)
        True
    """
    discrete: bool
    success: bool
    message: str | None

    @property
    def params(self) -> _T_params_co: ...

    def plot(
        self,
        ax: _PlotAxes | None = ...,
        *,
        plot_type: _PlotType = ...,
    ) -> _PlotAxes: ...


class RVFunction(Protocol[_Tss]):
    """
    Callable protocol for a vectorized distribution function. E.g. for
    the `cdf` and `ppf` methods of `scipy,stats.rv_generic`. In practice,
    the returned dtype is always `float64` (even `rv_discrete.ppf`).
    """
    @overload
    def __call__(
        self,
        x: lnpt.AnyArrayFloat,
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> onpt.Array[Any, np.float64]: ...
    @overload
    def __call__(
        self,
        x: lnpt.AnyScalarFloat,
        /,
        *args: _Tss.args,
        **kwds: _Tss.kwargs,
    ) -> float: ...


# TODO:
AnyFrozenRV: TypeAlias = Any
AnyArrayOrScalar: TypeAlias = np.generic | npt.NDArray[np.generic]


@runtime_checkable
class AnyRV(Protocol):
    """
    Compatible with pyright's inferred types of
    `scipy.stats.distributions.rv_generic`.
    """
    a: Final[float | None]
    b: Final[float | None]
    name: Final[str]
    badvalue: float
    numargs: int
    shapes: LiteralString | None
    # moment_type: Literal[0, 1]
    # xtol: float

    @property
    def random_state(self, /) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed, /) -> None: ...

    def __init__(self, /, *args: Any, **kwds: Any) -> None: ...
    def __call__(self, /, *args: Any, **kwds: Any) -> AnyFrozenRV: ...
    def freeze(self, /, *args: Any, **kwds: Any) -> AnyFrozenRV: ...

    def rvs(self, /, *args: Any, **kwds: Any) -> Any: ...
    def stats(
        self, /, *args: Any, **kwds: Any,
    ) -> AnyArrayOrScalar | tuple[AnyArrayOrScalar, ...]: ...
    def entropy(self, /, *args: Any, **kwds: Any) -> AnyArrayOrScalar: ...
    def median(self, /, *args: Any, **kwds: Any) -> AnyArrayOrScalar: ...
    def mean(
        self, /, *args: Any, **kwds: Any,
    ) -> AnyArrayOrScalar | tuple[AnyArrayOrScalar, ...]: ...
    def var(
        self, /, *args: Any, **kwds: Any,
    ) -> AnyArrayOrScalar | tuple[AnyArrayOrScalar, ...]: ...
    def std(self, /, *args: Any, **kwds: Any) -> AnyArrayOrScalar: ...

    @overload
    def interval(
        self,
        confidence: _Real0D,
        /,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[_F8, _F8]: ...
    @overload
    def interval(
        self,
        confidence: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        /,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[onpt.Array[_ShapeT1, _F8], onpt.Array[_ShapeT1, _F8]]: ...
    @overload
    def interval(
        self,
        confidence: Sequence[npt.ArrayLike],
        /,
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> _Tuple2[npt.NDArray[_F8]]: ...

    def support(
        self, /, *args: Any, **kwds: Any,
    ) ->  _Tuple2[np.generic] | _Tuple2[npt.NDArray[np.generic]]: ...

    def nnlf(self, theta: _Real1D, x: _Real1D, /) -> float: ...


@runtime_checkable
class RV(Protocol):
    """Runtime-checkable interface for `scipy.stats.rv_generic`."""
    a: float
    b: float
    name: LiteralString
    badvalue: float
    numargs: int
    shapes: LiteralString | None
    # moment_type: Literal[0, 1]
    # xtol: float

    @property
    def random_state(self) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed) -> None: ...

    def __init__(self, seed: _Seed = ...) -> None: ...

    def freeze(self, /, *args: _Real0D, **kwds: _Real0D) -> RVFrozen[Self]: ...
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
    ) -> onpt.Array[tuple[int], _F8]: ...
    @overload
    def rvs(
        self,
        *args: _Real0D,
        size: _ShapeT0,
        random_state: _Seed = ...,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT0, _F8]: ...

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
        order: int | np.integer[Any],
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
        confidence: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> tuple[onpt.Array[_ShapeT1, _F8], onpt.Array[_ShapeT1, _F8]]: ...
    @overload
    def interval(
        self,
        confidence: Sequence[npt.ArrayLike],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> _Tuple2[npt.NDArray[_F8]]: ...

    def support(self, *args: _Real0D, **kwds: _Real0D) -> tuple[_F8, _F8]: ...

    @overload
    def nnlf(self, theta: _Real1D, x: _Real1D, /) -> _F8: ...
    @overload
    def nnlf(
        self,
        theta: _Real1D,
        x: _Real2D | _Real3ND,
    /) -> npt.NDArray[_F8]: ...


@runtime_checkable
class RVDiscrete(RV, Protocol):
    """
    Runtime-checkable interface for discrete probability distributions,
    like [`scipy.stats.rv_discrete`][scipy.stats.rv_discrete] subtype
    instances.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import distributions as distrs
        >>> isinstance(distrs.binom, RVDiscrete)
        True

        Continuous distributions aren't included:

        >>> isinstance(distrs.norm, RVDiscrete)
        False

        Note that for "frozen" distributions (a.k.a. random variables),
        this is not the case:

        >>> isinstance(distrs.binom(5, .42), RVDiscrete)
        False
    """

    @property
    def inc(self) -> int: ...

    def __init__(
        self,
        a: float | None = ...,
        b: float | None = ...,
        name: LiteralString | None = ...,
        badvalue: float | None = ...,
        moment_tol: float = ...,
        values: tuple[npt.ArrayLike, npt.ArrayLike] | None = ...,
        inc: int = ...,
        longname: LiteralString | None = ...,
        shapes: LiteralString | None = ...,
        seed: _Seed = ...,
    ) -> None: ...

    @overload
    def pmf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def pmf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logpmf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logpmf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def cdf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def cdf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logcdf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logcdf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def sf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def sf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logsf(self, k: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logsf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def ppf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def ppf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def isf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def isf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    def expect(
        self,
        func: Callable[[_F8], float] | None = ...,
        args: tuple[_Real0D, ...] = ...,
        loc: _Real0D = ...,
        lb: _Real0D | None = ...,
        ub: _Real0D | None = ...,
        conditional: bool = ...,
        maxcount: int = ...,
        tolerance: float = ...,
        chunksize: int = ...,
    ) -> _F8: ...


@runtime_checkable
class RVContinuous(RV, Protocol):
    """
    Runtime-checkable interface for continuous probability distributions,
    like [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] subtype
    instances.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import distributions as distrs
        >>> from lmo import distributions as l_distrs
        >>> isinstance(distrs.norm, RVContinuous)
        True
        >>> isinstance(l_distrs.wakeby, RVContinuous)
        True

        This also works if `rv_continuous` isn't a base class, but it
        looks and quacks like one, e.g. [`l_poly`][lmo.distributions.l_poly].

        >>> isinstance(l_distrs.l_poly, RVContinuous)
        True

        Discrete distributions aren't included:

        >>> isinstance(distrs.binom, RVContinuous)
        False

        Note that for "frozen" distributions (a.k.a. random variables),
        this is not the case:

        >>> isinstance(distrs.norm(), RVContinuous)
        False
        >>> isinstance(l_distrs.wakeby(5, 1, .5), RVContinuous)
        False
    """
    def __init__(
        self,
        momtype: Literal[0, 1] = ...,
        a: float | None = ...,
        b: float | None = ...,
        xtol: float = ...,
        badvalue: float | None = ...,
        name: LiteralString | None = ...,
        longname: LiteralString | None = ...,
        shapes: LiteralString | None = ...,
        seed: _Seed = ...,
    ) -> None: ...

    @overload
    def pdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def pdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logpdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logpdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def cdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def cdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logcdf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logcdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def sf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def sf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logsf(self, x: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def logsf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def ppf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def ppf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def isf(self, q: _Real0D, *args: _Real0D, **kwds: _Real0D) -> _F8: ...
    @overload
    def isf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
        *args: _Real0D,
        **kwds: _Real0D,
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    def fit(
        self,
        data: lnpt.AnyArray,
        *args: _Real0D,
        optimizer: Callable[..., _Real1D] = ...,
        method: Literal['MLE', 'MM'] = ...,
        floc: _Real0D = ...,
        fscale: _Real0D = ...,
        **kwds: _Real0D,
    ) -> tuple[float | lnpt.Real, ...]: ...

    # def fit_loc_scale(
    #     self,
    #     data: npt.ArrayLike,
    #     *args: _Real0D,
    # ) -> tuple[_F8, _F8]: ...

    def expect(
        self,
        func: Callable[[_F8], float] | None = ...,
        args: tuple[_Real0D, ...] = ...,
        loc: _Real0D = ...,
        scale: _Real0D = ...,
        lb: _Real0D | None = ...,
        ub: _Real0D | None = ...,
        conditional: bool = ...,
        **kwds: Unpack[QuadOptions],
    ) -> _F8: ...


_RV_co = TypeVar('_RV_co', bound=RV, covariant=True)


@runtime_checkable
class RVFrozen(Protocol[_RV_co]):
    """Currently limited to scalar arguments."""

    def __init__(
        self,
        dist: _RV_co,
        /,
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
    def random_state(self, seed: _Seed, /) -> None: ...

    @overload
    def rvs(self, size: None = ..., random_state: _Seed = ...) -> _F8: ...
    @overload
    def rvs(
        self,
        size: int,
        random_state: _Seed = ...,
    ) -> onpt.Array[tuple[int], _F8]: ...
    @overload
    def rvs(
        self,
        size: _ShapeT0,
        random_state: _Seed = ...,
    ) -> onpt.Array[_ShapeT0, _F8]: ...

    @overload
    def cdf(self, x: _Real0D) -> _F8: ...
    @overload
    def cdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logcdf(self, x: _Real0D) -> _F8: ...
    @overload
    def logcdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def sf(self, x: _Real0D) -> _F8: ...
    @overload
    def sf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def ppf(self, q: _Real0D) -> _F8: ...
    @overload
    def ppf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def isf(self, q: _Real0D) -> _F8: ...
    @overload
    def isf(
        self,
        q: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8] | _F8: ...

    @overload
    def stats(self, moments: _Moments1) -> _F8: ...
    @overload
    def stats(self, moments: _Moments2 = ...) -> tuple[_F8, _F8]: ...
    @overload
    def stats(self, moments: _Moments3) -> tuple[_F8, _F8, _F8]: ...
    @overload
    def stats(self, moments: _Moments4) -> tuple[_F8, _F8, _F8, _F8]: ...

    def moment(self, order: int | np.integer[Any]) -> _F8: ...

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
        confidence: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> _Tuple2[onpt.Array[_ShapeT1, _F8] | _F8]: ...
    @overload
    def interval(
        self,
        confidence: Sequence[npt.ArrayLike],
    ) -> _Tuple2[onpt.Array[onpt.AtLeast1D, _F8] | _F8]: ...

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
    """
    Runtime-checkable interface for discrete probability distributions,
    like [`scipy.stats.rv_discrete`][scipy.stats.rv_discrete] subtype
    instances.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import distributions as distrs
        >>> isinstance(distrs.bernoulli, RVDiscreteFrozen)
        False
        >>> isinstance(distrs.bernoulli(.42), RVDiscreteFrozen)
        True
        >>> isinstance(distrs.uniform(), RVDiscreteFrozen)
        False
    """
    @overload
    def pmf(self, k: _Real0D) -> _F8: ...
    @overload
    def pmf(
        self,
        k: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logpmf(self, k: _Real0D) -> _F8: ...
    @overload
    def logpmf(self, k: lnpt.AnyArrayFloat) -> onpt.Array[_ShapeT1, _F8]: ...


_RV_C_co = TypeVar('_RV_C_co', bound=RVContinuous, covariant=True)


@runtime_checkable
class RVContinuousFrozen(RVFrozen[_RV_C_co], Protocol[_RV_C_co]):
    """
    Runtime-checkable interface for discrete probability distributions,
    like [`scipy.stats.rv_discrete`][scipy.stats.rv_discrete] subtype
    instances.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import distributions as distrs
        >>> isinstance(distrs.uniform, RVContinuousFrozen)
        False
        >>> isinstance(distrs.uniform(), RVContinuousFrozen)
        True
        >>> isinstance(distrs.bernoulli(.5), RVContinuousFrozen)
        False

        >>> from lmo.distributions import l_poly
        >>> isinstance(l_poly([0, 1/6]), RVContinuousFrozen)
        True
        >>> isinstance(l_poly, RVContinuousFrozen)
        True
    """

    @overload
    def pdf(self, x: _Real0D) -> _F8: ...
    @overload
    def pdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...

    @overload
    def logpdf(self, x: _Real0D) -> _F8: ...
    @overload
    def logpdf(
        self,
        x: onpt.CanArray[_ShapeT1, np.dtype[lnpt.Real]],
    ) -> onpt.Array[_ShapeT1, _F8]: ...
