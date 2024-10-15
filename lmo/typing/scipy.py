"""SciPy-related type aliases for internal use."""

# ruff: noqa: PLC2701, D102

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.stats._distn_infrastructure import (
    rv_continuous,
    rv_continuous_frozen,
    rv_discrete,
    rv_discrete_frozen,
    rv_frozen,
    rv_generic,
)


if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        ParamSpec,
        Protocol,
        TypeVar,
        overload,
        runtime_checkable,
    )
else:
    from typing import LiteralString, overload

    from typing_extensions import (
        ParamSpec,
        Protocol,
        TypeVar,
        runtime_checkable,
    )

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

    import optype as opt
    import optype.numpy as onpt

    import lmo.typing.np as lnpt

__all__ = (
    'FitResult',
    'OptimizeResult',
    'QuadWeights',
    'QuadOptions',
    'RVFunction',
    'RV',
    'RVFrozen',
    'RVContinuous',
    'RVContinuousFrozen',
    'RVDiscrete',
    'RVDiscreteFrozen',
)

_Tss = ParamSpec('_Tss')

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
    points: opt.CanGetitem[
        int,
        float | np.floating[Any] | np.integer[Any] | np.bool_,
    ]
    weight: QuadWeights
    wvar: float | tuple[float, float]
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]]

    limlst: int
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


RV: TypeAlias = rv_generic
RVFrozen: TypeAlias = rv_frozen
RVContinuous: TypeAlias = rv_continuous
RVContinuousFrozen: TypeAlias = rv_continuous_frozen
RVDiscrete: TypeAlias = rv_discrete
RVDiscreteFrozen: TypeAlias = rv_discrete_frozen
