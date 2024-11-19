"""SciPy-related type aliases for internal use."""

# ruff: noqa: PLC2701, D102

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, overload

import numpy as np
import numpy.typing as npt
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen, rv_generic

if sys.version_info >= (3, 13):
    from typing import ParamSpec, Protocol
else:
    from typing_extensions import ParamSpec, Protocol

if TYPE_CHECKING:
    import optype.numpy as onp
    from numpy._typing import _ArrayLikeFloat_co  # pyright: ignore[reportPrivateUsage]

    import lmo.typing.np as lnpt


__all__ = "RV", "QuadOptions", "QuadWeights", "RVContinuous", "RVFrozen", "RVFunction"


def __dir__() -> tuple[str, ...]:
    return __all__


_Tss = ParamSpec("_Tss")

# scipy.integrate

QuadWeights: TypeAlias = Literal[
    "cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy"
]

_IntLike: TypeAlias = int | np.integer[Any]
_FloatLike: TypeAlias = float | np.floating[Any]


class QuadOptions(TypedDict, total=False):
    """
    Optional quadrature options to be passed to
    [`scipy.integrate.quad`][scipy.integrate.quad].
    """

    epsabs: _FloatLike
    epsrel: _FloatLike
    limit: _IntLike
    points: _ArrayLikeFloat_co
    weight: QuadWeights
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike]
    wopts: tuple[_IntLike, npt.NDArray[np.float32 | np.float64]]


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
    ) -> onp.Array[Any, np.float64]: ...
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
