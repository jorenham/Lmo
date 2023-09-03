# ruff: noqa: D102,D105,D107

"""Numpy-related type aliases for internal use."""

__all__ = (
    'SupportsArray',

    'AnyScalar',
    'AnyNDArray',

    'AnyBool',
    'AnyInt',
    'AnyFloat',

    'IntVector',
    'IntMatrix',
    'IntTensor',

    'FloatVector',
    'FloatMatrix',
    'FloatTensor',

    'SortKind',
    'IndexOrder',

    'PolySeries',

    'LMomentOptions',
    'LComomentOptions',

    'QuadOptions',

    'AnyTrim',
)

from collections.abc import Iterator, Sequence
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    SupportsInt,
    TypeAlias,
    TypeGuard,
    TypeVar,
    TypedDict,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

T = TypeVar('T', bound=np.generic)
T_co = TypeVar('T_co', covariant=True, bound=np.generic)


@runtime_checkable
class SupportsArray(Protocol[T_co]):
    """
    Custom numpy array containers.

    See Also:
        - https://numpy.org/doc/stable/user/basics.dispatch.html
    """
    def __array__(self) -> npt.NDArray[T_co]: ...


# scalar types
_NpBool: TypeAlias = np.bool_
_NpInt: TypeAlias = np.integer[Any]
_NpFloat: TypeAlias = np.floating[Any]
_NpComplex: TypeAlias = np.complexfloating[Any, Any]
_NpNumber: TypeAlias = np.number[Any] | _NpBool
_NpScalar: TypeAlias = np.generic

AnyBool: TypeAlias = bool | _NpBool
AnyInt: TypeAlias = int | _NpInt | _NpBool
AnyFloat: TypeAlias = float | _NpFloat | AnyInt
AnyComplex: TypeAlias = complex | _NpComplex | AnyFloat  # no float
AnyNumber: TypeAlias = int | float | complex | _NpNumber
AnyScalar: TypeAlias = int | float | complex | str | bytes | _NpScalar

# array-like flavours (still waiting on numpy's shape typing)
# - `{}Vector`: ndim == 1
# - `{}Matrix`: ndim == 2
# - `{}Tensor`: ndim >= 3
AnyNDArray: TypeAlias = npt.NDArray[T] | SupportsArray[T]

_ArrayZ: TypeAlias = AnyNDArray[_NpInt] | AnyNDArray[_NpBool]
IntVector: TypeAlias = _ArrayZ | Sequence[AnyInt]
IntMatrix: TypeAlias = _ArrayZ | Sequence[Sequence[AnyInt]]
IntTensor: TypeAlias = _ArrayZ | Sequence[IntMatrix | 'IntTensor']

_ArrayR: TypeAlias = AnyNDArray[_NpFloat] | _ArrayZ
FloatVector: TypeAlias = _ArrayR | Sequence[AnyFloat]
FloatMatrix: TypeAlias = _ArrayR | Sequence[Sequence[AnyFloat]]
FloatTensor: TypeAlias = _ArrayR | Sequence[FloatMatrix | 'FloatTensor']

_ArrayC: TypeAlias = AnyNDArray[_NpComplex] | _ArrayR
ComplexVector: TypeAlias = _ArrayC | Sequence[AnyComplex]
ComplexMatrix: TypeAlias = _ArrayC | Sequence[Sequence[AnyComplex]]
ComplexTensor: TypeAlias = _ArrayC | Sequence[ComplexMatrix | 'ComplexTensor']

# for numpy.sort
SortKind: TypeAlias = Literal['quicksort', 'heapsort', 'stable']
IndexOrder: TypeAlias = Literal['C', 'F', 'A', 'K']

# numpy.polynomial

@runtime_checkable
class _SupportsCoef(Protocol):
    coef: npt.NDArray[Any] | SupportsArray[Any]


@runtime_checkable
class _SupportsDomain(Protocol):
    domain: npt.NDArray[Any] | SupportsArray[Any]


@runtime_checkable
class _SupportsWindow(Protocol):
    window: npt.NDArray[Any] | SupportsArray[Any]

@runtime_checkable
class _SupportsLessThanInt(Protocol):
    def __lt__(self, __other: int) -> bool: ...

_P = TypeVar('_P', bound='PolySeries')


@runtime_checkable
class PolySeries(Protocol):
    """
    Annotations for the (private) `numpy.polynomial._polybase.ABCPolyBase`
    subtypes, e.g. [`numpy.polynomial.Legendre`][numpy.polynomial.Legendre].
    """
    __hash__: ClassVar[None]  # type: ignore[assignment]
    __array_ufunc__: ClassVar[None]
    maxpower: ClassVar[int]

    basis_name: str | None

    coef: npt.NDArray[_NpFloat | _NpComplex]
    domain: npt.NDArray[_NpInt | _NpFloat | _NpComplex]
    window: npt.NDArray[_NpInt | _NpFloat | _NpComplex]

    @property
    def symbol(self) -> str: ...

    def has_samecoef(self, __other: _SupportsCoef) -> bool: ...
    def has_samedomain(self, __other: _SupportsDomain) -> bool:...
    def has_samewindow(self, __other: _SupportsWindow) -> bool: ...
    def has_sametype(self: _P, __other: type[Any]) -> TypeGuard[type[_P]]: ...
    def __init__(
        self,
        coef: npt.ArrayLike,
        domain: ComplexVector | None = ...,
        window: ComplexVector | None = ...,
        symbol: str = ...,
    ) -> None: ...
    def __format__(self, __fmt_str: str) -> str: ...
    @overload
    def __call__(self, __arg: _P) -> _P: ...
    @overload
    def __call__(self, __arg: complex | _NpComplex) -> _NpComplex: ...
    @overload
    def __call__(self, __arg: AnyNumber) -> _NpFloat | _NpComplex: ...
    @overload
    def __call__(
        self,
        __arg: AnyNDArray[_NpNumber],
    ) -> npt.NDArray[_NpFloat] | npt.NDArray[_NpComplex]: ...
    def __iter__(self) -> Iterator[_NpFloat | _NpComplex]: ...
    def __len__(self) -> int: ...
    def __neg__(self: _P) -> _P: ...
    def __pos__(self: _P) -> _P: ...
    def __add__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __sub__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __mul__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __truediv__(self: _P, __other: AnyNumber) -> _P: ...
    def __floordiv__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __mod__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __divmod__(self: _P, __other: npt.ArrayLike | _P) -> tuple[_P, _P]: ...
    def __radd__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __rsub__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __rmul__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __rtruediv__(self: _P, __other: AnyNumber) -> _P: ...
    def __rfloordiv__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __rmod__(self: _P, __other: npt.ArrayLike | _P) -> _P: ...
    def __rdivmod__(
        self: _P,
        __other: npt.ArrayLike | _P,
    ) -> tuple[_P, _P]: ...
    def __pow__(self: _P, __other: AnyInt) -> _P: ...
    def __eq__(self, __other: Any) -> bool: ...
    def __ne__(self, __other: Any) -> bool: ...
    def copy(self: _P) -> _P: ...
    def degree(self) -> int: ...
    def cutdeg(self: _P, deg: SupportsInt) -> _P: ...
    def trim(self: _P, tol: AnyFloat | _SupportsLessThanInt = ...) -> _P: ...
    def truncate(self: _P, size: AnyInt) -> _P: ...
    @overload
    def convert(
        self,
        domain: ComplexVector | None = ...,
        *,
        kind: type[_P],
        window: ComplexVector = ...,
    ) -> _P: ...
    @overload
    def convert(
        self,
        domain: ComplexVector,
        kind: type[_P],
        window: ComplexVector = ...,
    ) -> _P: ...
    @overload
    def convert(
        self: _P,
        domain: ComplexVector = ...,
        kind: type[_P] | None = ...,
        window: ComplexVector = ...,
    ) -> _P: ...
    def mapparms(self) -> tuple[_NpFloat, _NpFloat]: ...
    def integ(
        self: _P,
        m: AnyInt = ...,
        k: npt.ArrayLike = ...,
        lbnd: AnyNumber | None = ...,
    ) -> _P: ...
    def deriv(self: _P, m: AnyInt = ...) -> _P: ...
    def roots(self) -> npt.NDArray[_NpFloat | _NpComplex]: ...
    def linspace(
        self,
        n: AnyInt = ...,
        domain: npt.ArrayLike | None = ...,
    ) -> tuple[
        npt.NDArray[_NpFloat | _NpComplex],
        npt.NDArray[_NpFloat | _NpComplex],
    ]: ...
    @overload
    @classmethod
    def fit(
        cls: type[_P],
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        deg: AnyInt | IntVector,
        domain: ComplexVector | None = ...,
        rcond: AnyFloat | None = ...,
        *,
        full: Literal[False],
        w: FloatVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> _P: ...
    @overload
    @classmethod
    def fit(
        cls: type[_P],
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        deg: AnyInt | IntVector,
        domain: ComplexVector | None = ...,
        rcond: AnyFloat | None = ...,
        *,
        full: Literal[True],
        w: FloatVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> tuple[_P, list[Any]]: ...
    @overload
    @classmethod
    def fit(
        cls: type[_P],
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        deg: AnyInt | IntVector,
        domain: ComplexVector | None = ...,
        rcond: AnyFloat | None = ...,
        full: bool = ...,
        w: FloatVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> _P: ...
    @classmethod
    def fromroots(
        cls: type[_P],
        roots: npt.ArrayLike,
        domain: ComplexVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> _P: ...
    @classmethod
    def identity(
        cls: type[_P],
        domain: ComplexVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> _P: ...
    @classmethod
    def basis(
        cls: type[_P],
        deg: AnyInt,
        domain: ComplexVector | None = ...,
        window: ComplexVector | None = ...,
        # symbol: str = ...,
    ) -> _P: ...
    @classmethod
    def cast(
        cls: type[_P],
        series: 'PolySeries',
        domain: ComplexVector | None = ...,
        window: ComplexVector | None = ...,
    ) -> _P: ...



# PEP 692 precise **kwargs typing

class _LOptions(TypedDict, total=False):
    sort: SortKind | None
    cache: bool

class LMomentOptions(_LOptions, total=False):
    """Use like `def spam(**kwargs: Unpack[LMomentOptions]): ...`."""
    fweights: IntVector | None
    aweights: npt.ArrayLike | None

class LComomentOptions(_LOptions, total=False):
    """Use like `def spam(**kwargs: Unpack[LComomentOptions]): ...`."""
    rowvar: bool


# scipy

class QuadOptions(TypedDict, total=False):
    """
    Optional quadrature options to be passed to
    [`scipy.integrate.quad`][scipy.integrate.quad].
    """
    epsabs: float
    epsrel: float
    limit: int
    maxp1: int
    limlst: int
    points: Sequence[float] | npt.NDArray[np.floating[Any]]
    weight: Literal[
        'cos',
        'sin',
        'alg',
        'alg-loga',
        'alg-logb',
        'alg-log',
        'cauchy',
    ]
    wvar: float | tuple[float, float]
    wopts: tuple[int, npt.NDArray[np.float_]]


# Lmo specific aliases

AnyTrim: TypeAlias = (
    tuple[AnyFloat, AnyFloat]
    | Sequence[AnyFloat]
    | SupportsArray[_NpInt | _NpFloat]
)
