# ruff: noqa: D105
"""Numpy-related type aliases for internal use."""

__all__ = (
    'NP_V2',
    'Bool', 'Int', 'Float', 'Natural', 'Integer', 'Real',
    'AtLeast0D', 'AtLeast1D', 'AtLeast2D', 'AtLeast3D',
    'Array', 'CanArray',
    'AnyScalarBool', 'AnyVectorBool', 'AnyMatrixBool', 'AnyTensorBool',
    'AnyScalarInt', 'AnyVectorInt', 'AnyMatrixInt', 'AnyTensorInt',
    'AnyScalarFloat', 'AnyVectorFloat', 'AnyMatrixFloat', 'AnyTensorFloat',
    'SortKind',
    'Order',
    'RandomState',
)

from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from optype import CanSequence

from .compat import Unpack


NP_V2: Final[bool] = np.__version__.startswith('2.')


# Some handy scalar type aliases

if NP_V2:
    Bool: TypeAlias = np.bool   # noqa: NPY001
else:
    Bool: TypeAlias = np.bool_
UInt: TypeAlias = np.unsignedinteger[Any]
Int: TypeAlias = np.signedinteger[Any]
Float: TypeAlias = np.floating[Any]

Natural: TypeAlias = UInt | Bool
Integer: TypeAlias = Int | Natural
Real: TypeAlias = Float | Integer


# Shapes

AtLeast0D: TypeAlias = tuple[int, ...]
AtLeast1D: TypeAlias = tuple[int, Unpack[AtLeast0D]]
AtLeast2D: TypeAlias = tuple[int, Unpack[AtLeast1D]]
AtLeast3D: TypeAlias = tuple[int, Unpack[AtLeast2D]]


# Array and array-likes, with generic shape

_DN = TypeVar('_DN', bound=tuple[int, ...])
_ST = TypeVar('_ST', bound=np.generic)
_ST_co = TypeVar('_ST_co', bound=np.generic, covariant=True)

Array: TypeAlias = np.ndarray[_DN, np.dtype[_ST]]


@runtime_checkable
class CanArray(Protocol[_DN, _ST_co]):
    """
    Anything that can be converted to a (numpy) array, e.g. with `np.asarray`,
    similarly to `collections.abc.Sequence`.

    Specifically, this includes instances of types that implement the
    `__array__` method (which would return a `np.ndarray`).

    Note that `isinstance` can also be used, but keep in mind that due to the
    performance-first implementation of `typing.runtime_checkable`, it often
    will lead to false positives. So keep in mind that at runtime,
    `isinstance(x, CanArray)` is the (cached) equivalent of
    `inspect.getattr_static(x, '__array__', nil := object()) is not nil`.


    Examples:
        >>> isinstance([1, 2, 3], CanArray)
        False
        >>> isinstance(np.array([1, 2, 3]), CanArray)
        True
        >>> isinstance(np.ma.array([1, 2, 3]), CanArray)
        True
        >>> isinstance(np.matrix([[1, 0], [0, 1]]), CanArray)
        True

        Note that `numpy.generic` instances (which numpy calls "scalars",
        even though anyone that knows a bit of linear algebra (i.e. the
        entirety of numpy's audience) will find this very confusing)
        also implement the `__array__` method, returning a 0-dimensional
        `np.ndarray`, i.e. `__array__: (T: np.generic) -> Array0D[T]`:

        >>> isinstance(np.uint(42), CanArray)
        True
    """
    def __array__(self) -> Array[_DN, _ST_co]: ...


# `str` and `bytes` tend to complicate things because they're sequences
# themselves, and aren't even relevant for Lmo, so they're omitted here.
_ST_py = TypeVar('_ST_py', bool, int, float, complex)

_T = TypeVar('_T')
_PyVector: TypeAlias = CanSequence[int, _T]


_AnyScalar: TypeAlias = _ST | _ST_py | CanArray[tuple[()], _ST]
_AnyVector: TypeAlias = (
    CanArray[tuple[int], _ST]
    | _PyVector[_AnyScalar[_ST, _ST_py]]
)
_AnyMatrix: TypeAlias = (
    CanArray[tuple[int, int], _ST]
    | _PyVector[_AnyVector[_ST, _ST_py]]
)

# these will result in {0,1,2,N}-D arrays when passed to `np.array` (no need
# for a broken "nested sequence" type)

AnyScalarBool: TypeAlias = _AnyScalar[Bool, bool]
AnyVectorBool: TypeAlias = _AnyVector[Bool, bool]
AnyMatrixBool: TypeAlias = _AnyMatrix[Bool, bool]
AnyTensorBool: TypeAlias = (
    CanArray[AtLeast3D, Bool]
    | _PyVector[AnyMatrixBool]
    | _PyVector['AnyTensorBool']
)

AnyScalarInt: TypeAlias = _AnyScalar[Integer, int]
AnyVectorInt: TypeAlias = _AnyVector[Integer, int]
AnyMatrixInt: TypeAlias = _AnyMatrix[Integer, int]
AnyTensorInt: TypeAlias = (
    CanArray[AtLeast3D, Integer]
    | _PyVector[AnyMatrixInt]
    | _PyVector['AnyTensorInt']
)

AnyScalarFloat: TypeAlias = _AnyScalar[Real, float]
AnyVectorFloat: TypeAlias = _AnyVector[Real, float]
AnyMatrixFloat: TypeAlias = _AnyMatrix[Real, float]
AnyTensorFloat: TypeAlias = (
    CanArray[AtLeast3D, Real]
    | _PyVector[AnyMatrixFloat]
    | _PyVector['AnyTensorFloat']
)


# Various type aliases


Order: TypeAlias = Literal['C', 'F']
"""Type of the `order` parameter of e.g. [`np.empty`][numpy.empty]."""

OrderCopy: TypeAlias = Literal['K', 'A'] | Order
"""Type of the `order` parameter of e.g. [`np.array`][numpy.array]."""

SortKind: TypeAlias = Literal['quicksort', 'heapsort', 'stable']
"""
Type of the `kind` parameter of e.g. [`np.sort`][numpy.sort], as
allowed by numpy's own stubs.
Note that the actual implementation just looks at `kind[0].lower() == 'q'`.
This means that it's possible to select stable-sort by passing
`kind='SnailSort'` instead of `kind='stable'` (although your typechecker might
ruin the fun).
"""

RandomState: TypeAlias = np.random.Generator | np.random.RandomState
"""
Union of the [`numpy.random.Generator`][numpy.random.Generator] and the
(legacy) [`numpy.random.RandomState`][numpy.random.RandomState] "RNG" types,
that are mostly compatible.
"""

Seed: TypeAlias = (
    int
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
)
"""
Any acceptable "seed" type that can be passed to
[`numpy.random.default_rng`][numpy.random.default_rng].
"""
