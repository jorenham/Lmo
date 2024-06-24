# ruff: noqa: D105
"""Numpy-related type aliases for internal use."""
from collections.abc import Sequence
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

from .compat import Unpack


__all__ = (
    'NP_VERSION', 'NP_V2',
    'Bool', 'Int', 'Float', 'Natural', 'Integer', 'Real',
    'AtLeast0D', 'AtLeast1D', 'AtLeast2D', 'AtLeast3D',
    'Array', 'CanArray',
    'AnyScalar', 'AnyScalarBool', 'AnyScalarInt', 'AnyScalarFloat',
    'AnyVector', 'AnyVectorBool', 'AnyVectorFloat', 'AnyVectorFloat',
    'AnyMatrix', 'AnyMatrixBool', 'AnyMatrixInt', 'AnyMatrixFloat',
    'AnyTensor', 'AnyTensorBool', 'AnyTensorInt', 'AnyTensorFloat',
    'AnyArray', 'AnyArrayBool', 'AnyArrayInt', 'AnyArrayFloat',
    'AnyObjectDType', 'AnyBoolDType',
    'AnyUIntDType', 'AnyIntDType', 'AnyFloatDType',
    'SortKind',
    'Order', 'OrderReshape', 'OrderCopy',
    'RandomState',
    'Casting',
)

_NP_MAJOR: Final[int] = int(np.__version__.split('.', 1)[0])
_NP_MINOR: Final[int] = int(np.__version__.split('.', 2)[1])
NP_VERSION: Final[tuple[int, int]] = _NP_MAJOR, _NP_MINOR
NP_V2: Final[bool] = _NP_MAJOR == 2


# Some handy scalar type aliases

if NP_V2:
    Bool: TypeAlias = np.bool
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

_DN = TypeVar('_DN', bound=tuple[()] | tuple[int, ...])
_DN_co = TypeVar('_DN_co', bound=tuple[()] | tuple[int, ...], covariant=True)
_ST = TypeVar('_ST', bound=np.generic)
_ST_co = TypeVar('_ST_co', bound=np.generic, covariant=True)

Array: TypeAlias = np.ndarray[_DN, np.dtype[_ST]]


@runtime_checkable
class CanArray(Protocol[_DN_co, _ST_co]):  # pyright: ignore[reportInvalidTypeVarUse]
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

        Note that `numpy.generic` instances (which numpy calls "scalars",
        even though anyone that knows a bit of linear algebra (i.e. the
        entirety of numpy's audience) will find this very confusing)
        also implement the `__array__` method, returning a 0-dimensional
        `np.ndarray`, i.e. `__array__: (T: np.generic) -> Array0D[T]`:

        >>> isinstance(np.uint(42), CanArray)
        True
    """
    def __array__(self) -> Array[_DN_co, _ST_co]: ...


_PyScalar: TypeAlias = bool | int | float | complex | str | bytes
# _PyScalar: TypeAlias = bool | int | float | complex
_ST_py = TypeVar('_ST_py', bound=_PyScalar)

_T = TypeVar('_T')
_PyVector: TypeAlias = Sequence[_T]


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

AnyScalar: TypeAlias = _AnyScalar[np.generic, _PyScalar]
AnyVector: TypeAlias = _AnyVector[np.generic, _PyScalar]
AnyMatrix: TypeAlias = _AnyMatrix[np.generic, _PyScalar]
AnyTensor: TypeAlias = (
    CanArray[AtLeast3D, np.generic]
    | _PyVector[AnyMatrix]
    | _PyVector['AnyTensor']
)
AnyArray: TypeAlias = AnyScalar | AnyVector | AnyMatrix | AnyTensor

AnyScalarBool: TypeAlias = _AnyScalar[Bool, bool]
AnyVectorBool: TypeAlias = _AnyVector[Bool, bool]
AnyMatrixBool: TypeAlias = _AnyMatrix[Bool, bool]
AnyTensorBool: TypeAlias = (
    CanArray[AtLeast3D, Bool]
    | _PyVector[AnyMatrixBool]
    | _PyVector['AnyTensorBool']
)
AnyArrayBool: TypeAlias = AnyVectorBool | AnyMatrixBool | AnyTensorBool

AnyScalarInt: TypeAlias = _AnyScalar[Integer, int]
AnyVectorInt: TypeAlias = _AnyVector[Integer, int]
AnyMatrixInt: TypeAlias = _AnyMatrix[Integer, int]
AnyTensorInt: TypeAlias = (
    CanArray[AtLeast3D, Integer]
    | _PyVector[AnyMatrixInt]
    | _PyVector['AnyTensorInt']
)
AnyArrayInt: TypeAlias = AnyVectorInt | AnyMatrixInt | AnyTensorInt

AnyScalarFloat: TypeAlias = _AnyScalar[Real, float]
AnyVectorFloat: TypeAlias = _AnyVector[Real, float]
AnyMatrixFloat: TypeAlias = _AnyMatrix[Real, float]
AnyTensorFloat: TypeAlias = (
    CanArray[AtLeast1D, Real]
    | _PyVector[AnyMatrixFloat]
    | _PyVector['AnyTensorFloat']
)
AnyArrayFloat: TypeAlias = AnyVectorFloat | AnyMatrixFloat | AnyTensorFloat

# some of the allowed `np.dtype` argument

_AnyDType: TypeAlias = np.dtype[_ST] | type[_ST]
AnyObjectDType: TypeAlias = (
    _AnyDType[np.object_]
    | Literal['object', 'object_', 'O', '=O', '<O', '>O']
)
AnyBoolDType: TypeAlias = (
    _AnyDType[np.bool_]
    | type[bool]
    | Literal['bool', 'bool_', '?', '=?', '<?', '>?']
)
AnyUIntDType: TypeAlias = _AnyDType[UInt] | Literal[
    'uint8', 'u1', '=u1', '<u1', '>u1',
    'uint16', 'u2', '=u2', '<u2', '>u2',
    'uint32', 'u4', '=u4', '<u4', '>u4',
    'uint64', 'u8', '=u8', '<u8', '>u8',
    'ubyte', 'B', '=B', '<B', '>B',
    'ushort', 'H', '=H', '<H', '>H',
    'uintc', 'I', '=I', '<I', '>I',
    'uintp', 'P', '=P', '<P', '>P',
    'uint', 'N', '=N', '<N', '>N',
    'ulong', 'L', '=L', '<L', '>L',
    'ulonglong', 'Q', '=Q', '<Q', '>Q',
]
AnyIntDType: TypeAlias = _AnyDType[Int] | Literal[
    'int8', 'i1', '=i1', '<i1', '>i1',
    'int16', 'i2', '=i2', '<i2', '>i2',
    'int32', 'i4', '=i4', '<i4', '>i4',
    'int64', 'i8', '=i8', '<i8', '>i8',
    'byte', 'b', '=b', '<b', '>b',
    'short', 'h', '=h', '<h', '>h',
    'intc', 'i', '=i', '<i', '>i',
    'intp', 'p', '=p', '<p', '>p',
    'int', 'int_', 'n', '=n', '<n', '>n',
    'long', 'l', '=l', '<l', '>l',
    'longlong', 'q', '=q', '<q', '>q',
]
AnyFloatDType: TypeAlias = _AnyDType[Float] | Literal[
    'float16', 'f2', '=f2', '<f2', '>f2',
    'float32', 'f4', '=f4', '<f4', '>f4',
    'float64', 'f8', '=f8', '<f8', '>f8',
    'half', 'e', '=e', '<e', '>e',
    'single', 'f', '=f', '<f', '>f',
    'double', 'float', 'float_', 'd', '=d', '<d', '>d',
    'longdouble', 'g', '=g', '<g', '>g',
]


# Various type aliases


Order: TypeAlias = Literal['C', 'F']
"""Type of the `order` parameter of e.g. [`np.empty`][numpy.empty]."""
OrderReshape: TypeAlias = Order | Literal['A']
"""Type of the `order` parameter of e.g. [`np.reshape`][numpy.array]."""
OrderCopy: TypeAlias = OrderReshape | Literal['K']
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

Casting: TypeAlias = Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']
"""See [`numpy.can_cast`][numpy.can_cast]."""
