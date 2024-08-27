"""NumPy-related type aliases for internal use."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onpt


__all__ = (
    'Bool',
    'Int',
    'Float',
    'Number',
    'Natural',
    'Integer',
    'Real',
    'AnyScalar',
    'AnyScalarBool',
    'AnyScalarInt',
    'AnyScalarFloat',
    'AnyVector',
    'AnyVectorBool',
    'AnyVectorFloat',
    'AnyVectorFloat',
    'AnyMatrix',
    'AnyMatrixBool',
    'AnyMatrixInt',
    'AnyMatrixFloat',
    'AnyTensor',
    'AnyTensorBool',
    'AnyTensorInt',
    'AnyTensorFloat',
    'AnyArray',
    'AnyArrayBool',
    'AnyArrayInt',
    'AnyArrayFloat',
    'SortKind',
    'Order',
    'OrderReshape',
    'OrderCopy',
    'RandomState',
    'Casting',
)


# Some handy scalar type aliases


Bool: TypeAlias = np.bool_
UInt: TypeAlias = np.unsignedinteger[Any]
Int: TypeAlias = np.integer[Any]
Float: TypeAlias = np.floating[Any]
Number: TypeAlias = np.number[Any]

Natural: TypeAlias = UInt | Bool
Integer: TypeAlias = np.integer[Any] | np.bool_
Real: TypeAlias = Float | Integer


# Array and array-likes, with generic shape

_ST = TypeVar('_ST', bound=np.generic)


_PyScalar: TypeAlias = bool | int | float | complex | str | bytes
# _PyScalar: TypeAlias = bool | int | float | complex
_ST_py = TypeVar('_ST_py', bound=_PyScalar)

_T = TypeVar('_T')
_PyVector: TypeAlias = Sequence[_T]


_AnyScalar: TypeAlias = _ST | _ST_py | onpt.CanArray[tuple[()], np.dtype[_ST]]
_AnyVector: TypeAlias = (
    onpt.CanArray[tuple[int], np.dtype[_ST]]
    | _PyVector[_AnyScalar[_ST, _ST_py]]
)
_AnyMatrix: TypeAlias = (
    onpt.CanArray[tuple[int, int], np.dtype[_ST]]
    | _PyVector[_AnyVector[_ST, _ST_py]]
)

# these will result in {0,1,2,N}-D arrays when passed to `np.array` (no need
# for a broken "nested sequence" type)

AnyScalar: TypeAlias = _AnyScalar[np.generic, _PyScalar]
AnyVector: TypeAlias = _AnyVector[np.generic, _PyScalar]
AnyMatrix: TypeAlias = _AnyMatrix[np.generic, _PyScalar]
AnyTensor: TypeAlias = (
    onpt.CanArray[onpt.AtLeast3D, np.dtype[np.generic]]
    | _PyVector[AnyMatrix]
    | _PyVector['AnyTensor']
)
AnyArray: TypeAlias = AnyScalar | AnyVector | AnyMatrix | AnyTensor

AnyScalarBool: TypeAlias = _AnyScalar[Bool, bool]
AnyVectorBool: TypeAlias = _AnyVector[Bool, bool]
AnyMatrixBool: TypeAlias = _AnyMatrix[Bool, bool]
AnyTensorBool: TypeAlias = (
    onpt.CanArray[onpt.AtLeast3D, np.dtype[Bool]]
    | _PyVector[AnyMatrixBool]
    | _PyVector['AnyTensorBool']
)
AnyArrayBool: TypeAlias = AnyVectorBool | AnyMatrixBool | AnyTensorBool

AnyScalarInt: TypeAlias = _AnyScalar[Integer, int]
AnyVectorInt: TypeAlias = _AnyVector[Integer, int]
AnyMatrixInt: TypeAlias = _AnyMatrix[Integer, int]
AnyTensorInt: TypeAlias = (
    onpt.CanArray[onpt.AtLeast3D, np.dtype[Integer]]
    | _PyVector[AnyMatrixInt]
    | _PyVector['AnyTensorInt']
)
AnyArrayInt: TypeAlias = AnyVectorInt | AnyMatrixInt | AnyTensorInt

AnyScalarFloat: TypeAlias = _AnyScalar[Real, float]
AnyVectorFloat: TypeAlias = _AnyVector[Real, float]
AnyMatrixFloat: TypeAlias = _AnyMatrix[Real, float]
AnyTensorFloat: TypeAlias = (
    onpt.CanArray[onpt.AtLeast1D, np.dtype[Real]]
    | _PyVector[AnyMatrixFloat]
    | _PyVector['AnyTensorFloat']
)
AnyArrayFloat: TypeAlias = AnyVectorFloat | AnyMatrixFloat | AnyTensorFloat


# Various type aliases


Order: TypeAlias = Literal['C', 'F']
"""Type of the `order` parameter of e.g. [`np.empty`][numpy.empty]."""
OrderReshape: TypeAlias = Literal[Order, 'A']
"""Type of the `order` parameter of e.g. [`np.reshape`][numpy.array]."""
OrderCopy: TypeAlias = Literal[OrderReshape, 'K']
"""Type of the `order` parameter of e.g. [`np.array`][numpy.array]."""

SortKind: TypeAlias = Literal[
    'quick',
    'quicksort',
    'stable',
    'stablesort',
    'heap',
    'heapsort',
]
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
)  # fmt: skip
"""
Any acceptable "seed" type that can be passed to
[`numpy.random.default_rng`][numpy.random.default_rng].
"""

Casting: TypeAlias = Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']
"""See [`numpy.can_cast`][numpy.can_cast]."""
