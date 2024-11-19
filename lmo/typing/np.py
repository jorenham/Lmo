"""NumPy-related type aliases for internal use."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp

__all__ = (
    "AnyArrayFloat",
    "AnyArrayInt",
    "AnyMatrixFloat",
    "AnyMatrixInt",
    "AnyScalarFloat",
    "AnyScalarInt",
    "AnyTensorFloat",
    "AnyTensorInt",
    "AnyVector",
    "AnyVectorFloat",
    "AnyVectorFloat",
    "Float",
    "Int",
    "Integral",
    "OrderReshape",
    "Real",
    "SortKind",
)


def __dir__() -> tuple[str, ...]:
    return __all__


# Some handy scalar type aliases


Int: TypeAlias = np.integer[Any]
Float: TypeAlias = np.floating[Any]
Number: TypeAlias = np.number[Any]

Integral: TypeAlias = Int | np.bool_
Real: TypeAlias = Float | Integral


# Array and array-likes, with generic shape

_ST = TypeVar("_ST", bound=np.generic)


_PyScalar: TypeAlias = complex | str | bytes
_ST_py = TypeVar("_ST_py", bound=_PyScalar)

_T = TypeVar("_T")
_PyVector: TypeAlias = Sequence[_T]


_AnyScalar: TypeAlias = _ST | _ST_py | onp.CanArray[tuple[()], np.dtype[_ST]]
_AnyVector: TypeAlias = (
    onp.CanArray[tuple[int], np.dtype[_ST]]
    | _PyVector[_AnyScalar[_ST, _ST_py]]
)  # fmt: skip
_AnyMatrix: TypeAlias = (
    onp.CanArray[tuple[int, int], np.dtype[_ST]]
    | _PyVector[_AnyVector[_ST, _ST_py]]
)  # fmt: skip

# these will result in {0,1,2,N}-D arrays when passed to `np.array` (no need
# for a broken "nested sequence" type)

AnyVector: TypeAlias = _AnyVector[np.generic, _PyScalar]

AnyScalarInt: TypeAlias = _AnyScalar[Integral, int]
AnyVectorInt: TypeAlias = _AnyVector[Integral, int]
AnyMatrixInt: TypeAlias = _AnyMatrix[Integral, int]
AnyTensorInt: TypeAlias = (
    onp.CanArray[onp.AtLeast3D, np.dtype[Integral]]
    | _PyVector[AnyMatrixInt]
    | _PyVector["AnyTensorInt"]
)
AnyArrayInt: TypeAlias = AnyVectorInt | AnyMatrixInt | AnyTensorInt

AnyScalarFloat: TypeAlias = _AnyScalar[Real, float]
AnyVectorFloat: TypeAlias = _AnyVector[Real, float]
AnyMatrixFloat: TypeAlias = _AnyMatrix[Real, float]
AnyTensorFloat: TypeAlias = (
    onp.CanArray[onp.AtLeast1D, np.dtype[Real]]
    | _PyVector[AnyMatrixFloat]
    | _PyVector["AnyTensorFloat"]
)
AnyArrayFloat: TypeAlias = AnyVectorFloat | AnyMatrixFloat | AnyTensorFloat


# Various type aliases


OrderReshape: TypeAlias = Literal["C", "F", "A"]
"""Type of the `order` parameter of e.g. [`np.reshape`][numpy.array]."""

SortKind: TypeAlias = Literal[
    "quick", "quicksort",
    "stable", "stablesort",
    "heap", "heapsort",
]  # fmt: skip
"""
Type of the `kind` parameter of e.g. [`np.sort`][numpy.sort], as
allowed by numpy's own stubs.
Note that the actual implementation just looks at `kind[0].lower() == 'q'`.
This means that it's possible to select stable-sort by passing
`kind='SnailSort'` instead of `kind='stable'` (although your typechecker might
ruin the fun).
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
