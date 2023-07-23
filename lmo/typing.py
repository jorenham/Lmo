"""Numpy-related type aliasses for internal use."""

__all__ = (
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

    'LMomentOptions',
    'LComomentOptions',
)

from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

T = TypeVar('T', bound=np.generic)
T_co = TypeVar('T_co', covariant=True, bound=np.generic)

@runtime_checkable
class _SupportsArray(Protocol[T_co]):
    def __array__(self) -> npt.NDArray[T_co]: ...


# scalar types
_NumpyBool: TypeAlias = np.bool_
_NumpyInteger: TypeAlias = np.integer[Any] | _NumpyBool
_NumpyFloating: TypeAlias = np.floating[Any] | _NumpyInteger

AnyBool: TypeAlias = _NumpyBool | bool
AnyInt: TypeAlias = _NumpyInteger | int | bool
AnyFloat: TypeAlias = _NumpyFloating | float | int | bool

# array-like flavours (still waiting on numpy's shape typing)
AnyNDArray: TypeAlias = npt.NDArray[T] | _SupportsArray[T]

IntVector: TypeAlias = AnyNDArray[_NumpyInteger] | Sequence[AnyInt]
IntMatrix: TypeAlias = AnyNDArray[_NumpyInteger] | Sequence[IntVector]
IntTensor: TypeAlias = AnyNDArray[_NumpyInteger] | Sequence['IntTensor']

FloatVector: TypeAlias = AnyNDArray[_NumpyFloating] | Sequence[AnyFloat]
FloatMatrix: TypeAlias = AnyNDArray[_NumpyFloating] | Sequence[FloatVector]
FloatTensor: TypeAlias = AnyNDArray[_NumpyFloating] | Sequence['FloatTensor']

# for numpy.sort
SortKind: TypeAlias = Literal['quicksort', 'heapsort', 'stable']
IndexOrder: TypeAlias = Literal['C', 'F', 'A', 'K']


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
