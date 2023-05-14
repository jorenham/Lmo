__all__ = (
    'AnyBool',
    'AnyInt',
    'AnyFloat',
    'AnyVector',
    'AnyMatrix',
    'AnyTensor',
    'ScalarOrArray',
    'SortKind',
)

from typing import (
    Any,
    Literal,
    Sequence,
    SupportsFloat,
    TypeAlias,
    TypeVar
)

import numpy as np

AnyBool: TypeAlias = bool | np.bool_
AnyInt: TypeAlias = int | np.integer[Any] | AnyBool
AnyFloat: TypeAlias = float | np.floating[Any] | AnyInt

_R = TypeVar('_R', bound=np.floating[Any] | np.integer[Any] | np.bool_)
_AnyR = TypeVar('_AnyR', bound=SupportsFloat)

AnyVector: TypeAlias = np.ndarray[Any, np.dtype[Any]] | Sequence[SupportsFloat]
AnyMatrix: TypeAlias = AnyVector | Sequence[AnyVector]
AnyTensor: TypeAlias = AnyMatrix | Sequence['AnyTensor']

# complex numbers aren't relevant (and calling them scalars is far-fetched IMHO)
ScalarOrArray: TypeAlias = _R | np.ndarray[Any, np.dtype[_R]]

SortKind: TypeAlias = Literal['quicksort', 'heapsort', 'stable']
