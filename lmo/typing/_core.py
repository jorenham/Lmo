from typing import Any, TypeAlias, TypedDict

import numpy as np

from .np import AnyArrayFloat, AnyArrayInt, AnyVectorInt, SortKind


_AnyTrimOrder = float | np.integer[Any] | np.floating[Any]
AnyTrim: TypeAlias = _AnyTrimOrder | tuple[_AnyTrimOrder, _AnyTrimOrder]

AnyOrder: TypeAlias = int | np.integer[Any]
AnyOrderND: TypeAlias = AnyArrayInt

AnyFWeights: TypeAlias = AnyVectorInt
AnyAWeights: TypeAlias = AnyArrayFloat


class LMomentOptions(TypedDict, total=False):
    """Use as e.g. `def spam(**kwargs: Unpack[LMomentOptions]): ...`."""
    sort: SortKind
    cache: bool
    fweights: AnyFWeights
    aweights: AnyAWeights


class LComomentOptions(TypedDict, total=False):
    """Use as e.g. `def spam(**kwargs: Unpack[LComomentOptions]): ...`."""
    sort: SortKind
    cache: bool
    rowvar: bool
