from typing import Any, TypeAlias, TypedDict

import numpy as np
import optype as opt

from . import np as lnpt


_AnyTrimI: TypeAlias = int | tuple[int, int]
_AnyTrimF: TypeAlias = float | tuple[float, float]
AnyTrim: TypeAlias = _AnyTrimI | _AnyTrimF

AnyOrder: TypeAlias = int | np.integer[Any]
AnyOrderND: TypeAlias = opt.CanSequence[int, int] | lnpt.AnyArrayInt

AnyFWeights: TypeAlias = lnpt.Array[tuple[int], np.integer[Any]]
AnyAWeights: TypeAlias = lnpt.Array[lnpt.AtLeast1D, np.floating[Any]]


class LMomentOptions(TypedDict, total=False):
    """
    Use as e.g. `**kwds: Unpack[LMomentOptions]` (on `python<3.11`) or
    `**kwds: *LMomentOptions` (on `python>=3.11`).
    """
    sort: lnpt.SortKind | bool
    cache: bool
    fweights: AnyFWeights
    aweights: AnyAWeights


class LComomentOptions(TypedDict, total=False):
    """
    Use as e.g. `**kwds: Unpack[LComomentOptions]` (on `python<3.11`) or
    `**kwds: *LComomentOptions` (on `python>=3.11`).
    """
    sort: lnpt.SortKind
    cache: bool
    rowvar: bool | None
