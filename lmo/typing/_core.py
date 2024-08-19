from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

import numpy as np
import optype as opt
import optype.numpy as onpt


if TYPE_CHECKING:
    import lmo.typing.np as lnpt


_AnyTrimI: TypeAlias = int | tuple[int, int]
_AnyTrimF: TypeAlias = float | tuple[float, float]
AnyTrim: TypeAlias = _AnyTrimI | _AnyTrimF

AnyOrder: TypeAlias = int | np.integer[Any]
AnyOrderND: TypeAlias = opt.CanSequence[int, int, int] | onpt.AnyIntegerArray

AnyFWeights: TypeAlias = onpt.Array[tuple[int], np.integer[Any]]
AnyAWeights: TypeAlias = onpt.Array[onpt.AtLeast1D, np.floating[Any]]


class LMomentOptions(TypedDict, total=False):
    """
    Use as e.g. `**kwds: Unpack[LMomentOptions]` (on `python<3.11`) or
    `**kwds: *LMomentOptions` (on `python>=3.11`).
    """
    sort: lnpt.SortKind | bool
    cache: bool | None
    fweights: AnyFWeights
    aweights: AnyAWeights


class LComomentOptions(TypedDict, total=False):
    """
    Use as e.g. `**kwds: Unpack[LComomentOptions]` (on `python<3.11`) or
    `**kwds: *LComomentOptions` (on `python>=3.11`).
    """
    sort: lnpt.SortKind
    cache: bool | None
    rowvar: bool | None
