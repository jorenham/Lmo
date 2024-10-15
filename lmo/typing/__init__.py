"""Typing utilities meant for internal usage."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import optype.numpy as onpt
from optype import CanSequence

if sys.version_info >= (3, 13):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    import lmo.typing.np as lnpt


__all__ = [
    "AnyAWeights",
    "AnyFWeights",
    "AnyOrder",
    "AnyOrderND",
    "AnyTrim",
    "AnyTrimInt",
    "LComomentOptions",
    "LMomentOptions",
]


AnyTrimInt: TypeAlias = int | tuple[int, int]
AnyTrimFloat: TypeAlias = float | tuple[float, float]
AnyTrim: TypeAlias = AnyTrimInt | AnyTrimFloat

AnyOrder: TypeAlias = int | np.integer[Any]
AnyOrderND: TypeAlias = (
    CanSequence[int, int, int]
    | onpt.CanArray[tuple[int, ...], np.dtype[np.integer[Any]]]
)

AnyFWeights: TypeAlias = onpt.Array[tuple[int], np.integer[Any]]
AnyAWeights: TypeAlias = onpt.Array[onpt.AtLeast1D, np.floating[Any]]


class LMomentOptions(TypedDict, total=False):
    """Use as e.g. `**kwds: Unpack[LMomentOptions]`."""

    sort: lnpt.SortKind | bool
    cache: bool | None
    fweights: AnyFWeights
    aweights: AnyAWeights


class LComomentOptions(TypedDict, total=False):
    """Use as e.g. `**kwds: Unpack[LComomentOptions]`."""

    sort: lnpt.SortKind
    cache: bool | None
    rowvar: bool | None
