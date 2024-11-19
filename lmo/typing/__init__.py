"""Typing utilities meant for internal usage."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

# pyright: reportPrivateUsage=false
import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _NestedSequence  # noqa: PLC2701

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
    "AnyTrimFloat",
    "AnyTrimInt",
    "LComomentOptions",
    "LMomentOptions",
]


def __dir__() -> list[str]:
    return __all__


AnyTrimInt: TypeAlias = int | tuple[int, int]
AnyTrimFloat: TypeAlias = float | tuple[float, float]
AnyTrim: TypeAlias = AnyTrimInt | AnyTrimFloat


class _CanIntegerArray(Protocol):
    def __len__(self, /) -> int: ...  # this excludes scalar types
    def __array__(self, /) -> npt.NDArray[np.integer[Any]]: ...


AnyOrder: TypeAlias = int | np.integer[Any]
AnyOrderND: TypeAlias = _CanIntegerArray | _NestedSequence[int | np.integer[Any]]

AnyFWeights: TypeAlias = onp.Array[tuple[int], np.integer[Any]]
AnyAWeights: TypeAlias = onp.Array[onp.AtLeast1D, np.floating[Any]]


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
