from typing import TypeAlias, TypedDict

import numpy.typing as npt

from .np import SortKind


_TrimOrder = int | float
AnyTrim: TypeAlias = _TrimOrder | tuple[_TrimOrder, _TrimOrder]


class LMomentOptions(TypedDict, total=False):
    """Use as e.g. `def spam(**kwargs: Unpack[LMomentOptions]): ...`."""
    sort: SortKind
    cache: bool
    fweights: npt.ArrayLike
    aweights: npt.ArrayLike


class LComomentOptions(TypedDict, total=False):
    """Use as e.g. `def spam(**kwargs: Unpack[LComomentOptions]): ...`."""
    sort: SortKind
    cache: bool
    rowvar: bool
