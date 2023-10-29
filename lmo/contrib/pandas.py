"""Extension methods for `pandas.Series` and `pandas.DataFrame`."""
from __future__ import annotations

__all__ = (
    'l_moment',
    'install',
)

import sys
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd

from lmo import l_moment as _l_moment

if TYPE_CHECKING:
    from lmo.typing import AnyInt, AnyTrim, IntVector, LMomentOptions

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

T = TypeVar(
    'T',
    bound=(
        bool
        | int
        | float
        | np.dtype[np.bool_]
        | np.dtype[np.integer[Any]]
        | np.dtype[np.floating[Any]]
    ),
)

class l_moment(Generic[T]):  # noqa: N801
    """Extension method for `pandas.Series`."""
    __slots__ = ('_obj',)

    _obj: pd.Series[T]

    def __init__(self, obj: pd.Series[T]) -> None:  # noqa: D107
        self._validate(obj)
        self._obj = obj

    @classmethod
    def _validate(cls, obj: pd.Series[T]) -> None:
        if not pd.api.types.is_any_real_numeric_dtype(obj):  # type: ignore
            msg = f'Can only use .{cls.__name__} accessor with numeric values'
            raise AttributeError(msg)

    def __call__(  # noqa: D102
        self,
        r: IntVector | AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> pd.Series[float] | float:
        _r = np.asarray(r).ravel()

        res = _l_moment(
            np.asarray(self._obj, float),
            _r,
            trim=trim,
            **kwargs,
        )

        if len(res) == 1:
            return res[0]

        return pd.Series(
            res,
            index=_r,
            name='l_moment',
            dtype=float,
        )


def install():
    """Register the accessor methods."""
    for method in [  # type: ignore
        l_moment,
    ]:
        pd.api.extensions.register_series_accessor(  # type: ignore
            method.__name__,
        )(method)
