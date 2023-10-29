"""
Extension methods for `pandas.Series` and `pandas.DataFrame`.

Pandas is an optional dependency, and can be installed using
`pip install lmo[pandas]`.
"""
__all__ = (
    'Series',
    'install',
)

import functools
import sys
from collections.abc import Callable
from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    final,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from lmo import l_moment as _l_moment
from lmo._utils import clean_trim
from lmo.typing import AnyInt, AnyTrim, IntVector, LMomentOptions

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

_FloatOrSeries: TypeAlias = Union[float, 'pd.Series[float]']
_SeriesOrFrame: TypeAlias = Union['pd.Series[float]', pd.DataFrame]
_FloatOrFrame: TypeAlias = _FloatOrSeries | pd.DataFrame

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

@final
class Series(pd.Series):  # type: ignore [missingTypeArguments]
    """
    Extension methods for [`pandas.Series`][pandas.Series].

    This class is not meant to be used directly. These methods are curried
    and registered as
    [series accessors][pandas.api.extensions.register_series_accessor].
    """
    @staticmethod
    def __lmo_register__(  # noqa: D105
        name: str,
        method: Callable[..., _FloatOrSeries | pd.DataFrame],
    ) -> None:
        def fn(obj: 'pd.Series[Any]') -> Callable[..., _FloatOrSeries]:
            return functools.partial(method, obj)  # type: ignore

        pd.api.extensions.register_series_accessor(name)(fn)  # type: ignore


    def l_moment(
        self,
        r: IntVector | AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> _FloatOrSeries:
        """
        Wrapper around [`lmo.l_moment`][lmo.l_moment].

        Returns:
            A scalar or [`pd.Series[float]`][pandas.Series] with `r` as index.
        """
        _trim = clean_trim(trim)
        res = _l_moment(self, r, trim=_trim, **kwargs)

        if np.isscalar(res):
            return cast(float, res)

        return pd.Series(
            cast(npt.NDArray[np.float64], res),
            index=pd.Index(np.asarray(r), name='r', dtype=int),
            dtype=float,
            copy=False,
        )


@final
class DataFrame(pd.DataFrame):
    """
    Extension methods for [`pandas.DataFrame`][pandas.DataFrame].

    This class is not meant to be used directly. These methods are curried
    and registered as
    [dataframe accessors][pandas.api.extensions.register_dataframe_accessor].
    """
    @staticmethod
    def __lmo_register__(
        name: str,
        method: Callable[..., _FloatOrFrame],
    ) -> None:
        def fn(obj: pd.DataFrame) -> Callable[..., _FloatOrFrame]:
            return functools.partial(method, obj)  # type: ignore

        pd.api.extensions.register_dataframe_accessor(name)(fn)  # type: ignore

    def l_moment(
        self,
        r: IntVector | AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        axis: Literal[0, 'index', 1, 'columns'] = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> _SeriesOrFrame:
        """
        Wrapper around [`lmo.l_moment`][lmo.l_moment].

        Returns:
            A [`pd.DataFrame`][pandas.DataFrame] or
                [`pd.Series[float]`][pandas.Series] with `r` as index.
        """
        # .aggregate only works correctly with axis=0 for some dumb reason
        transpose = axis == 1 or axis == 'columns'
        obj = self.T if transpose else self

        res = cast(
            _SeriesOrFrame,
            obj.aggregate(  # type: ignore
                _l_moment,
                0,
                r,
                trim=trim,
                **kwargs,
            ),
        )
        if isinstance(res, pd.DataFrame):
            res.index = pd.Index(np.asarray(r), name='r', dtype=int)

        return res.T if transpose else res


class _Registerable(Protocol):
    @staticmethod
    def __lmo_register__(name: str, method: Callable[..., Any]) -> None: ...


def _register_methods(cls: type[_Registerable]):
    for k, method in cls.__dict__.items():
        if not k.startswith('_') and callable(method):
            cls.__lmo_register__(k, method)


def install():
    """Register the accessor methods."""
    _register_methods(Series)
    _register_methods(DataFrame)
