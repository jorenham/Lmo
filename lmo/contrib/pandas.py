"""
Extension methods for `pandas.Series` and `pandas.DataFrame`.

Pandas is an optional dependency, and can be installed using
`pip install lmo[pandas]`.
"""
__all__ = (
    'Series',
    'DataFrame',
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
    Union,
    cast,
    final,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from lmo import (
    l_moment as _l_moment,
    l_ratio as _l_ratio,
    l_stats as _l_stats,
)
from lmo._utils import broadstack, clean_trim, moments_to_ratio
from lmo.typing import AnyInt, AnyTrim, IntVector, LMomentOptions

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

_FloatOrSeries: TypeAlias = Union[float, 'pd.Series[float]']
_SeriesOrFrame: TypeAlias = Union['pd.Series[float]', pd.DataFrame]
_FloatOrFrame: TypeAlias = _FloatOrSeries | pd.DataFrame

AxisDF: TypeAlias = Literal[0, 'index', 1, 'columns']


def _setindex(
    df: pd.DataFrame,
    axis: AxisDF,
    index: 'pd.Index[Any]',
) -> None:
    if axis == 0 or axis == 'index':
        df.index = index
    elif axis == 1 or axis == 'columns':
        df.columns = index
    else:
        msg = f"axis must be one of {{0, 'index', 1, 'columns'}}, got {axis}"
        raise TypeError(msg)


def _ratio_index(rk: npt.NDArray[np.int64]) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(rk, names=('r', 'k'))  # type: ignore


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
        r: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> _FloatOrSeries:
        """
        See [`lmo.l_moment`][lmo.l_moment].

        Returns:
            out: A scalar, or a [`pd.Series[float]`][pandas.Series], indexed
                by `r`.
        """
        out = _l_moment(self, r, trim=trim, **kwargs)
        if np.isscalar(out):
            return cast(float, out)

        return pd.Series(
            cast(npt.NDArray[np.float64], out),
            index=pd.Index(np.asarray(r), name='r'),
            dtype=float,
            copy=False,
        )

    def l_ratio(
        self,
        r: AnyInt | IntVector,
        k: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> _FloatOrSeries:
        """
        See [`lmo.l_ratio`][lmo.l_ratio].

        Returns:
            out: A scalar, or [`pd.Series[float]`][pandas.Series], with a
            [`MultiIndex`][pandas.MultiIndex] of `r` and `k`.
        """
        rk = broadstack(r, k)
        out = moments_to_ratio(rk, _l_moment(self, rk, trim=trim, **kwargs))
        if rk.ndim == 1:
            return cast(float, out)

        return pd.Series(
            cast(npt.NDArray[np.float64], out),
            index=_ratio_index(rk),
            dtype=float,
            copy=False,
        )

    def l_stats(
        self,
        trim: AnyTrim = (0, 0),
        num: int = 4,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_stats`][lmo.l_stats].

        Returns:
            A [`pd.Series[float]`][pandas.Series] with index `r = 1, ..., num`.
        """
        return pd.Series(
            _l_stats(self, trim=trim, num=num, **kwargs),
            index=pd.RangeIndex(1, num + 1, name='r'),
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
    def __lmo_register__(  # noqa: D105
        name: str,
        method: Callable[..., _FloatOrFrame],
    ) -> None:
        def fn(obj: pd.DataFrame) -> Callable[..., _FloatOrFrame]:
            return functools.partial(method, obj)  # type: ignore

        pd.api.extensions.register_dataframe_accessor(name)(fn)  # type: ignore

    def l_moment(
        self,
        r: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> _SeriesOrFrame:
        """
        See [`lmo.l_moment`][lmo.l_moment].

        Returns:
            out: A [`Series[float]`][pandas.Series], or
                a [`DataFrame`][pandas.DataFrame] with `r` as index along the
                specified axis.
        """
        out = cast(
            _SeriesOrFrame,
            self.apply(  # type: ignore
                _l_moment,
                axis=axis,
                result_type='expand',
                args=(r, trim),
                **kwargs,
            ),
        )
        if isinstance(out, pd.DataFrame):
            _setindex(out, axis, pd.Index(np.asarray(r), name='r'))
            out.attrs['l_kind'] = 'moment'
            out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_ratio(
        self,
        r: AnyInt | IntVector,
        k: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> _SeriesOrFrame:
        """
        See [`lmo.l_ratio`][lmo.l_ratio].

        Returns:
            out: A [`Series[float]`][pandas.Series], or a
                [`DataFrame`][pandas.DataFrame], with a
                [`MultiIndex`][pandas.MultiIndex] of `r` and `k` along the
                specified axis.
        """
        rk = broadstack(r, k)
        if rk.ndim > 2:
            rk = np.r_[rk[0].reshape(-1), rk[1].reshape(-1)]

        out = cast(
            _SeriesOrFrame,
            self.apply(  # type: ignore
                _l_ratio,
                axis=axis,
                result_type='expand',
                args=(rk[0], rk[1], trim),
                **kwargs,
            ),
        )
        if isinstance(out, pd.DataFrame):
            assert rk.ndim > 1
            _setindex(out, axis, _ratio_index(rk))
            out.attrs['l_kind'] = 'ratio'
            out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_stats(
        self,
        trim: AnyTrim = (0, 0),
        num: int = 4,
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> pd.DataFrame:
        """
        See [`lmo.l_stats`][lmo.l_stats].

        Returns:
            out: A [`DataFrame`][pandas.DataFrame] with `r = 1, ..., num` as
                index along the specified axis.
        """
        out = cast(
            pd.DataFrame,
            self.apply(  # type: ignore
                _l_stats,
                axis=axis,
                result_type='expand',
                args=(trim, num),
                **kwargs,
            ),
        )
        _setindex(out, axis, pd.RangeIndex(1, num + 1, name='r'))
        out.attrs['l_kind'] = 'stat'
        out.attrs['l_trim'] = clean_trim(trim)
        return out


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
