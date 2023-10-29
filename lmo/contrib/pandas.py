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
    l_comoment as _l_comoment,
    l_coratio as _l_coratio,
    l_moment as _l_moment,
    l_ratio as _l_ratio,
    l_stats as _l_stats,
)
from lmo._utils import broadstack, clean_trim, moments_to_ratio
from lmo.typing import (
    AnyInt,
    AnyTrim,
    IntVector,
    LComomentOptions,
    LMomentOptions,
)

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
    @classmethod
    def __lmo_register__(  # noqa: D105
        cls,
        name: str,
        method: Callable[..., _FloatOrSeries | pd.DataFrame],
    ) -> None:
        def fn(obj: 'pd.Series[Any]') -> Callable[..., _FloatOrSeries]:
            return method.__get__(obj, Series)

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
            out: A [`pd.Series[float]`][pandas.Series] with index
                `r = 1, ..., num`.
        """
        return pd.Series(
            _l_stats(self, trim=trim, num=num, **kwargs),
            index=pd.RangeIndex(1, num + 1, name='r'),
            copy=False,
        )

    def l_loc(
        self,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_loc`][lmo.l_loc].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_moment(self, 1, trim, **kwargs))

    def l_scale(
        self,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_scale`][lmo.l_scale].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_moment(self, 2, trim, **kwargs))

    def l_variation(
        self,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_variation`][lmo.l_variation].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_ratio(self, 2, 1, trim, **kwargs))

    def l_skew(
        self,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_skew`][lmo.l_skew].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_ratio(self, 3, 2, trim, **kwargs))

    def l_kurtosis(
        self,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_kurtosis`][lmo.l_kurtosis].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_ratio(self, 4, 2, trim, **kwargs))

    l_kurt = l_kurtosis


@final
class DataFrame(pd.DataFrame):
    """
    Extension methods for [`pandas.DataFrame`][pandas.DataFrame].

    This class is not meant to be used directly. These methods are curried
    and registered as
    [dataframe accessors][pandas.api.extensions.register_dataframe_accessor].
    """
    @classmethod
    def __lmo_register__(  # noqa: D105
        cls,
        name: str,
        method: Callable[..., _FloatOrFrame],
    ) -> None:
        def fn(obj: pd.DataFrame) -> Callable[..., _FloatOrFrame]:
            # return functools.partial(method, obj)  # type: ignore
            return method.__get__(obj, cls)

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

    def l_loc(
        self,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_loc`][lmo.l_loc].

        Returns:
            out: A [`Series[float]`][pandas.Series].
        """
        return self.apply(  # type: ignore
            _l_moment,
            axis=axis,
            args=(1, trim),
            **kwargs,
        )

    def l_scale(
        self,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_scale`][lmo.l_scale].

        Returns:
            out: A [`Series[float]`][pandas.Series].
        """
        return self.apply(  # type: ignore
            _l_moment,
            axis=axis,
            args=(2, trim),
            **kwargs,
        )

    def l_variation(
        self,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_variation`][lmo.l_variation].

        Returns:
            out: A [`Series[float]`][pandas.Series].
        """
        return self.apply(  # type: ignore
            _l_ratio,
            axis=axis,
            args=(2, 1, trim),
            **kwargs,
        )

    def l_skew(
        self,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_skew`][lmo.l_skew].

        Returns:
            out: A [`Series[float]`][pandas.Series].
        """
        return self.apply(  # type: ignore
            _l_ratio,
            axis=axis,
            args=(3, 2, trim),
            **kwargs,
        )

    def l_kurtosis(
        self,
        trim: AnyTrim = (0, 0),
        axis: AxisDF = 0,
        **kwargs: Unpack[LMomentOptions],
    ) -> 'pd.Series[float]':
        """
        See [`lmo.l_kurtosis`][lmo.l_kurtosis].

        The alias `l_kurt` is also available.

        Returns:
            out: A [`Series[float]`][pandas.Series].
        """
        return self.apply(  # type: ignore
            _l_ratio,
            axis=axis,
            args=(4, 2, trim),
            **kwargs,
        )

    l_kurt = l_kurtosis

    def l_comoment(
        self,
        r: AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LComomentOptions],
    ) -> pd.DataFrame:
        """
        See [`lmo.l_comoment`][lmo.l_comoment].

        Args:
            r: The L-moment order, as a non-negative scalar.
            trim: Left- and right-trim orders.
            **kwargs: Additional options to pass to
                [`lmo.l_comoment`][lmo.l_comoment].

        Returns:
            out: A [`DataFrame`][pandas.DataFrame] of the column-to-column
                L-comoment matrix.

        Raises:
            TypeError: If `rowvar=True`, use `df.T.l_comoment` instead.
        """
        if kwargs.pop('rowvar', False):
            msg = 'rowvar=True is not supported; use df.T instead'
            raise TypeError(msg)

        kwargs = kwargs | {'rowvar': False}
        out = pd.DataFrame(
            _l_comoment(self, _r := int(r), trim=trim, **kwargs),
            index=self.columns,
            columns=self.columns,
            copy=False,
        )
        out.attrs['l_kind'] = 'comoment'
        out.attrs['l_r'] = _r
        out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_coratio(
        self,
        r: AnyInt,
        k: AnyInt = 2,
        /,
        trim: AnyTrim = (0, 0),
        **kwargs: Unpack[LComomentOptions],
    ) -> pd.DataFrame:
        """
        See [`lmo.l_coratio`][lmo.l_coratio].

        Args:
            r: The L-moment order of the numerator, a non-negative scalar.
            k: The L-moment order of the denominator, a non-negative scalar.
                Defaults to 2. If set to 0, this is equivalent to `l_comoment`.
            trim: Left- and right-trim orders.
            **kwargs: Additional options to pass to
                [`lmo.l_comoment`][lmo.l_comoment].

        Returns:
            out: A [`DataFrame`][pandas.DataFrame] of the column-to-column
                matrix of L-comoment ratio's.

        Raises:
            TypeError: If `rowvar=True`, use `df.T.l_comoment` instead.
        """
        if kwargs.pop('rowvar', False):
            msg = 'rowvar=True is not supported; use df.T instead'
            raise TypeError(msg)

        kwargs = kwargs | {'rowvar': False}
        out = pd.DataFrame(
            _l_coratio(self, _r := int(r), _k := int(k), trim=trim, **kwargs),
            index=self.columns,
            columns=self.columns,
            copy=False,
        )
        out.attrs['l_kind'] = 'coratio'
        out.attrs['l_r'] = _r
        out.attrs['l_k'] = _k
        out.attrs['l_trim'] = clean_trim(trim)
        return out

    l_coloc = functools.partialmethod(l_comoment, 1)
    l_coscale = functools.partialmethod(l_comoment, 2)
    l_corr = functools.partialmethod(l_coratio, 2, 2)
    l_coskew = functools.partialmethod(l_coratio, 3, 2)
    l_cokurtosis = l_cokurt = functools.partialmethod(l_coratio, 4, 2)


class _Registerable(Protocol):
    @staticmethod
    def __lmo_register__(name: str, method: Callable[..., Any]) -> None: ...


def _register_methods(cls: type[_Registerable]):
    for k, method in cls.__dict__.items():
        if not k.startswith('_') and (
            callable(method)
            or isinstance(method, functools.partialmethod)
        ):
            cls.__lmo_register__(k, method)


def install():
    """Register the accessor methods."""
    _register_methods(Series)
    _register_methods(DataFrame)
