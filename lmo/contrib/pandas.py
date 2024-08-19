# pyright: reportUnknownMemberType=false
"""
Extension methods for `pandas.Series` and `pandas.DataFrame`.

Pandas is an optional dependency, and can be installed using
`pip install lmo[pandas]`.

Examples:
    Univariate summary statistics:

    ```pycon
    >>> df = pd.DataFrame({'a': [1, 2, 2, 3, 4], 'b': [3, 4, 4, 4, 4]})
    >>> df.l_stats()
              a    b
    r
    1  2.400000  3.8
    2  0.700000  0.2
    3  0.142857 -1.0
    4  0.285714  1.0
    >>> df.aggregate(['mean', 'std', 'skew', 'kurt'])
                 a         b
    mean  2.400000  3.800000
    std   1.140175  0.447214
    skew  0.404796 -2.236068
    kurt -0.177515  5.000000

    ```

    Comparison of L-correlation, and Pearson correlation matrices:

    ```pycon
    >>> df = pd.DataFrame({'dogs': [.2, .0, .5, .4], 'cats': [.3, .2, .0, .1]})
    >>> df.l_corr()
          dogs      cats
    dogs   1.0 -0.764706
    cats  -0.8  1.000000
    >>> df.corr()
              dogs      cats
    dogs  1.000000 -0.756889
    cats -0.756889  1.000000

    ```
"""
from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    cast,
    final,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from lmo._lm import (
    l_moment as _l_moment,
    l_ratio as _l_ratio,
    l_stats as _l_stats,
)
from lmo._lm_co import (
    l_comoment as _l_comoment,
    l_coratio as _l_coratio,
)
from lmo._utils import clean_trim, moments_to_ratio


if sys.version_info >= (3, 13):
    from typing import TypeVar, Unpack
else:
    from typing_extensions import TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable

    import lmo.typing as lmt


__all__ = 'DataFrame', 'Series', 'install'


_T = TypeVar('_T')

_Axis: TypeAlias = Literal[0, 'index', 1, 'columns']


# `from __future__ import annotations` won't solve this;
# see https://github.com/pandas-dev/pandas-stubs/discussions/308
def __ensure_generic(tp: type[_T]):
    if hasattr(tp, '__class_getitem__'):
        return

    def __class_getitem__(cls: _T, _: Any, /) -> _T:  # noqa: N807
        return cls

    tp.__class_getitem__ = classmethod(__class_getitem__)  # pyright: ignore[reportArgumentType,reportAttributeAccessIssue]


__ensure_generic(pd.Series)
__ensure_generic(pd.Index)


def _setindex(
    df: pd.DataFrame,
    axis: _Axis,
    index: pd.Index[Any],
) -> None:
    if axis in {0, 'index'}:
        df.index = index
    elif axis in {1, 'columns'}:
        df.columns = index
    else:
        msg = f"axis must be one of {{0, 'index', 1, 'columns'}}, got {axis}"
        raise TypeError(msg)


def _ratio_index(rk: npt.NDArray[np.int64]) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(rk, names=('r', 'k'))


if TYPE_CHECKING:
    _BaseSeries = pd.Series[float]
else:
    _BaseSeries = pd.Series


@final
class Series(_BaseSeries):  # pyright: ignore[reportMissingTypeArgument]
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
        method: Callable[..., pd.Series[float]],
    ) -> None:
        def fn(obj: pd.Series[Any]) -> Callable[..., float | pd.Series[float]]:
            return method.__get__(obj, Series)

        pd.api.extensions.register_series_accessor(name)(fn)

    def l_moment(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float | pd.Series[float]:
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
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float | pd.Series[float]:
        """
        See [`lmo.l_ratio`][lmo.l_ratio].

        Returns:
            out: A scalar, or [`pd.Series[float]`][pandas.Series], with a
                [`MultiIndex`][pandas.MultiIndex] of `r` and `k`.
        """
        rk = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(k)))
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
        trim: lmt.AnyTrim = 0,
        num: int = 4,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
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
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_loc`][lmo.l_loc].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_moment(self, 1, trim, **kwargs))

    def l_scale(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_scale`][lmo.l_scale].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_moment(self, 2, trim, **kwargs))

    def l_variation(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_variation`][lmo.l_variation].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_ratio(self, 2, 1, trim, **kwargs))

    def l_skew(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> float:
        """
        See [`lmo.l_skew`][lmo.l_skew].

        Returns:
            out: A scalar.
        """
        return cast(float, _l_ratio(self, 3, 2, trim, **kwargs))

    def l_kurtosis(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
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
        method: Callable[..., float | pd.Series[float] | pd.DataFrame],
    ) -> None:
        def fn(
            obj: pd.DataFrame,
            /,
        ) -> Callable[..., float | pd.Series[float] | pd.DataFrame]:
            # return functools.partial(method, obj)
            return method.__get__(obj, cls)

        pd.api.extensions.register_dataframe_accessor(name)(fn)

    def l_moment(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float] | pd.DataFrame:
        """
        See [`lmo.l_moment`][lmo.l_moment].

        Returns:
            out:
                A [`Series[float]`][pandas.Series], or
                a [`DataFrame`][pandas.DataFrame] with `r` as index along the
                specified axis.
        """
        out = self.apply(
            _l_moment,
            axis=axis,
            result_type='expand',
            args=(r, trim),
            **kwargs,
        )
        if isinstance(out, pd.DataFrame):
            _setindex(out, axis, pd.Index(np.asarray(r), name='r'))
            out.attrs['l_kind'] = 'moment'
            out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_ratio(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float] | pd.DataFrame:
        """
        See [`lmo.l_ratio`][lmo.l_ratio].

        Returns:
            out: A [`Series[float]`][pandas.Series], or a
                [`DataFrame`][pandas.DataFrame], with a
                [`MultiIndex`][pandas.MultiIndex] of `r` and `k` along the
                specified axis.
        """
        rk = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(k)))
        if rk.ndim > 2:
            rk = np.r_[rk[0].reshape(-1), rk[1].reshape(-1)]

        out = self.apply(
            _l_ratio,
            axis=axis,
            result_type='expand',
            args=(rk[0], rk[1], trim),
            **kwargs,
        )
        if isinstance(out, pd.DataFrame):
            assert rk.ndim > 1
            _setindex(out, axis, _ratio_index(rk))
            out.attrs['l_kind'] = 'ratio'
            out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_stats(
        self,
        trim: lmt.AnyTrim = 0,
        num: int = 4,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.DataFrame:
        """
        See [`lmo.l_stats`][lmo.l_stats].

        Returns:
            out: A [`DataFrame`][pandas.DataFrame] with `r = 1, ..., num` as
                index along the specified axis.
        """
        out = self.apply(
            _l_stats,
            axis=axis,
            result_type='expand',
            args=(trim, num),
            **kwargs,
        )
        _setindex(out, axis, pd.RangeIndex(1, num + 1, name='r'))
        out.attrs['l_kind'] = 'stat'
        out.attrs['l_trim'] = clean_trim(trim)
        return out

    def l_loc(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_moment(1, ...)`][lmo.contrib.pandas.DataFrame.l_moment].
        See [`lmo.l_loc`][lmo.l_loc] for details.
        """
        return self.apply(  # pyright: ignore[reportReturnType]
            _l_moment,
            axis=axis,
            args=(1, trim),
            **kwargs,
        )

    def l_scale(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_moment(2, ...)`][lmo.contrib.pandas.DataFrame.l_moment].
        See [`lmo.l_scale`][lmo.l_scale] for details.
        """
        return self.apply(  # pyright: ignore[reportReturnType]
            _l_moment,
            axis=axis,
            args=(2, trim),
            **kwargs,
        )

    def l_variation(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_ratio(2, 1, ...)`][lmo.contrib.pandas.DataFrame.l_ratio].
        See [`lmo.l_variation`][lmo.l_variation] for details.
        """
        return self.apply(  # pyright: ignore[reportReturnType]
            _l_ratio,
            axis=axis,
            args=(2, 1, trim),
            **kwargs,
        )

    def l_skew(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_ratio(3, 2, ...)`][lmo.contrib.pandas.DataFrame.l_ratio].
        See [`lmo.l_skew`][lmo.l_skew] for details.
        """
        return self.apply(  # pyright: ignore[reportReturnType]
            _l_ratio,
            axis=axis,
            args=(3, 2, trim),
            **kwargs,
        )

    def l_kurtosis(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_ratio(4, 2, ...)`][lmo.contrib.pandas.DataFrame.l_ratio].
        See [`lmo.l_kurtosis`][lmo.l_kurtosis] for details.
        """
        return self.apply(  # pyright: ignore[reportReturnType]
            _l_ratio,
            axis=axis,
            args=(4, 2, trim),
            **kwargs,
        )

    def l_kurt(
        self,
        trim: lmt.AnyTrim = 0,
        axis: _Axis = 0,
        **kwargs: Unpack[lmt.LMomentOptions],
    ) -> pd.Series[float]:
        """
        Alias for
        [`l_kurtosis`][lmo.contrib.pandas.DataFrame.l_kurtosis].
        """
        return self.l_kurtosis(trim=trim, axis=axis, **kwargs)

    def l_comoment(
        self,
        r: lmt.AnyOrder,
        /,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
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

        kwargs |= {'rowvar': False}
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
        r: lmt.AnyOrder,
        k: lmt.AnyOrder = 2,
        /,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
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

        kwargs |= {'rowvar': False}
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

    def l_coloc(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for [`l_comoment(1, trim, **kwargs)
        `][lmo.contrib.pandas.DataFrame.l_comoment].
        See [`lmo.l_coloc`][lmo.l_coloc] for details.
        """
        return self.l_comoment(1, trim=trim, **kwargs)

    def l_coscale(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for [`l_comoment(2, trim, **kwargs)
        `][lmo.contrib.pandas.DataFrame.l_comoment].
        See [`lmo.l_coscale`][lmo.l_coscale] for details.
        """
        return self.l_comoment(2, trim=trim, **kwargs)

    def l_corr(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for [`l_coratio(2, 2, trim, **kwargs)
        `][lmo.contrib.pandas.DataFrame.l_coratio].
        See [`lmo.l_corr`][lmo.l_corr] for details.
        """
        return self.l_coratio(2, trim=trim, **kwargs)

    def l_coskew(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for [`l_coratio(3, 2, trim, **kwargs)
        `][lmo.contrib.pandas.DataFrame.l_coratio].
        See [`lmo.l_coskew`][lmo.l_coskew] for details.
        """
        return self.l_coratio(3, trim=trim, **kwargs)

    def l_cokurtosis(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for [`l_coratio(4, 2, trim, **kwargs)
        `][lmo.contrib.pandas.DataFrame.l_coratio].
        See [`lmo.l_cokurtosis`][lmo.l_cokurtosis] for details.
        """
        return self.l_coratio(4, trim=trim, **kwargs)

    def l_cokurt(
        self,
        trim: lmt.AnyTrim = 0,
        **kwargs: Unpack[lmt.LComomentOptions],
    ) -> pd.DataFrame:
        """
        Alias for
        [`l_cokurtosis`][lmo.contrib.pandas.DataFrame.l_cokurtosis].
        """
        return self.l_cokurtosis(trim=trim, **kwargs)


class _Registerable(Protocol):
    @staticmethod
    def __lmo_register__(
        name: str,
        method: Callable[..., Any],
    ) -> None: ...


def _register_methods(cls: type[_Registerable]):
    for k, method in cls.__dict__.items():
        if not k.startswith('_') and callable(method):
            cls.__lmo_register__(k, method)


def install() -> None:
    """Register the accessor methods."""
    _register_methods(Series)
    _register_methods(DataFrame)
