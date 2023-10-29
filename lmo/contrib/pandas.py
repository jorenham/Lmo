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
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar, Union, cast

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
Ps = ParamSpec('Ps')
# R1 = TypeVar('R1', bound=pd.Series[float] | float)
R1 = TypeVar('R1', bound=_FloatOrSeries)
R2 = TypeVar('R2', bound=_FloatOrSeries | pd.DataFrame)


_XSR_SERIES: dict[
    str,
    Callable[['pd.Series[Any]'], Callable[..., _FloatOrSeries]],
] = {}


def _xsr_series(
    fn: Callable[Concatenate['pd.Series[Any]', Ps], R1],
    /,
) -> Callable[Concatenate['pd.Series[Any]', Ps], R1]:
    def _xsr(obj: 'pd.Series[Any]') -> Callable[Ps, R1]:
        return functools.partial(fn, obj)  # type: ignore

    _XSR_SERIES[fn.__name__] = _xsr

    return fn


class Series(pd.Series):  # type: ignore [missingTypeArguments]
    """Extension methods for [`pandas.Series`][pandas.Series]."""
    @_xsr_series
    def l_moment(
        self: 'pd.Series[Any]',
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

        path = f'{self.name}.' if self.name else ''
        opts = f'({trim=})' if all(_trim) else ''
        return pd.Series(
            cast(npt.NDArray[np.float64], res),
            index=pd.Index(np.asarray(r), name='r', dtype=int),
            name=f'{path}l_moment{opts}',
            dtype=float,
            copy=False,
        )


def install():
    """Register the accessor methods."""
    for name, xsr in _XSR_SERIES.items():
        pd.api.extensions.register_series_accessor(name)(xsr)  # type: ignore
