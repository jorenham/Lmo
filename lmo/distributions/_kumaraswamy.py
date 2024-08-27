from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeAlias, cast, final, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy.special as sc
from scipy.stats._distn_infrastructure import (
    _ShapeInfo,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from scipy.stats.distributions import rv_continuous as _rv_continuous

import lmo.typing.scipy as lspt
from lmo.special import harmonic
from lmo.theoretical import l_moment_from_ppf
from ._lm import get_lm_func


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ('kumaraswamy_gen',)


_ArrF8: TypeAlias = onpt.Array[tuple[int, ...], np.float64]

_lm_kumaraswamy = get_lm_func('kumaraswamy')


@final
class kumaraswamy_gen(cast(type[lspt.AnyRV], _rv_continuous)):  # pyright: ignore[reportGeneralTypeIssues]
    def _argcheck(self, a: float, b: float) -> bool:
        return (a > 0) & (b > 0)

    def _shape_info(self) -> Sequence[_ShapeInfo]:
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _get_support(self, a: float, b: float) -> tuple[float, float]:  # noqa: ARG002
        return 0.0, 1.0

    def _pdf(self, x: _ArrF8, a: float, b: float) -> _ArrF8:
        return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)

    def _logpdf(self, x: _ArrF8, a: float, b: float) -> _ArrF8:
        return np.log(a * b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)

    def _cdf(self, x: _ArrF8, a: float, b: float) -> _ArrF8:
        return 1 - (1 - x**a) ** b

    def _sf(self, x: _ArrF8, a: float, b: float) -> _ArrF8:
        return (1 - x**a) ** (b - 1)

    def _isf(self, q: _ArrF8, a: float, b: float) -> _ArrF8:
        return (1 - q ** (1 / b)) ** (1 / a)

    def _qdf(self, q: _ArrF8, a: float, b: float) -> _ArrF8:
        return (
            (1 - q) ** (1 / (b - 1))
            * (1 - (1 - q) ** (1 / b)) ** (1 / (a - 1))
            / (a * b)
        )

    @overload
    def _ppf(self, q: _ArrF8, a: float, b: float) -> _ArrF8: ...
    @overload
    def _ppf(self, q: float, a: float, b: float) -> np.float64: ...
    def _ppf(
        self,
        q: float | _ArrF8,
        a: float,
        b: float,
    ) -> np.float64 | _ArrF8:
        return (1 - (1 - q) ** (1 / b)) ** (1 / a)

    def _entropy(self, a: float, b: float) -> np.float64:
        # https://wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - np.log(a * b)

    def _munp(self, n: int, a: float, b: float) -> float:
        return b * cast(float, sc.beta(1 + n / a, b))

    def _l_moment(
        self,
        r: npt.NDArray[np.intp],
        a: float,
        b: float,
        *,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim
        if quad_opts is not None or isinstance(trim[0], float):
            out = l_moment_from_ppf(
                functools.partial(self._ppf, a=a, b=b),
                r,
                trim=trim,
                quad_opts=quad_opts,
            )
        else:
            out = _lm_kumaraswamy(r, s, t, a, b)

        return np.atleast_1d(out)
