from __future__ import annotations

import functools
import math
import sys
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, final

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy.special as sc
from scipy.stats._distn_infrastructure import (
    _ShapeInfo,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from scipy.stats.distributions import rv_continuous as _rv_continuous

from lmo.special import harmonic
from lmo.theoretical import l_moment_from_ppf

from ._lm import get_lm_func

if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    import lmo.typing.scipy as lspt


__all__ = ("kumaraswamy_gen",)

_F8: TypeAlias = float | np.float64
_ArrF8: TypeAlias = onpt.Array[tuple[int, ...], np.float64]

_XT = TypeVar("_XT", _F8, _ArrF8)


_lm_kumaraswamy = get_lm_func("kumaraswamy")


# pyright: reportIncompatibleMethodOverride=false

@final
class kumaraswamy_gen(_rv_continuous):
    @override
    def _argcheck(self, /, a: _F8, b: _F8) -> bool | np.bool_:
        return (a > 0) & (b > 0)

    @override
    def _shape_info(self, /) -> list[_ShapeInfo]:
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    @override
    def _get_support(self, /, a: _F8, b: _F8) -> tuple[_F8, _F8]:
        return 0.0, 1.0

    @override
    def _pdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return cast(_XT, a * b * x ** (a - 1) * (1 - x**a) ** (b - 1))

    @override
    def _logpdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return cast(
            _XT,
            np.log(a * b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a),
        )

    @override
    def _cdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return cast(_XT, 1 - (1 - x**a) ** b)

    @override
    def _sf(self, /, x: _XT, a: float, b: float) -> _XT:
        return cast(_XT, (1 - x**a) ** (b - 1))

    @override
    def _isf(self, /, q: _XT, a: float, b: float) -> _XT:
        return cast(_XT, (1 - q ** (1 / b)) ** (1 / a))

    def _qdf(self, /, q: _XT, a: float, b: float) -> _XT:
        p = 1 - q
        return cast(
            _XT,
            p ** (1 / (b - 1)) * (1 - p ** (1 / b)) ** (1 / (a - 1)) / (a * b),
        )

    @override
    def _ppf(self, /, q: _XT, a: float, b: float) -> _XT:
        return cast(_XT, (1 - (1 - q) ** (1 / b)) ** (1 / a))

    def _entropy(self, a: float, b: float) -> float:
        # https://wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - math.log(a * b)

    @override
    def _munp(
        self,
        /,
        n: int | np.integer[Any] | npt.NDArray[np.integer[Any]],
        a: float,
        b: float,
    ) -> _ArrF8:
        return b * sc.beta(1 + n / a, b)

    def _l_moment(
        self,
        r: npt.NDArray[np.int_],
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
