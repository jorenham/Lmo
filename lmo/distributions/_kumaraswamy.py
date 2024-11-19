from __future__ import annotations

import math
import sys
from typing import TypeAlias, TypeVar, final

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import scipy.special as sc
from scipy.stats.distributions import rv_continuous

from lmo.special import harmonic
from ._lm import get_lm_func
from ._utils import ShapeInfo

if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override

__all__ = ("kumaraswamy_gen",)

_ArrF8: TypeAlias = onp.Array[tuple[int, ...], np.float64]

_XT = TypeVar("_XT", float | np.float64, _ArrF8)


_lm_kumaraswamy = get_lm_func("kumaraswamy")


# pyright: reportIncompatibleMethodOverride=false


@final
class kumaraswamy_gen(rv_continuous):
    @override
    def _argcheck(self, /, a: float, b: float) -> bool | np.bool_:
        return (a > 0) & (b > 0)

    @override
    def _shape_info(self, /) -> list[ShapeInfo]:
        ia = ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    @override
    def _get_support(self, /, a: float, b: float) -> tuple[float, float]:
        return 0.0, 1.0

    @override
    def _pdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)  # pyright: ignore[reportReturnType]

    @override
    def _logpdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return np.log(a * b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)

    @override
    def _cdf(self, /, x: _XT, a: float, b: float) -> _XT:
        return 1 - (1 - x**a) ** b  # pyright: ignore[reportReturnType]

    @override
    def _sf(self, /, x: _XT, a: float, b: float) -> _XT:
        return (1 - x**a) ** (b - 1)  # pyright: ignore[reportReturnType]

    @override
    def _isf(self, /, q: _XT, a: float, b: float) -> _XT:
        return (1 - q ** (1 / b)) ** (1 / a)  # pyright: ignore[reportReturnType]

    def _qdf(self, /, q: _XT, a: float, b: float) -> _XT:
        p = 1 - q
        return p ** (1 / (b - 1)) * (1 - p ** (1 / b)) ** (1 / (a - 1)) / (a * b)  # pyright: ignore[reportReturnType]

    @override
    def _ppf(self, /, q: _XT, a: float, b: float) -> _XT:
        return (1 - (1 - q) ** (1 / b)) ** (1 / a)  # pyright: ignore[reportReturnType]

    def _entropy(self, a: float, b: float) -> float:
        # https://wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - math.log(a * b)

    @override
    def _munp(self, /, n: int | npt.NDArray[np.intp], a: float, b: float) -> _ArrF8:
        return b * sc.beta(1 + n / a, b)
