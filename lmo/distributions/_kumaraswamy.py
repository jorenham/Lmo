from __future__ import annotations

import math
import sys
from typing import Any, Final, TypeAlias, final

import numpy as np
import optype.numpy as onp
import scipy.special as sps

import lmo.typing as lmt
from lmo.special import harmonic
from ._lm import get_lm_func

if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override

__all__ = ("kumaraswamy_gen",)

_Float: TypeAlias = float | np.floating[Any]
_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_FloatOrND: TypeAlias = _Float | onp.ArrayND[np.floating[Any]]

_lm_kumaraswamy: Final = get_lm_func("kumaraswamy")


# pyright: reportIncompatibleMethodOverride=false
# pyright: reportUnusedFunction=false


@final
class kumaraswamy_gen(lmt.rv_continuous):
    # https://wikipedia.org/wiki/Kumaraswamy_distribution

    @override
    def _argcheck(self, /, a: float, b: float) -> bool | np.bool_:
        return (a > 0) & (b > 0)

    @override
    def _shape_info(self, /) -> list[lmt.ShapeInfo]:
        ia = lmt.ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = lmt.ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    @override
    def _get_support(self, /, a: float, b: float) -> tuple[float, float]:
        return 0.0, 1.0

    @override
    def _pdf(self, /, x: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)

    @override
    def _logpdf(self, /, x: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return np.log(a * b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)

    @override
    def _cdf(self, /, x: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return 1 - (1 - x**a) ** b

    @override
    def _sf(self, /, x: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return (1 - x**a) ** (b - 1)

    @override
    def _isf(self, /, q: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return (1 - q ** (1 / b)) ** (1 / a)

    def _qdf(self, /, q: _FloatOrND, a: float, b: float) -> _FloatOrND:
        p = 1 - q
        return p ** (1 / (b - 1)) * (1 - p ** (1 / b)) ** (1 / (a - 1)) / (a * b)

    @override
    def _ppf(self, /, q: _FloatOrND, a: float, b: float) -> _FloatOrND:
        return (1 - (1 - q) ** (1 / b)) ** (1 / a)

    def _entropy(self, a: float, b: float) -> float:
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - math.log(a * b)

    @override
    def _munp(
        self,
        /,
        n: int | onp.ArrayND[np.intp],
        a: float,
        b: float,
    ) -> _Float | _FloatND:
        return b * sps.beta(1 + n / a, b)
