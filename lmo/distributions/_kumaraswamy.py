from __future__ import annotations

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


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ('kumaraswamy_gen',)


_ArrF8: TypeAlias = onpt.Array[tuple[int, ...], np.float64]


def _kumaraswamy_lmo0(
    r: int,
    s: int,
    t: int,
    a: float,
    b: float,
) -> np.float64:
    if r == 0:
        return np.float64(1)

    k = np.arange(t + 1, r + s + t + 1)
    return (
        np.sum(
            (-1) ** (k - 1)
            * cast(_ArrF8, sc.comb(r + k - 2, r + t - 1))  # pyright: ignore[reportUnknownMemberType]
            * cast(_ArrF8, sc.comb(r + s + t, k))  # pyright: ignore[reportUnknownMemberType]
            * cast(_ArrF8, sc.beta(1 / a, 1 + k * b))
            / a,
        )
        / r
    )


_kumaraswamy_lmo = np.vectorize(_kumaraswamy_lmo0, [float], excluded={1, 2})


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

    def _qdf(self, u: _ArrF8, a: float, b: float) -> _ArrF8:
        return (
            (1 - u) ** (1 / (b - 1))
            * (1 - (1 - u) ** (1 / b)) ** (1 / (a - 1))
            / (a * b)
        )

    @overload
    def _ppf(self, u: float, a: float, b: float) -> np.float64: ...
    @overload
    def _ppf(self, u: _ArrF8, a: float, b: float) -> _ArrF8: ...
    def _ppf(
        self,
        u: float | _ArrF8,
        a: float,
        b: float,
    ) -> np.float64 | _ArrF8:
        return (1 - (1 - u) ** (1 / b)) ** (1 / a)

    def _entropy(self, a: float, b: float) -> np.float64:
        # https://wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - np.log(a * b)

    def _munp(self, n: int, a: float, b: float) -> float:
        return b * cast(float, sc.beta(1 + n / a, b))

    def _l_moment(
        self,
        r: npt.NDArray[np.int64],
        a: float,
        b: float,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim
        if quad_opts is not None or isinstance(s, float):

            def _ppf(u: float, /) -> np.float64:
                return self._ppf(u, a, b)

            out = l_moment_from_ppf(_ppf, r, trim=trim, quad_opts=quad_opts)
        else:
            out = cast(_ArrF8, _kumaraswamy_lmo(r, s, t, a, b))

        return np.atleast_1d(out)
