from __future__ import annotations

import functools
import math
import sys
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy.special as sc
from scipy.stats._distn_infrastructure import _ShapeInfo  # noqa: PLC2701
from scipy.stats.distributions import rv_continuous


if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override

from lmo.theoretical import l_moment_from_ppf
from ._lm import get_lm_func


if TYPE_CHECKING:
    import lmo.typing.scipy as lspt


__all__ = ('wakeby_gen',)


# NOTE: this is equivalent to `float` IFF `numpy >= 2.2`, see:
# https://github.com/numpy/numpy/pull/27334
_F8: TypeAlias = float | np.float64
_ArrF8: TypeAlias = onpt.Array[tuple[int, ...], np.float64]

_XT = TypeVar('_XT', _F8, _ArrF8)

_MICRO: Final = 1e-6
_NaN: Final = float('nan')
_INF: Final = float('inf')


def _wakeby_ub(b: _F8, d: _F8, f: _F8) -> _F8:
    """Upper bound of the domain of Wakeby's distribution function."""
    if d < 0:
        return f / b - (1 - f) / d
    if b and (f == 1 or abs(f - 1) <= _MICRO):
        return 1 / b
    return _INF


def _wakeby_isf0(q: _F8, b: _F8, d: _F8, f: _F8) -> _F8:
    """Inverse survival function, does not validate params."""
    if q <= 0:
        return _wakeby_ub(b, d, f)
    if q >= 1:
        return 0.0

    if f == 0:
        u = 0.0
    elif b == 0:
        u = math.log(q)
    else:
        u = (q**b - 1) / b

    if f == 1:
        v = 0.0
    elif d == 0:
        v = u if b == 0 and f != 0 else math.log(q)
    else:
        v = -(q ** (-d) - 1) / d

    return -f * u - (1 - f) * v


_wakeby_isf = np.vectorize(_wakeby_isf0, [float])


def _wakeby_qdf(p: _XT, b: _F8, d: _F8, f: _F8) -> _XT:
    """Quantile density function (QDF), the derivative of the PPF."""
    q = 1 - p
    lhs = f * q ** (b - 1)
    rhs = (1 - f) / q ** (d + 1)
    return cast(_XT, lhs + rhs)


def _wakeby_sf0(x: _F8, b: _F8, d: _F8, f: _F8) -> _F8:  # noqa: C901
    """
    Numerical approximation of Wakeby's survival function.

    Uses a modified version of Halley's algorithm, as originally implemented
    by J.R.M. Hosking in fortran: https://lib.stat.cmu.edu/general/lmoments
    """
    if x <= 0:
        return 1.0

    if x >= _wakeby_ub(b, d, f):
        return 0.0

    if b == f == 1:
        # standard uniform
        return 1 - x
    if b == d == 0:
        # standard exponential
        assert f == 1
        return math.exp(-x)
    if f == 1:
        # GPD (bounded above)
        return (1 - b * x) ** (1 / b)
    if f == 0:
        # GPD (no upper bound)
        return (1 + d * x) ** (-1 / d)
    if b == d and b > 0:
        # unnamed special case
        cx = b * x
        return (
            (2 * f - cx - 1 + math.sqrt((cx + 1) ** 2 - 4 * cx * f)) / (2 * f)
        ) ** (1 / b)
    if b == 0 and d != 0:
        # https://wikipedia.org/wiki/Lambert_W_function
        # it's easy to show that this is valid for all x, f, and d
        w = (1 - f) / f
        return float(
            (w / sc.lambertw(w * math.exp((1 + d * x) / f - 1))) ** (1 / d),
        )

    z: _F8
    if x < _wakeby_isf0(0.9, b, d, f):
        z = 0
    elif x >= _wakeby_isf0(0.01, b, d, f):
        if d < 0:
            z = math.log(1 + (x - f / b) * d / (1 - f)) / d
        elif d > 0:
            z = math.log(1 + x * d / (1 - f)) / d
        else:
            z = (x - f / b) / (1 - f)
    else:
        z = 0.7

    eps = 1e-8
    maxit = 50
    ufl = math.log(math.nextafter(0, 1))

    for _ in range(maxit):
        bz = -b * z
        eb = math.exp(bz) if bz >= ufl else 0 * bz
        gb = (1 - eb) / b if abs(b) > eps else z

        ed = math.exp(d * z)
        gd = (1 - ed) / d if abs(d) > eps else -z

        x_est = f * gb - (1 - f) * gd
        qd0 = x - x_est
        qd1 = f * eb + (1 - f) * ed
        qd2 = -f * b * eb + (1 - f) * d * ed

        tmp: _F8 = qd1 - 0.5 * qd0 * qd2 / qd1
        if tmp <= 0:
            tmp = qd1

        # NOTE: float64 will be fixed
        z_inc = min(qd0 / tmp, 3)  # pyright: ignore[reportArgumentType]
        z_new = z + z_inc
        if z_new <= 0:
            z /= 5
            continue
        z = z_new

        if abs(z_inc) <= eps:
            break
    else:
        warnings.warn(
            'Wakeby SF did not converge, the result may be unreliable',
            RuntimeWarning,
            stacklevel=4,
        )

    return math.exp(-z) if -z >= ufl else 0


_wakeby_sf: Final = np.vectorize(_wakeby_sf0, [float])
_wakeby_lm: Final = get_lm_func('wakeby')


class wakeby_gen(rv_continuous):
    @override
    def _argcheck(self, /, b: _F8, d: _F8, f: _F8) -> bool | np.bool_:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            np.isfinite(b)
            & np.isfinite(d)
            & (b + d >= 0)
            & ((b + d > 0) | (f == 1))
            & (f >= 0)
            & (f <= 1)
            & ((f > 0) | (b == 0))
            & ((f < 1) | (d == 0))
        )

    @override
    def _shape_info(self) -> list[_ShapeInfo]:
        ibeta = _ShapeInfo('b', False, (-np.inf, np.inf), (False, False))
        idelta = _ShapeInfo('d', False, (-np.inf, np.inf), (False, False))
        iphi = _ShapeInfo('f', False, (0, 1), (True, True))
        return [ibeta, idelta, iphi]

    @override
    def _get_support(self, /, b: _F8, d: _F8, f: _F8) -> tuple[_F8, _F8]:
        if not self._argcheck(b, d, f):
            return _NaN, _NaN

        return 0.0, _wakeby_ub(b, d, f)

    @override
    def _fitstart(
        self,
        /,
        data: _ArrF8,
        args: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float, float, float]:
        #  Arbitrary, but the default f=1 is a bad start
        return cast(
            tuple[float, float, float, float, float],
            super()._fitstart(data, args or (1.0, 1.0, 0.5)),
        )

    @override
    def _pdf(self, /, x: _XT, b: float, d: float, f: float) -> _XT:  # pyright: ignore[reportIncompatibleMethodOverride]
        # application of the inverse function theorem
        return 1 / self._qdf(self._cdf(x, b, d, f), b, d, f)

    @override
    def _cdf(self, /, x: _XT, b: float, d: float, f: float) -> _XT:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1 - _wakeby_sf(x, b, d, f)

    def _qdf(self, /, q: _XT, b: float, d: float, f: float) -> _XT:
        return _wakeby_qdf(q, b, d, f)

    @override
    def _ppf(self, /, q: _XT, b: float, d: float, f: float) -> _XT:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _wakeby_isf(1 - q, b, d, f)

    @override
    def _isf(self, /, q: _XT, b: float, d: float, f: float) -> _XT:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _wakeby_isf(q, b, d, f)

    @override
    def _stats(
        self,
        b: float,
        d: float,
        f: float,
    ) -> tuple[float, float, float | None, float | None]:
        if d >= 1:
            # hard NaN (not inf); indeterminate integral
            return _NaN, _NaN, _NaN, _NaN

        u = f / (1 + b)
        v = (1 - f) / (1 - d)

        m1 = u + v

        if d >= 1 / 2:
            return m1, _NaN, _NaN, _NaN

        m2 = u**2 / (1 + 2 * b) + 2 * u * v / (1 + b - d) + v**2 / (1 - 2 * d)

        # no skewness and kurtosis (yet?); the equations are kinda huge...
        if d >= 1 / 3:
            return m1, m2, _NaN, _NaN
        m3 = None

        if d >= 1 / 4:
            return m1, m2, m3, _NaN
        m4 = None

        return m1, m2, m3, m4

    def _entropy(self, b: float, d: float, f: float) -> float:
        """
        Entropy can be calculated from the QDF (PPF derivative) as the
        Integrate[Log[QDF[u]], {u, 0, 1}]. This is the (semi) closed-form
        solution in this case.
        At the time of writing, this result appears to be novel.

        The `f` conditionals are the limiting cases, e.g. for uniform,
        exponential, and GPD (genpareto).
        """
        if f == 0:
            return 1 + d
        if f == 1:
            return 1 - b

        bd = b + d
        assert bd > 0

        ibd = 1 / bd
        return 1 - b + bd * float(sc.hyp2f1(1, ibd, 1 + ibd, f / (f - 1)))

    def _l_moment(
        self,
        r: npt.NDArray[np.int64],
        b: float,
        d: float,
        f: float,
        *,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = l_moment_from_ppf(
                cast(
                    Callable[[float], float],
                    functools.partial(self._ppf, b=b, d=d, f=f),
                ),
                r,
                trim=trim,
                quad_opts=quad_opts,
            )
            out = lmbda_r
        else:
            out = _wakeby_lm(r, s, t, b, d, f)

        return np.atleast_1d(out)
