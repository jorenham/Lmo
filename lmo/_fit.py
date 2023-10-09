__all__ = ('l_gmm',)

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # type: ignore

from ._lm import l_moment
from .theoretical import l_moment_from_ppf
from .typing import AnyTrim


def _midspace(n: int) -> npt.NDArray[np.float64]:
    start = 1 / (2 * n)
    return np.linspace(start, 1 - start, n)


def l_gmm(
    ppf: Callable[..., npt.NDArray[np.float64]],
    data: npt.ArrayLike,
    b0: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    moments: int | None = None,
    *,
    pdf: Callable[..., npt.NDArray[np.float64]] | None = None,
    parametric: Literal[True, False, 'semi'] = True,
    **kwds: Any,
) -> npt.NDArray[np.float64]:
    """
    Generalized method of L-moments.

    Todo:
        k-step estimation: repeat 2nd stel until np.max(np.abs(b2)) < eps.

    """
    b0 = np.atleast_1d(np.asarray_chkfinite(b0, np.float64))
    if b0.ndim > 1:
        msg = 'b0 must be 1-d'
        raise TypeError(msg)
    if not len(b0):
        msg = 'b0 cannot be empty'
        raise TypeError(msg)

    n_lmo = moments or len(b0)
    if n_lmo < len(b0):
        msg = 'moments cannot be smaller than len(b0)'
        raise ValueError(msg)

    y = np.sort(data)
    n_obs = len(y)

    l_y = l_moment(y, np.arange(1, n_lmo + 1), trim=trim, sort='stable')

    q_hat = _midspace(n_obs)

    # q_pow = np.asarray([mid_pointer**x for x in range(moments)])
    q_pow = np.vander(q_hat, n_lmo, increasing=True) / n_obs

    def lmoment_func(*args: float) -> npt.NDArray[np.float64]:
        # return np.mean(powers * ppf(mid_pointer, *args), axis=1)
        # return q_pow.T @ ppf(q_hat, *args)
        return l_moment_from_ppf(
            lambda q: ppf(q, *args),  # type: ignore
            np.arange(1, n_lmo + 1),
            trim=trim,
        )

    def g1(theta: npt.NDArray[np.float64]) -> float:
        err = lmoment_func(*theta) - l_y
        return err @ err.T  # type: ignore

    b1 = cast(
        npt.NDArray[np.float64],
        minimize(g1, b0, **kwds).x,  # type: ignore [reportUnknownMemberType]
    )

    if parametric and pdf is not None:
        x = ppf(q_hat, *b1) if parametric == 'semi' else y
        qdf = 1 / pdf(x, *b1)
    else:
        qdf = 1 / np.histogram(y, bins='fd', density=True)[0]

    weight = np.minimum(q_hat, q_hat[:, None]) - np.outer(q_hat, q_hat)
    weight *= np.outer(qdf, qdf)
    weight = np.linalg.pinv(q_pow.T @ weight @ q_pow)

    def g2(theta: npt.NDArray[np.float64]) -> float:
        err = lmoment_func(*theta) - l_y
        return err @ weight @ err.T  # type: ignore

    return cast(
        npt.NDArray[np.float64],
        minimize(g2, b1, **kwds).x,  # type: ignore
    )
