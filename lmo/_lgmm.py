__all__ = ('GMMResult', 'l_gmm')

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # type: ignore

from ._lm import l_moment
from ._utils import clean_order, clean_trim, plotting_positions
from .theoretical import l_moment_from_ppf
from .typing import AnyTrim


class GMMResult(NamedTuple):
    params: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    niter: int
    abserr: float
    success: bool

    lmoments: npt.NDArray[np.float64]
    trim: tuple[int, int] | tuple[float, float]


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
    maxiter: int = 50,
    tol: float = 1e-4,
    alpha: float = 0.35,
    beta: float | None = None,
    **kwds: Any,
) -> GMMResult:
    """Generalized method of L-moments."""
    b0 = np.asarray_chkfinite(b0, np.float64)
    if b0.ndim > 1:
        msg = 'b0 must be 1-d'
        raise TypeError(msg)
    if not len(b0):
        msg = 'b0 cannot be empty'
        raise TypeError(msg)

    if moments:
        n_lmo = clean_order(moments, name='moments', rmin=len(b0))
    else:
        n_lmo = len(b0)

    y = np.sort(data)
    n_obs = len(y)

    l_y = l_moment(y, np.arange(1, n_lmo + 1), trim=trim, sort='stable')

    pp = plotting_positions(n_obs, alpha, beta)
    pp_pow = np.vander(pp, n_lmo, increasing=True) / n_obs

    def lmoment_func(*args: float) -> npt.NDArray[np.float64]:
        # return np.mean(powers * ppf(mid_pointer, *args), axis=1)
        # return q_pow.T @ ppf(q_hat, *args)
        return l_moment_from_ppf(
            lambda q: ppf(q, *args),  # type: ignore
            np.arange(1, n_lmo + 1),
            trim=trim,
        )

    def objective(
        theta: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64],
    ) -> float:
        err = lmoment_func(*theta) - l_y
        return err @ weight @ err.T  # type: ignore

    weights = np.eye(n_lmo)
    k = 1

    b_k = cast(
        npt.NDArray[np.float64],
        minimize(
            objective,
            b0,
            (weights,),
            **kwds,
        ).x,  # type: ignore [reportUnknownMemberType]
    )
    eps = np.inf

    for k in range(2, maxiter + 1):  # noqa: B007
        if parametric and pdf is not None:
            x = ppf(pp, *b_k) if parametric == 'semi' else y
            qdf = 1 / pdf(x, *b_k)
        else:
            qdf = 1 / np.histogram(y, bins='fd', density=True)[0]

        weights = np.minimum(pp, pp[:, None]) - np.outer(pp, pp)
        weights *= np.outer(qdf, qdf)
        weights = np.linalg.pinv(pp_pow.T @ weights @ pp_pow)

        b_k_prev = b_k
        b_k = cast(
            npt.NDArray[np.float64],
            minimize(objective, b_k_prev, (weights,), **kwds).x,
        )

        eps = np.max(np.abs(b_k - b_k_prev))
        if eps <= tol:
            break

    return GMMResult(
        params=b_k,
        weights=weights,
        niter=k,
        abserr=eps,
        success=eps <= tol or k <= 2,
        lmoments=l_y,
        trim=clean_trim(trim),
    )
