"""Parametric inference."""

__all__ = 'GMMResult', 'fit'

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from scipy import optimize, special  # type: ignore

from ._lm import l_moment as l_moment_est
from ._lm_co import l_coscale as l_coscale_est
from ._utils import clean_orders, clean_trim
from .diagnostic import HypothesisTestResult, l_moment_bounds
from .theoretical import l_moment_from_ppf
from .typing import (
    AnyTrim,
    DistributionFunction,
    IntVector,
    OptimizeResult,
)


class GMMResult(NamedTuple):
    """
    Represents the Generalized Method of L-Moments (L-GMM) results.
    See [`lmo.inference.fit`][lmo.inference.fit] for details.

    Attributes:
        args:
            The estimated distribution arguments, as `(*shapes, loc, scale)`.
        success:
            Whether or not the optimizer exited successfully.
        eps:
            Final relative difference in the (natural) L-moment conditions.
        statistic:
            The minimized objective value, corresponding to the `weights`.
        n_samp:
            Amount of samples used to calculate the sample L-moment (after
            trimming).
        n_step:
            Number of GMM steps (the amount of times the weight matrix has
            been estimated).
        n_iter:
            Number of evaluations of the objective function (the theoretical
            L-moments).
        weights:
            The final weight (precision, inverse covariance) matrix.

    """
    n_samp: int
    n_step: int
    n_iter: int

    args: tuple[float | int, ...]
    success: bool
    statistic: float
    eps: npt.NDArray[np.float64]

    weights: npt.NDArray[np.float64]

    @property
    def n_arg(self) -> int:
        """The number of model parameters."""
        return len(self.args)

    @property
    def n_con(self) -> int:
        """The amount of L-moment conditions of the model."""
        return self.weights.shape[0]

    @property
    def n_extra(self) -> int:
        """
        The number of over-identifying L-moment conditions. For L-MM this is
        zero, otherwise, for L-GMM, it is strictly positive.
        """
        return self.n_con - self.n_arg

    @property
    def j_test(self) -> HypothesisTestResult:
        """
        Sargan-Hansen J-test for over-identifying restrictions; a hypothesis
        test for the invalidity of the model.

        The test is defined through two hypotheses:

        - $H_0$: The data satisfies the L-moment conditions, i.e. the model is
            "valid".
        - $H_1$: The data does not satisfy the L-moment conditions, i.e. the
            model is "invalid".

        References:
            - [J. D. Sargan (1958) - The Estimation of Economic Relationships
            Using Instrumental Variables](https://doi.org/10.2307%2F1907619)
            - [J. P. Hansen (1982) - Large Sample Properties of Generalized
            Method of Moments Estimators](https://doi.org/10.2307%2F1912775)

        """
        if not (df := self.n_extra):
            msg = 'The Sargan Hansen J-test requires `n_extra > 0`'
            raise ValueError(msg)

        stat = self.statistic
        pvalue = special.chdtr(df, stat)  # type: ignore
        return HypothesisTestResult(stat, pvalue)

    @property
    def AIC(self) -> float:  # noqa: N802
        """
        Akaike Information Criterion, based on the p-value of the J-test.
        Requires over-identified L-moment conditions, i.e. `n_extra > 0`.

        The AIC is useful for model selection, e.g. for finding the most
        appropriate probability distribution from the data (smaller is better).

        References:
            - [H. Akaike (1974) - A new look at the statistical model
            identification](https://doi.org/10.1109%2FTAC.1974.1100705)

        """
        return 2 * (self.n_arg - np.log(cast(float, self.j_test.pvalue)))

    @property
    def AICc(self) -> float:  # noqa: N802
        """
        A modification of the AIC that includes a bias-correction small
        sample sizes.

        References:
            - [N. Sugiura (1978) - Further analysis of the data by Akaike's
            information criterion and the finite
            corrections](https://doi.org/10.1080%2F03610927808827599)

        """
        n, k = self.n_samp, self.n_arg
        return self.AIC + 2 * k * (k + 1) / (n - k - 1)


def _loss_step(
    args: npt.NDArray[np.float64],
    l_fn: Callable[..., npt.NDArray[np.float64]],
    r: npt.NDArray[np.int64],
    l_r: npt.NDArray[np.float64],
    trim: AnyTrim,
    w_rr: npt.NDArray[np.float64],
) -> float:
    lmbda_r = l_fn(r, *args, trim=trim)

    if not np.all(np.isfinite(lmbda_r)):
        msg = f'failed to find the L-moments of ppf{args} with {trim=}'
        raise ValueError(msg)

    g_r = lmbda_r - l_r
    return np.sqrt(g_r.T @ w_rr @ g_r)  # type: ignore


def _get_l_moment_fn(ppf: DistributionFunction[...]):
    def l_moment_fn(
        r: IntVector,
        *args: Any,
        trim: AnyTrim = (0, 0),
    ) -> npt.NDArray[np.float64]:
        return l_moment_from_ppf(lambda q: ppf(q, *args), r, trim=trim)

    return l_moment_fn


def _get_weights_mc(
    y: npt.NDArray[np.float64],
    r: npt.NDArray[np.int64],
    trim: tuple[int, int] | tuple[float, float] = (0, 0),
) -> npt.NDArray[np.float64]:
    l_r = l_moment_est(
        y,
        r,
        trim=trim,
        axis=-1,
        # cache=True,
        # `y` is sorted -> stablesort is faster than quicksort
        sort='stable',
    )

    # l_rr = np.cov(l_r)

    # Use L-coscale instead of np.cov (more efficient and robust)
    # L-moment estimates follow a normal distribution.
    # Note that the L-scale of standard normal is 1/sqrt(pi).
    # l_r is fully unordered, so quicksort is likely to be faster than stable
    l_rr = l_coscale_est(l_r, sort='quicksort')
    # convert the L-coscale to an (asymmetric) quasi-covariance matrix
    l_rr *= l_rr.diagonal() * np.pi

    try:
        return np.linalg.inv(l_rr)
    except np.linalg.LinAlgError:
        # can occur for e.g. 1x1 or sub-rank cov matrices
        return np.linalg.pinv(l_rr)


def fit(  # noqa: C901
    ppf: DistributionFunction[...],
    args0: npt.ArrayLike,
    n_obs: int,
    l_moments: npt.ArrayLike,
    r: IntVector | None = None,
    trim: AnyTrim = (0, 0),
    *,
    k: int | None = None,
    k_max: int = 50,
    l_tol: float = 1e-4,

    l_moment_fn: Callable[..., npt.NDArray[np.float64]] | None = None,
    n_mc_samples: int = 9999,
    random_state: (
        int
        | np.random.Generator
        | np.random.RandomState
        | None
    ) = None,
    **kwds: Any,
) -> GMMResult:
    r"""
    Fit the distribution parameters using the (Generalized) Method of
    L-Moments (L-(G)MM).

    The goal is to find the "true" parameters $\theta_0$ of the distribution.
    In practise, this is done using a reasonably close estimate, $\theta$.

    In the (non-Generalized) Method of L-moments (L-MM), this is done by
    solving the system of equations $l^{(s, t)}_r = \lambda^{(s, t)}_r$,
    for $r = 1, \dots, k$, with $n = |\theta|$ the number of parameters.
    Because the amount of parameters matches the amount of *L-moment
    conditions*, the solution is *point-defined*, and can be found using
    simple least squares.

    L-GMM extends L-MM by allowing more L-moment conditions than there are
    free parameters, $m > n$. This requires solving an *over-identified*
    system of $m$ equations:

    $$
    \hat{\theta} =
        \mathop{\arg \min} \limits_{\theta \in \Theta}
        (\vec{\lambda}^{(s, t)}_r - \vec{l}^{(s, t)})^T
        W_m
        (\vec{\lambda}^{(s, t)}_r - \vec{l}^{(s, t)})
        \, ,
    $$

    where $W_m$ is a $m \times m$ weight matrix.

    The weight matrix is initially chosen as the matrix inverse of the
    non-parametric L-moment covariance matrix, see
    [`lmo.l_moment_cov`][lmo.l_moment_cov]. These weights are then plugged
    into the the equation above, and fed into
    [`scipy.optimize.minimize`][scipy.optimize.minimize], to obtain the
    initial parameter estimates.

    In the next step(s), Monte-Carlo sampling is used to draw samples from
    the distribution (using the current parameter estimates), with sample
    sizes matching that of the data. The L-moments of these samples are
    consequently used to to calculate the new weight matrix.

    Todo:
        - Raise on minimization error, warn on failed k-step convergence
        - Optional `integrality` kwarg with boolean mask for integral params.
        - Implement CUE: Continuously Updating GMM (i.e. implement and
            use  `_loss_cue()`, then run with `k=1`).

    Parameters:
        ppf:
            The (vectorized) quantile function of the probability distribution,
            with signature `(*args: float, q: T) -> T`.
        args0:
            Initial estimate of the distribution's parameter values.
        n_obs:
            Amount of observations.
        l_moments:
            Estimated sample L-moments. Must be a 1-d array-like s.t.
            `len(l_moments) >= len(args0)`.
        r:
            The orders of `l_moments`. Defaults to `[1, ..., len(l_moments)]`.
        trim:
            The L-moment trim-length(s) to use. Currently, only integral
            trimming is supported.

    Other Parameters:
        k:
            If set to a positive integer, exactly $k$ steps will be run.
            Will be ignored if `n_extra=0`.
        k_max:
            Maximum amount of steps to run while not reaching convergence.
            Will be ignored if $k$ is specified or if `n_extra=0`.
        l_tol:
            Error tolerance in the parametric L-moments (unit-standardized).
            Will be ignored if $k$ is specified or if `n_extra=0`.
        l_moment_fn:
            Function for parametric L-moment calculation, with signature:
            `(r: int64[], *args, trim: float[2] | int[2]) -> float64[]`.
        n_mc_samples:
            The number of Monte-Carlo (MC) samples drawn from the
            distribution to to form the weight matrix in step $k > 1$.
            Will be ignored if `n_extra=0`.
        random_state:
            A seed value or [`numpy.random.Generator`][numpy.random.Generator]
            instance, used for weight matrix estimation in step $k > 1$.
            Will be ignored if `n_extra=0`.
        **kwds:
            Additional keyword arguments to be passed to
            [`scipy.optimize.minimize`][scipy.optimize.minimize].

    Raises:
        ValueError: Invalid arguments.

    Returns:
        result: An instance of [`GMMResult`][`lmo.inference.GMMResult`].

    References:
        - [Alvarez et al. (2023) - Inference in parametric models with
        many L-moments](https://doi.org/10.48550/arXiv.2210.04146)

    """
    # Validate the input
    theta = np.asarray_chkfinite(args0, np.float64)
    n_par = len(theta)

    l_r = np.asarray_chkfinite(l_moments)
    if l_r.ndim != 1:
        msg = 'l_moments must be a 1-d array-like'
        raise ValueError(msg)

    if r is None:
        _r = np.arange(1, len(l_r) + 1)
    else:
        _r = clean_orders(np.asarray(r, np.int64))

        _r_nonzero = _r != 0
        l_r, _r = l_r[_r_nonzero], _r[_r_nonzero]

    if (n_con := len(_r)) < n_par:
        msg = f'under-determined L-moment conditions: {n_con} < {n_par}'
        raise ValueError(msg)

    _trim = clean_trim(trim)
    _r = np.arange(1, n_con + 1, dtype=np.int64)

    # Individual L-moment "natural" scaling constants, making their magnitudes
    # order- and trim- agnostic (used in convergence criterion)
    l_r_ub = np.r_[1, l_moment_bounds(_r[1:], trim=_trim)]
    l_2c = l_r[1] / l_r_ub[1]
    scale_r = cast(npt.NDArray[np.float64], 1 / (l_2c * l_r_ub))

    # Initial parametric population L-moments
    _l_moment_fn = l_moment_fn or _get_l_moment_fn(ppf)
    lmbda_r = _l_moment_fn(_r, *theta, trim=_trim)

    # Prepare the converge criteria
    if k:
        k_min = k_max = k
        epsmax = np.inf
    elif n_par == n_con:
        k_min = k_max = 1
        epsmax = np.inf
    else:
        k_min, epsmax = 1, l_tol

    # Random number generator
    if isinstance(random_state, np.random.RandomState | np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if n_con > n_par:
        # Draw random quantiles, and pre-sort for L-moment estimation speed
        qs = rng.random((n_mc_samples, n_obs))
        qs.sort(axis=1)
    else:
        qs = None

    # Set the default `scipy.optimize.minimize` method
    kwds.setdefault('method', 'Nelder-Mead')

    # Initial state
    _k = 0
    i = 1
    eps = np.full(n_con, np.nan)
    fun = np.nan
    success = False
    w_rr = np.eye(n_con) * scale_r

    for _k in range(1, k_max + 1):
        # calculate the weight matrix
        if n_con > n_par:
            w_rr = _get_weights_mc(
                ppf(cast(npt.NDArray[np.float64], qs), *theta),
                _r,
                trim=_trim,
            )

        # run the optimizer
        res = cast(
            OptimizeResult,
            optimize.minimize(  # type: ignore
                _loss_step,
                theta,
                args=(_l_moment_fn, _r, l_r, _trim, w_rr),
                **kwds,
            ),
        )
        i += res.nfev
        fun = res.fun
        theta = res.x

        if not res.success:
            break

        if _k < k_min:
            continue

        # re-evaluate the theoretical L-moments for the new params
        lmbda_r0 = lmbda_r
        lmbda_r = _l_moment_fn(_r, *theta, trim=_trim)

        # convergence criterion
        eps = (lmbda_r - lmbda_r0) * scale_r
        if np.max(np.abs(eps)) <= epsmax:
            success = True
            break

    return GMMResult(
        args=tuple(theta),
        success=success,
        statistic=fun**2,
        eps=eps,
        n_samp=cast(int, n_obs - sum(_trim)),
        n_step=_k,
        n_iter=i,
        weights=w_rr,
    )
