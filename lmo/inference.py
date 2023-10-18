"""Parametric inference."""

__all__ = 'GMMResult', 'fit'

import functools
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from scipy import optimize, special  # type: ignore

from ._lm import l_moment, l_moment_cov
from ._utils import clean_trim
from .diagnostic import HypothesisTestResult
from .theoretical import l_moment_from_ppf
from .typing import AnyInt, AnyTrim, OptimizeResult

GMMMethod: TypeAlias = Literal[
    # 1-step GMM:
    # minimize the objective using initial (nonparametric) weights
    'L-GMM-1',

    # 2-step GMM:
    # start with `L-GMM-1`, then run again with new (parametric) weights
    'L-GMM-2',

    # $k$-step Iterative GMM:
    # start with `L-GMM-2`, then iterate the 2nd step until convergence
    'L-GMM-k',

    # Continuously Updating GMM (TODO):
    # similar to `L-GMM-1`, but estimate the weights within objective function
    # 'L-GMM-CUE',
]


class GMMResult(NamedTuple):
    """
    Represents the Generalized Method of L-Moments (L-GMM) results.

    Attributes:
        args:
            The estimated distribution arguments, as `(*shapes, loc, scale)`.
        success:
            Whether or not the optimizer exited successfully.
        eps:
            Maxmimum absolute difference of the change in args.
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
    eps: float

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
        pvalue = special.chdtrc(df, stat)  # type: ignore
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
        pval_gof = special.chdtrc(self.n_con, self.statistic)  # type: ignore
        return 2 * (self.n_arg - np.log(pval_gof))

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
    ppf: Callable[..., float],
    l_r: npt.NDArray[np.float64],
    trim: AnyTrim,
    w_rr: npt.NDArray[np.float64],
) -> float:
    lmbda_r = l_moment_from_ppf(
        lambda q: ppf(q, *args),
        np.arange(1, len(l_r) + 1),
        trim=trim,
    )
    if not np.all(np.isfinite(lmbda_r)):
        msg = f'failed to find the L-moments of ppf{args} with {trim=}'
        raise ValueError(msg)

    g_r = lmbda_r - l_r
    return cast(float, g_r.T @ w_rr @ g_r)


def fit(
    ppf: Callable[..., npt.NDArray[np.float64] | float],
    args0: npt.ArrayLike,
    data: npt.ArrayLike,
    trim: AnyTrim = (0, 0),
    *,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    method: GMMMethod = 'L-GMM-1',
    n_extra: int = 1,
    n_mc_samples: int = 9999,
    maxiter: int = 50,
    tol: float = 1e-4,

    optimizer: str | Callable[..., OptimizeResult] = 'Nelder-Mead',
    random_state: AnyInt | np.random.Generator | None = None,
) -> GMMResult:
    """
    Fit the distribution parameters using the Generalized Method of L-Moments
    (L-GMM).

    Todo:
        - Short explanation of L-GMM, the GMM methods, and the "warm start"
        - Code examples (e.g. GEV)
        - Implement L-MM (i.e. `L-GMM-1` but with `extra = 0` and
            `w_rr = np.eye(n_lmo)`)
        - Implement L-GMM-CUE: Continuously Updating GMM (i.e. implement and
            use  `_loss_cue()`, then run `L-GMM-1`).
        - Automatic `extra` selection (e.g. AICc, or Sargan's J test)
        - consistent reporting of minimize errors / no L-GMM-k convergence
        - `integrality` param, see `scipy.optimize.differential_evolution`

    Parameters:
        ppf:
            The (vectorized) quantile function of the probability distribution,
            with signature `(*args: float, q: T) -> T`.
        args0:
            Initial estimate of the distribution's parameter values.
        data:
            The sample data as 1-d array-like.
        n_extra:
            The amount of over-identifying L-moment conditions to use. E.g.
            if `len(args0) == 3` and `extra == 1`, `4` L-moment conditions
            will be used in total. Must be `>=0`.
        trim:
            The L-moment trim-length(s) to use. Currently, only integral
            trimming is supported.
        method:
            The GMM method to use. Currently, 1-step (`L-GMM-1`), 2-step
            (`L-GMM-2`), and k-step (`L-GMM-k`), are supported.
        maxiter:
            Maximum amount of optimization steps to use. Defaults to 50.
            Only relevant for `method=L-GMM-k`.
        tol:
            Absolute maximum error in the maximum absolute change of the
            parameter values. Only relevant for k-step GMM
            (`method='L-GMM-k'`).
        optimizer:
            See the `method` parameter in
            [`scipy.optimize.minimize`][scipy.optimize.minimize]. Defaults
            to `'Nelder-Mead'`. The optimizer must accept `bounds`.
        bounds:
            See the `bounds` parameter in
            [`scipy.optimize.minimize`][scipy.optimize.minimize].
        n_mc_samples:
            The number of Monte-Carlo samples drawn from the distribution to
            to form the weight matrix. Only relevant for 2-step and $k$-step
            GMM (`method = 'L-GMM-2' | 'L-GMM-k'`).
        random_state:
            A seed value or [`numpy.random.Generator`][numpy.random.Generator]
            instance, used in Monte-Carlo weight matrix estimation.
            Only relevant for 2-step and $k$-step GMM
            (`method = 'L-GMM-2' | 'L-GMM-k'`).

    Raises:
        ValueError: Invalid arguments.

    Returns:
        result: An instance of [`GMMResult`][`lmo.inference.GMMResult`].

    References:
        - [Alvarez et al. (2023) - Inference in parametric models with
        many L-moments](https://doi.org/10.48550/arXiv.2210.04146)
    """
    theta = np.asarray_chkfinite(args0)
    x = np.sort(np.asarray_chkfinite(data))

    n_obs = x.size
    n_par = len(theta)
    n_con = n_par + n_extra

    _trim = clean_trim(trim)
    r = np.arange(1, n_con + 1)
    l_r = l_moment(x, r, trim=_trim, sort='stable')

    loss_fn = _loss_step

    if method == 'L-GMM-1':
        k_min, k_max, eps_max = 1, 1, np.inf
    elif method == 'L-GMM-2':
        k_min, k_max, eps_max = 2, 2, np.inf
    elif method == 'L-GMM-k':
        k_min, k_max, eps_max = 2, maxiter, tol
    else:
        msg = f'unknown GMM method {method!r}'
        raise TypeError(msg)

    # random number generator
    rng = np.random.default_rng(random_state)

    # initial nonparametric weight matrix
    l_rr = l_moment_cov(x, n_con, trim=_trim, sort='stable')
    l_rr = (l_rr + l_rr.T) / 2  # enforce symmetry; workaround numerical errors
    w_rr = np.linalg.inv(l_rr)

    # initial state
    k = 0
    i = 0
    eps = np.inf
    stat = np.nan
    success = False
    qs = None

    for k in range(1, k_max + 1):
        # run the optimizer
        res = cast(
            OptimizeResult,
            optimize.minimize(  # type: ignore
                loss_fn,
                theta,
                args=(ppf, l_r, _trim, w_rr),
                method=optimizer,
                bounds=bounds,
            ),
        )
        if not res.success:
            break

        i += res.nfev
        stat = res.fun
        theta_k = res.x

        # check the exit/convergence criteria
        eps = np.max(np.abs(theta_k - theta))
        theta = theta_k
        if k >= k_min and (eps <= eps_max or k == k_max):
            success = eps <= eps_max
            break

        # re-evaluate the cov- and weight matrices with a Monte-Carlo approach
        qs = rng.uniform(0, 1, (n_obs, n_mc_samples)) if qs is None else qs
        ys = ppf(qs, *theta)
        l_rr = np.cov(l_moment(ys, r, trim=_trim, axis=0), ddof=n_par)
        w_rr = np.linalg.inv(l_rr)

    return GMMResult(
        args=tuple(theta),
        success=success,
        statistic=stat,
        eps=eps,
        n_samp=cast(int, n_obs - sum(_trim)),
        n_step=k,
        n_iter=i,
        weights=w_rr,
    )
