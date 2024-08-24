# Sample L-moments

Estimation of (trimmed) L-moments from sample data.
<!--
TODO: Maths
-->

## L-moment Estimators

Unbiased sample estimators of the L-moments and the (generalized) trimmed
L-moments.

::: lmo.l_moment
::: lmo.l_ratio
::: lmo.l_stats

### Shorthand aliases

Some of the commonly used L-moment and L-moment ratio's have specific names,
analogous to the named raw-, central-, and standard product-moments:

|   |   |   |
|---|---|---|
| $\lambda_r / \lambda_s$ | $s = 0$ | $r = 2$ |
| $r=1$ | $\lambda_1$ -- "L-loc[ation]" | $\tau$ -- "L-variation" or "L-CV" |
| $r=2$ | $\lambda_2$ -- "L-scale"      | ~~$1$ -- "L-one" or "L-unit"~~ |
| $r=3$ | $\lambda_3$                   | $\tau_3$ -- "L-skew[ness]" |
| $r=4$ | $\lambda_4$                   | $\tau_4$ -- "L-kurt[osis]" |

/// note
The "L-" prefix often refers to untrimmed L-moments, i.e. $(s, t) = (0, 0)$.

For some of the trimmed L-moments trim-lengths, specific alternative prefixes
are used:

|   |   |   |
|---|---|---|
| $\lambda_r^{(s, t)}$ | $t = 0$ | $t = 1$ |
| $s = 0$ | L-moment  | LL-moment |
| $s = 1$ | LH-moment | TL-moment |

The "L-" prefix refers to "Linear", i.e. an L-moment is a
"Linear combination of order statistics" [@hosking1990].
Usually "TL-moments" are used to describe symmetrically *T*rimmed L-moments,
in most cases those with a trim-length of 1 [@elamir2003, @hosking2007].
Similarly, "LH-moments" describe "linear combinations of order of *higher*-order
statistics" [@wang1997], and "LL-moments" that of "... the lowest order
statistics" [@bayazit2002].

Lmo supports all possible trim-lengths.
Technically, these are the "generalized trimmed L-moments".
But for the sake of brevity Lmo abbreviates this "L-moments".
///

::: lmo.l_loc
    options:
      heading_level: 4
::: lmo.l_scale
    options:
      heading_level: 4
::: lmo.l_variation
    options:
      heading_level: 4
::: lmo.l_skew
    options:
      heading_level: 4
::: lmo.l_kurt
    options:
      heading_level: 4
::: lmo.l_kurtosis
    options:
      heading_level: 4

## L-moment Accuracy

<!--
TODO: Short description
    (i.e. L-moment estimates are RV's with an approx normal distribution)
TODO: Example
TODO: Maths
-->

::: lmo.l_moment_cov
::: lmo.l_ratio_se
::: lmo.l_stats_se

## Sensitivity & Robustness

[Wikipedia](https://w.wiki/Azf$#Empirical_influence_function) describes the
*empirical influence function (EIF)* as follows:

> The empirical influence function is a measure of the dependence of the
> estimator on the value of any one of the points in the sample.
> It is a model-free measure in the sense that it simply relies on calculating
> the estimator again with a different sample.

<!-- markdownlint-disable MD052 -->
/// tip
The EIF can be used to calculate some useful
[properties](https://w.wiki/Azf$#Desirable_properties) related to the
robustness of the estimate.

- [`lmo.diagnostic.rejection_point`][lmo.diagnostic.rejection_point]
- [`lmo.diagnostic.error_sensitivity`][lmo.diagnostic.error_sensitivity]
- [`lmo.diagnostic.shift_sensitivity`][lmo.diagnostic.shift_sensitivity]

///

<!--
TODO: Explain that L-moments are *linear functionals*
TODO: Maths
-->

::: lmo.l_moment_influence
::: lmo.l_ratio_influence

## L-moment sample weights

<!--
TODO: Refer to the related `lmo.diagnostic` functions.
TODO: Maths
-->

::: lmo.l_weights

## References

\bibliography
