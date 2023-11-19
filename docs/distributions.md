# L-moments of common probability distributions

This page lists L-moment statistics
(L -location, scale, skewness, and kurtosis) of common univariate
probability distributions, most of them continuous.

Each of the listed expressions have been validated, both numerically and
symbolically (with either Wolfram Alpha, SymPy, or pen and paper).

Most of the closed-form expressions that are listed here, have been
previously reported in the literature. But for the sake of interpretability,
several have been algebraically rearranged.

Due to the exploratory use of symbolic computation software, this listing is
likely to include some novel solutions. This is also the reason for the lack
of references. But this should pose no problems in practise, since Lmo makes
it trivial to check if they aren't incorrect.


!!! tip

    Numerical calculation of these L-statistics using `scipy.stats`
    distributions, refer to
    [`rv_continuous.l_stats`][lmo.contrib.scipy_stats.l_rv_generic.l_stats].

    For direct calculation of the L-stats from a CDF or PPF (quantile function,
    inverse CDF), see [`l_stats_from_cdf`][lmo.theoretical.l_stats_from_cdf] or
    [`l_stats_from_ppf`][lmo.theoretical.l_stats_from_ppf], respectively.


## L-stats

An overview of the untrimmed L-location, L-scale, L-skewness and L-kurtosis,
of a bunch of popular univariate probability distributions, for which they
exist (in closed form).

<table>
<thead>
<tr>
    <th>Name /<br> <code>scipy.stats</code></th>
    <th>Params</th>
    <th>\( \lmoment{1} \)</th>
    <th>\( \lmoment{2} \)</th>
    <th>\( \lratio{3} = \lmoment{3}/\lmoment{2} \)</th>
    <th>\( \lratio{4} = \lmoment{4}/\lmoment{2} \)</th>
</tr>
</thead>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Continuous_uniform_distribution"
            target="_blank"
            title="Continuous uniform distribution - Wikipedia"
        >
            Uniform
        </a>
        <br>
        <code>uniform</code>
    </td>
    <td>\( a < b \)</td>
    <td>\[ \frac{a + b}{2} \]</td>
    <td>\[ \frac{b - a}{6} \]</td>
    <td>\( 0 \)</td>
    <td>\( 0 \)</td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Normal_distribution"
            target="_blank"
            title="Normal distribution - Wikipedia"
        >
            Normal
        </a>
        <br>
        <code>norm</code>
    </td>
    <td>
        \( \mu \)<br>
        \( \sigma>0 \)
    </td>
    <td>\( \mu \)</td>
    <td>
        \[ \frac{\sigma}{\sqrt \pi} \]
        \( \approx 0.5642 \ \sigma \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 30 \ \frac{\arctan{\sqrt 2}}{\pi} - 9 \]
        \( \approx 0.1226 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Logistic_distribution"
            target="_blank"
            title="Logistic distribution - Wikipedia"
        >
            Logistic
        </a>
        <br>
        <code>logistic(μ, s)</code>
    </td>
    <td>
        \( \mu \)<br>
        \( s>0 \)
    </td>
    <td>\( \mu \)</td>
    <td>\( s \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ 1 / 6 \]
        \( = 0.16\overline{6}\dots \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Laplace_distribution"
            target="_blank"
            title="Laplace distribution - Wikipedia"
        >
            Laplace
        </a>
        <br>
        <code>laplace</code>
    </td>
    <td>
        \( \mu \)<br>
        \( b > 0 \)
    </td>
    <td>\( \mu \)</td>
    <td>
        \[ \frac 3 4 b \]
        \( = 0.75 \ b\)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{17}{72} \]
        \( \approx 0.2361 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (2 d.f.)
        <br>
        <code>t(2)</code>
    </td>
    <td>\( \nu = 2 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{\pi}{2 \sqrt{2}} \]
        \( \approx 1.1107 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac 3 8 \]
        \( = 0.375 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (3 d.f.)
        <br>
        <code>t(3)</code>
    </td>
    <td>\( \nu = 3 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{3 \sqrt 3}{\vphantom{\pi^2}2 \pi} \]
        \( \approx 0.8270 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 1 - \frac{\vphantom{\sqrt 3}175}{24 \pi^2} \]
        \( \approx 0.2612 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (4 d.f.)
        <br>
        <code>t(4)</code>
    </td>
    <td>\( \nu = 4 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{15}{64} \pi \]
        \( \approx 0.7363 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{111}{512} \]
        \( \approx 0.2168 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Exponential_distribution"
            target="_blank"
            title="Exponential distribution - Wikipedia"
        >
            Exponential
        </a>
        <br>
        <code>expon</code>
    </td>
    <td>\( \lambda>0 \)</td>
    <td>\[ \frac 1 \lambda \]</td>
    <td>\[ \frac{1}{2 \lambda} \]</td>
    <td>
        \[ \frac 1 3 \]
        \( = 0.3\overline{3}\dots \)
    </td>
    <td>
        \[ \frac 1 6 \]
        \( = 0.16\overline{6}\dots \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Rayleigh_distribution"
            target="_blank"
            title="Rayleigh distribution - Wikipedia"
        >
            Rayleigh
        </a>
        <br>
        <code>rayleigh</code>
    </td>
    <td>\( \sigma > 0 \)</td>
    <td>
        \[
            \frac 1 2
            \sqrt{2 \pi} \
            \sigma
        \]
        \( \approx 1.253 \ \sigma \)
    </td>
    <td>
        \[
            \frac {\sqrt 2 - 1}{2}
            \sqrt{\pi} \
            \sigma
        \]
        \( \approx 0.3671 \ \sigma \)
    </td>
    <td>
        \[ 2 \frac{2 + \sqrt 2}{\sqrt 3} - \frac{4 + \sqrt{2}}{\sqrt 2} \]
        \( \approx 0.1140 \)
    </td>
    <td>
        \[ 10 \frac{2 + \sqrt 2}{\sqrt 3} - 3 \frac{5 + 3 \sqrt 2}{\sqrt 2} \]
        \( \approx 0.1054 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Gumbel_distribution"
            target="_blank"
            title="Gumbel distribution - Wikipedia"
        >
            Gumbel
        </a>
        <br>
        <code>gumbel_r</code>
        <br>
        see eq. \( \eqref{eq:lr_gev} \) for \( \lmoment{r} \)
    </td>
    <td></td>
    <td>
        \[ \gamma_e \]
        \( \approx 0.5772 \)
    </td>
    <td>
        \[ \ln{2} \]
        \( \approx 0.6931 \)
    </td>
    <td>
        \[ 2 \log_2(3) - 3 \]
        \( \approx 0.1699 \)
    </td>
    <td>
        \[ 16 - 10 \log_2(3) \]
        \( \approx 0.1504 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Pareto_distribution"
            target="_blank"
            title="Pareto distribution - Wikipedia"
        >
            Pareto
        </a>
        <br>
        <code>pareto</code>
        <br>
        see eq. \( \eqref{eq:lr_pareto4} \) for \( \lmoment{r} \)
    </td>
    <td>
        \[
        \begin{align*}
            \alpha &> 0 \quad \text{(shape)} \\
            \beta  &> 0 \quad \text{(scale)}
        \end{align*}
        \]
    </td>
    <td>
        \[ \frac{\alpha}{\alpha - 1} \ \beta \]
    </td>
    <td>
        \[ \frac{\alpha}{\alpha - 1} \frac{1}{2 \alpha - 1} \ \beta \]
    </td>
    <td>
        \[ \frac{\alpha + 1}{3 \alpha - 1} \]
    </td>
    <td>
        \[ \frac{\alpha + 1}{3 \alpha - 1} \frac{2 \alpha + 1}{4 \alpha - 1} \]
    </td>
</tr>
</table>

## TL-stats

Collection of TL-location, -scale, -skewness, -kurtosis coefficients, with
symmetric trimming of order 1, i.e. `trim=(1, 1)`.

<table>
<thead>
    <tr>
        <th>Name / <br><code>scipy.stats</code></th>
        <th>Params</th>
        <th>\( \tlmoment{1}{1} \)</th>
        <th>\( \tlmoment{1}{2} \)</th>
        <th>\( \tlratio{1}{3} \)</th>
        <th>\( \tlratio{1}{4} \)</th>
    </tr>
</thead>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Continuous_uniform_distribution"
            target="_blank"
            title="Continuous uniform distribution - Wikipedia"
        >
            Uniform
        </a>
        <br>
        <code>uniform</code>
    </td>
    <td>\( a < b \)</td>
    <td>\[ (a + b) / 2 \]</td>
    <td>\[ (a - b) / 10 \]</td>
    <td>\( 0 \)</td>
    <td>\( 0 \)</td>
    <td></td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Normal_distribution"
            target="_blank"
            title="Normal distribution - Wikipedia"
        >
            Normal
        </a>
        <br>
        <code>norm</code>
    </td>
    <td>
        \( \mu \)<br>
        \( \sigma>0 \)
    </td>
    <td>\( \mu \)</td>
    <td>
        \[
            \left( 6 - 18 \ \frac{\arctan{\sqrt 2}}{\pi} \right)
            \frac{\sigma}{\sqrt \pi}
        \]
        \( \approx 0.2970 \ \sigma \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \( \approx 0.06248 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Logistic_distribution"
            target="_blank"
            title="Logistic distribution - Wikipedia"
        >
            Logistic
        </a>
        <br>
        <code>logistic(μ, s)</code>
    </td>
    <td>\( \mu \)<br>\( s>0 \)</td>
    <td>\( \mu \)</td>
    <td>\( s / 2 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ 1 / 12 \]
        \( = 0.083\overline{3} \dots \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Laplace_distribution"
            target="_blank"
            title="Laplace distribution - Wikipedia"
        >
            Laplace
        </a>
        <br>
        <code>laplace</code>
    </td>
    <td>
        \( \mu \)<br>
        \( b > 0 \)
    </td>
    <td>\( \mu \)</td>
    <td>
        \[ 11b / 32 \]
        \( = 0.34375 \ b\)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 3 / 22 \]
        \( = 0.136\overline{36} \dots \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://en.wikipedia.org/wiki/Cauchy_distribution"
            target="_blank"
            title="Cauchy distribution - Wikipedia"
        >
            Cauchy
        </a>
        /
        <br>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (1 d.f.)
        <br>
        <code>cauchy</code>
        /
        <code>t(2)</code>
    </td>
    <td>\( \nu = 1 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{18 \vphantom{)}}{\pi^3 \vphantom{)}} \ \zeta(3) \]
        \( \approx 0.6978 \ b \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{25}{6} - \frac{175}{4 \pi^2} \frac{\zeta(5)}{\zeta(3)} \]
        \( \approx 0.3428 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (2 d.f.)
        <br>
        <code>t(2)</code>
    </td>
    <td>\( \nu = 2 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{3 \pi}{16 \sqrt{2}} \]
        \( \approx 0.4165 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{5}{32} \]
        \( = 0.15625 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (3 d.f.)
        <br>
        <code>t(3)</code>
    </td>
    <td>\( \nu = 3 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{105 \sqrt 3}{16 \pi^3} \]
        \( \approx 0.3666 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{25}{6} -  \frac{23 \ 023}{(24 \pi)^2} \]
        \( \approx 0.1168 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Student%27s_t-distribution"
            target="_blank"
            title="Student's t-distribution - Wikipedia"
        >
            Student's <i>t</i>
        </a>
        (4 d.f.)
        <br>
        <code>t(4)</code>
    </td>
    <td>\( \nu = 4 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{3 \ 609 \ \pi}{32 \ 768} \]
        \( \approx 0.3460 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{164 \ 975}{1 \ 642 \ 496} \]
        \( \approx 0.1004 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Exponential_distribution"
            target="_blank"
            title="Exponential distribution - Wikipedia"
        >
            Exponential
        </a>
        <br>
        <code>expon</code>
    </td>
    <td>\( \lambda>0 \)</td>
    <td>\[ \frac{5}{6 \lambda} \]</td>
    <td>\[ \frac{1}{4 \lambda} \]</td>
    <td>
        \[ \frac 2 9 \]
        \( = 0.2\overline{2}\dots \)
    </td>
    <td>
        \[ \frac{1}{12} \]
        \( = 0.083\overline{3}\dots \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Rayleigh_distribution"
            target="_blank"
            title="Rayleigh distribution - Wikipedia"
        >
            Rayleigh
        </a>
        <br>
        <code>rayleigh</code>
    </td>
    <td></td>
    <td>
        <!-- \[ \left( \frac 3 2 - \sqrt{\frac{2}{3}} \right) \sqrt \pi \ \sigma \] -->
        \[
            \frac 1 6
            \bigl( 9 - 2 \sqrt 6 \bigr)
            \sqrt \pi \
        \]
        \( \approx 1.211  \)
    </td>
    <td>
        \[
            \frac 1 4
            \bigl( 6 - 4 \sqrt 6 + 3 \sqrt 2 \bigr)
            \sqrt \pi \
        \]
        \( \approx 0.1970 \)
    </td>
    <td>
        <!-- \[
            \frac 2 9
            \frac
                {30 - 12 \sqrt{10} - 40 \sqrt 6 + 75 \sqrt 2}
                {6 - 4 \sqrt 6 + 3 \sqrt 2}
        \] -->
        \[
            \frac{10}{9}
            - \frac{8}{9}
            \frac
                {3 \sqrt{10} + 5 \sqrt 6 - 15 \sqrt 2}
                {6 - 4 \sqrt 6 + 3 \sqrt 2}
        \]
        \( \approx 0.06951 \)
    </td>
    <td>
        \[
            \frac 5 4
            - \frac 7 6
            \frac
                {18 \sqrt{10} + 10 \sqrt 6 - 10 \sqrt 3 - 45 \sqrt 2}
                {6 - 4 \sqrt 6 + 3 \sqrt 2}
        \]
        \( \approx 0.05422 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Gumbel_distribution"
            target="_blank"
            title="Gumbel distribution - Wikipedia"
        >
            Gumbel
        </a>
        <br>
        <code>gumbel_r</code>
        <br>
        see eq. \( \eqref{eq:lr_gev} \) for \( \tlmoment{s,t}{r} \)
    </td>
    <td></td>
    <td>
        \[ \gamma_e + 3 \ln{2} - 2 \ln{3} \]
        \( \approx 0.4594 \)
    </td>
    <td>
        \[ 6 \ln{3} - 9 \ln{2} \]
        \( \approx 0.3533 \)
    </td>
    <td>
        \[
            -\frac{10}{9}
            \frac
                {2 \log_2(5) - 5}
                {2 \log_2(3) - 3}
            -
            \frac{20}{9}
        \]
        \( \approx 0.1065 \)
    </td>
    <td>
        \[
            \frac{35}{6}
            \frac
                {3 \log_2(5) - 7}
                {2 \log_2(3) - 3}
            +
            \frac{5}{4}
        \]
        \( \approx 0.07541 \)
    </td>
</tr>
</table>

## General distribution L-moments

Lmo derived a bunch of closed-form solutions for L-moments of several
distributions. The proofs are not published, but it isn't difficult
to validate their correctness, e.g. numerically, or symbolically with sympy or
wolfram alpha / mathematica.

### GEV

The [generalized extreme value (GEV)
](https://wikipedia.org/wiki/Generalized_extreme_value_distribution)
distribution unifies the
[Gumbel](https://wikipedia.org/wiki/Gumbel_distribution),
[Fréchet](https://wikipedia.org/wiki/Fr%C3%A9chet_distribution),
and [Weibull](https://wikipedia.org/wiki/Weibull_distribution) distributions.
It has one shape parameter \( \alpha \in \mathbb{R} \), and the following
distribution functions:

\[
    \begin{align*}
        F(x) &= e^{-\coxbox{-x}{\alpha}} \\
        x(F) &= -\boxcox{-\ln(F)}{\alpha}
    \end{align*}
\]

Here, \( \boxcox{\cdot}{\alpha} \) is the
[Box-Cox transformation function](#def-boxcox).

An alternative parametrization is sometimes used, e.g. on
[Wikipedia](https://wikipedia.org/wiki/Generalized_extreme_value_distribution),
where \( \xi = -\alpha \).
The convention that is used here, is the same as in
[`scipy.stats.genextreme`][scipy.stats.genextreme], where `c` corresponds to
\( \alpha \).

The trimmed L-moments of the GEV are

\[
    \begin{equation}
    \tlmoment{s, t}{r} =
        \frac{(-1)^{r}}{r}
        \sum_{k = s + 1}^{r + s + t}
            (-1)^{k - s}
            \binom{r + k - 2}{r + s - 1}
            \binom{r + s + t}{k}
            \left(
            \begin{cases}
                \gamma_e + \ln(k)
                    & \text{if } \alpha = 0 \\
                1 / \alpha - \Gamma(\alpha) \ k^{-\alpha}
                    & \text{if } \alpha \neq 0
            \end{cases}
            \right)
    \label{eq:lr_gev}
    \end{equation}
\]

### GPD

The [generalized Pareto distribution
](https://wikipedia.org/wiki/Generalized_Pareto_distribution) (GPD), with
shape parameter \( \alpha \in \mathbb{R} \), has for \( x \ge 0 \) the
distribution functions:

\[
    \begin{align*}
        F(x) &= 1 - 1 / \coxbox{x}{\alpha} \\
        x(F) &= \boxcox{1 / (1 - F)}{\alpha}
    \end{align*}
\]

Note that this distribution is standard uniform if \( \alpha = 1 \), and
standard exponential if \( \alpha = 0 \).

The general trimmed L-moments of the GPD are:

\[
    \begin{equation}
        \tlmoment{s,t}{r} = \begin{cases}
            \displaystyle \sum_{k = 1}^{s + 1} \frac{1}{t + k}
                & \text{if } \alpha = 0 \wedge r = 1 \\
            \frac{1}{r} \B(r - 1,\ t + 1)
                & \text{if } \alpha = 0 \wedge r > 1 \\
            \displaystyle \frac{r + s + t}{\alpha \ r} \sum_{k = 0}^{r + t - 1}
                \frac{(-1)^{r - k}}{k}
                \binom{r + s + t - 1}{k + s}
                \binom{r + s + k - 1}{k}
                \left(
                    1 - \frac{(k + 1)!}{\rfact{1 - \alpha}{k + 1}}
                \right)
                & \text{if } \alpha < 1
        \end{cases}
        \label{eq:lr_gpd}
    \end{equation}
\]

Apparently left-trimming the exponential distribution does not influence any
of the L-moments, besides the L-location.

For the general LH-moments, this simplifies to:

\[
    \begin{equation}
        \tlmoment{0,t}{r} = \begin{cases}
             \displaystyle \frac{1}{t + 1}
                & \text{if } \alpha = 0 \wedge r = 1 \\
            \frac{1}{r} \B(r - 1,\ t + 1)
                & \text{if } \alpha = 0 \wedge r > 1 \\
            \displaystyle \frac{r + t}{r}
                \frac{\rfact{1 + \alpha}{r - 2}}{\rfact{1 - \alpha + t}{r}}
                - \frac{\ffact{1}{r}}{\alpha}
                & \text{if } \alpha < 1
        \end{cases}
        \label{eq:lhr_gpd}
    \end{equation}
\]

See [`scipy.stats.genpareto`][scipy.stats.genpareto] for the implementation of
the GPD.

### Pareto Type IV

The [Pareto Type IV](https://wikipedia.org/wiki/Pareto_distribution) has two
shape parameters \( \alpha \in \mathbb{R} \) and
\( \gamma \in \mathbb{R}_{>0} \), and scale parameter \( \beta \).
For \( x \ge 0 \), the CDF and its inverse (the PPF) are

\[
\begin{align*}
    F(x)
        &= 1 - \left(
            1 + \left(\frac x \beta\right)^{\frac 1 \gamma}
        \right)^{-\alpha} \\
    x(F)
        &= \beta \left(
            (1 - F)^{-1 / \alpha} - 1
        \right)^\gamma
\end{align*}
\]

When \( \alpha > \gamma \), the trimmed L-moments are found to be:

\[
    \begin{equation}
        \tlmoment{s,t}{r}
            = \frac{\beta \gamma}{r}
            \sum_{k = t + 1}^{r + s + t}
                (-1)^{k - t - 1}
                \binom{r + k - 2}{r + t - 1}
                \binom{r + s + t}{k}
                \B(\gamma,\ k \alpha - \gamma)
            \label{eq:lr_pareto4}
    \end{equation}
\]

This distribution is currently not implemented in [`scipy.stats`][scipy.stats].

### Kumaraswamy

For [Kumaraswamy's distribution
](https://wikipedia.org/wiki/Kumaraswamy_distribution) with parameters
\( \alpha \in \mathbb{R}_{>0} \) and \( \beta \in \mathbb{R}_{>0} \),
the general solution for the \( r \)th L-moment has been derived by
[Jones (2009)](https://doi.org/10.1016/j.stamet.2008.04.001). This can be
extended for the general trimmed L-moments.

The distribution functions are for \( 0 \le x \le 1 \) defined as:

\[
\begin{align*}
F(x) &= 1 - (1 - x^\alpha)^\beta \\
x(F) &= \bigl(1 - (1 - F)^{1/\beta} \bigr)^{1/\alpha}
\end{align*}
\]

Its general \( r \)-th trimmed L-moment are:

\[
    \begin{equation}
        \tlmoment{s,t}{r} =
            \beta \
            \frac{r + s + t}{r}
            \sum_{k = t}^{r + s + t - 1}
                (-1)^k
                \binom{k + r - 1}{k - t}
                \binom{r + s + t - 1}{k}
                \B\bigl(1 + 1 / \alpha,\ \beta + k \beta \bigr)
            \label{eq:lr_kum}
    \end{equation}
\]

Unfortunately, the Kumaraswamy distribution is not implemented in
`scipy.stats`.

### Burr Type III / Dagum

The Burr type III distribution, also known as the
[Dagum distribution](https://wikipedia.org/wiki/Dagum_distribution), has two
shape parameters \( \alpha \) and \( \beta \), both restricted to the
positive reals

For \( x > 0 \), the distribution functions are:

\[
\begin{align*}
    F(x) &=
        (1 + x^{-\alpha})^{-\beta} \\
    x(F) &=
        (F^{-1 / \beta} - 1)^{-1 / \alpha}
\end{align*}
\]

For \( \alpha > 1 \), the general L-moments are:

\[
\begin{equation}
    \tlmoment{s,t}{r} =
        (-1)^{t - 1 / \alpha} \
        \beta \
        \frac{r + s + t}{r}
        \sum_{k = s}^{r + s + t - 1}
            (-1)^{k}
            \binom{k + r - 1}{k - s}
            \binom{r + s + t - 1}{k}
            \B(1 - 1 / \alpha, -\beta - k \beta)
    \label{eq:lr_burr3}
\end{equation}
\]

The Burr Type III distribution is implemented in
[`scipy.stats.burr`][scipy.stats.burr], where the shape parameters `c` and `d`
correspond to  \( \alpha \) and \( \beta \), respectively.

### Burr Type XII

Just like Kumaraswamy's distribution, the
[Burr (Type XII) distribution](https://wikipedia.org/wiki/Burr_distribution)
has two shape parameters \( \alpha \) and \( \beta \), both restricted to the
positive reals.

The distribution functions are for \( x > 0 \) defined as:

\[
\begin{align*}
    F(x) &= 1 - (1 - x^\alpha)^{-\beta} \\
    x(F) &= \bigl(1 - (1 - F)^{-1/\beta} \bigr)^{1/\alpha}
\end{align*}
\]

When \( \beta > 1 / \alpha \), the general \( r \)-th trimmed L-moment is:

\[
\begin{equation}
    \tlmoment{s,t}{r} =
        \beta \
        \frac{r + s + t}{r}
        \sum_{k = t}^{r + s + t - 1}
            (-1)^k
            \binom{k + r - 1}{k - t}
            \binom{r + s + t - 1}{k}
            \B\bigl(1 + 1 / \alpha,\ \beta + k \beta - 1 / \alpha \bigr)
        \label{eq:lr_burr12}
\end{equation}
\]

The Burr Type XII distribution is implemented in
[`scipy.stats.burr12`][scipy.stats.burr12], where the shape parameters `c`
and `d` correspond to  \( \alpha \) and \( \beta \), respectively.

### Wakeby

The [Wakeby distribution](https://wikipedia.org/wiki/Wakeby_distribution)
is quantile-based, without closed-form expressions for the PDF and CDF, whose
quantile function (PPF) is defined to be

\[
x(F) =
    \frac \alpha \beta \bigl(1 - (1 - F)^\beta\bigr)
    - \frac \gamma \delta \bigl(1 - (1 - F)^{-\delta}\bigr)
    + \mu
\]

Each of the scale- \( \alpha, \gamma \) and shape parameters
\( \beta, \delta \), are assumed to be positive real numbers.

Lmo figured out that the L-moments with any order \( r \in \mathbb{N}_{\ge 1} \)
and trim \( s, t \in \mathbb{N}^2_{\ge 1} \) can be expressed as

<!-- \[
\begin{equation}
    \tlmoment{s,t}{r}
        =
        \frac{\gamma}{r \delta}
        \frac
            {\B(\delta + r - 1,\ t - \delta + 1)}
            {\B(\delta,\ r + s + t - \delta + 1)}
        -
        \frac{\alpha}{r \beta}
        \frac
            {\B(-\beta + r - 1,\ t + \beta + 1)}
            {\B(-\beta,\ r + s + t + \beta + 1)}
    +
    \begin{cases}
         \mu + \frac \alpha \beta - \frac \gamma \delta
            & \text{if } r = 1 \\
        0
            & \text{if } r > 1
    \end{cases}
\end{equation}
\] -->
\[
\begin{equation}
    \tlmoment{s,t}{r}
        = \frac{\rfact{r + t}{s + 1}}{r} \left[
            \alpha \frac
                {\rfact{1 - \beta}{r - 2}}
                {\rfact{1 + \beta + t}{r + s}}
            + \gamma \frac
                {\rfact{1 + \delta}{r - 2}}
                {\rfact{1 - \delta + t}{r + s}}
        \right]
        + \underbrace{
            \ffact{1}{r} \left(
                \frac \alpha \beta - \frac \gamma \delta
            \right)
        }_{\text{will be } 0 \text{ if } r>1}
\end{equation}
\]

Unfortunately, the Wakeby distribution has currently no
[`scipy.stats`][scipy.stats] implementation.

### Generalized Lambda

The [Tukey lambda distribution
](https://wikipedia.org/wiki/Tukey_lambda_distribution) can be generalized
to two scale parameters \( \alpha, \gamma \), and two shape parameters
\( \beta, \delta \).

Like the Wakeby distribution, the generalized lambda has no closed-form PDF
or CDF. Instead, it is defined through its PPF:

\[
x(F)
    = \alpha \boxcox{F}{\beta}
    - \gamma \boxcox{-F}{\delta}
\]

Although its central product moments have no closed-form expression, the
general trimmed L-moments can be compactly expressed as:

\[
\begin{equation}
    \tlmoment{s,t}{r}
        = \alpha
        \frac
            {\rfact{r + s}{t + 1} \ \ffact{\beta + s}{r + s - 1}}
            {r \ \rfact{\beta}{r + s + t + 1}}
        + (-1)^r \gamma \
        \frac
            {\rfact{r + t}{s + t} \ \ffact{\delta + t}{r + t - 1}}
            {r \ \rfact{\delta}{r + s + t + 1}}
        - \underbrace{
            \ffact{1}{r} \left(
                \frac \alpha \beta - \frac \gamma \delta
            \right)
        }_{\text{will be } 0 \text{ if } r>1}
\end{equation}
\]

When \( \alpha = \gamma \) and \( \beta = \delta \), this is the
(non-generalized) Tukey-lambda distribution, which has been implemented as
[`scipy.stats.tukeylambda`][scipy.stats.tukeylambda]. Currently, this
4-parameter generalization has no [`scipy.stats`][scipy.stats] implementation.

<!-- TODO: Generalized Pareto (GPD / Pareto-Pickands) -->
<!-- TODO: Generalized Logistic -->

## Constants and special functions

An overview of the (non-obvious) mathematical notation of special functions
and constants.

<table>
<thead>
    <tr>
        <th>Name</th>
        <th>Notation</th>
        <th>Definition</th>
        <th>Python</th>
    </tr>
</thead>
<tr id="const-euler">
    <td>
        <a
            href="https://wikipedia.org/wiki/Euler-Mascheroni_constant"
            target="_blank"
            title="Euler's constant"
        >
            Euler–Mascheroni constant
        </a>
    </td>
    <td>\[ \gamma_e \]</td>
    <td>
        \[
            \begin{align*}
                &= \int_1^\infty
                    \left(
                        \frac{1}{\lfloor x \rfloor} - \frac 1 x
                    \right) \
                    \mathrm{d} x \\
                &= \lim_{x \to 0} \left( \frac 1 x - \Gamma(x) \right) \\
                &\approx 0.5772 \vphantom{\frac 1 1}
            \end{align*}
        \]
    </td>
    <td>
        <a
            href="https://numpy.org/doc/stable/reference/constants.html#numpy.euler_gamma"
            target="_blank"
        >
            <code>numpy.euler_gamma</code>
        </a>
    </td>
</tr>

<tr id="def-factorial" class="row-double-top">
    <td>
        <a
            href="https://wikipedia.org/wiki/Factorial"
            target="_blank"
            title="Factorial - Wikipedia"
        >
            Factorial
        </a>
    </td>
    <td>\[ n! \vphantom{\prod_{k=1}^n k} \]</td>
    <td>
        \[
            \begin{align*}
                &= \prod_{k=1}^n k \\
                &= \underbrace
                    {1 \times 2 \times \ldots \times n}
                    _{n{\text{ factors}}}
            \end{align*}
        \]
    </td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.factorial.html"
            target="_blank"
        >
            <code>scipy.special.factorial</code>
        </a>
    </td>
</tr>
<tr id="def-falling">
    <td>
        <a
            href="https://wikipedia.org/wiki/Falling_and_rising_factorials"
            target="_blank"
            title="Falling and rising factorials - Wikipedia"
        >
            Falling factorial
        </a>
    </td>
    <!-- <td>\[ x^{\underline{n}} \]</td> -->
    <td>\[ \ffact{x}{n} \]</td>
    <td>
        \[
            \begin{align*}
                &= \prod_{k=0}^{n-1} (x - k) \\
                &= \underbrace
                    {x \ (x - 1) \ldots (x - n + 1)}
                    _{n{\text{ factors}}}
            \end{align*}
        \]
    </td>
    <td></td>
</tr>
<tr id="def-rising">
    <td>
        <a
            href="https://wikipedia.org/wiki/Falling_and_rising_factorials"
            target="_blank"
            title="Falling and rising factorials - Wikipedia"
        >
            Rising factorial
        </a>
    </td>
    <!-- <td>\[ x^{\overline{n}} \]</td> -->
    <td>\[ \rfact{x}{n} \]</td>
    <td>
        \[
            \begin{align*}
                &= \prod_{k=0}^{n-1} (x + k) \\
                &= \underbrace
                    {x \ (x + 1) \ldots (x + n - 1)}
                    _{n{\text{ factors}}}
            \end{align*}
        \]
    </td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.poch.html"
            target="_blank"
        >
            <code>scipy.special.poch</code>
        </a>
    </td>
</tr>
<tr id="def-binom">
    <td>
        <a
            href="https://wikipedia.org/wiki/Binomial_coefficient"
            target="_blank"
            title="Binomial coefficient - Wikipedia"
        >
            Binomial coefficient
        </a>
    </td>
    <td>\[ \binom{n}{k} \]</td>
    <td>
        \[
            \begin{align*}
                &= \frac{n!}{k! \ (n - k)!} \\
                &= \frac{\ffact{n}{k}}{k!}
            \end{align*}
        \]
    </td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html"
            target="_blank"
        >
            <code>scipy.special.comb</code>
        </a>
    </td>
</tr>

<tr id="def-gamma" class="row-double-top">
    <!-- <td style="border-top-style: double;"> -->
    <td>
        <a
            href="https://wikipedia.org/wiki/Gamma_function"
            target="_blank"
            title="Gamma function - Wikipedia"
        >
            Gamma function
        </a>
    </td>
    <td>\[ \Gamma(z) \]</td>
    <td>\[ = \int_0^\infty t^{z-1} e^{-t} \, \mathrm{d} t \]</td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html"
            target="_blank"
        >
            <code>scipy.special.gamma</code>
        </a>
    </td>
</tr>
<tr id="def-beta">
    <td>
        <a
            href="https://wikipedia.org/wiki/Beta_function"
            target="_blank"
            title="Beta function - Wikipedia"
        >
            Beta function
        </a>
    </td>
    <td>\[ \B(z_1,\ z_2) \]</td>
    <td>\[ = \frac{\Gamma(z_1) \Gamma(z_2)}{\Gamma(z_1 + z_2)} \]</td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beta.html"
            target="_blank"
        >
            <code>scipy.special.beta</code>
        </a>
    </td>
</tr>
<tr id="def-zeta">
    <td>
        <a
            href="https://wikipedia.org/wiki/Riemann_zeta_function"
            target="_blank"
            title="Riemann zeta function - Wikipedia"
        >
            Riemann zeta function
        </a>
    </td>
    <td>\[ \zeta(z) \vphantom{\sum_{n = 1}^{\infty}} \]</td>
    <td>\[ = \sum_{n = 1}^{\infty} n^{-z} \]</td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html"
            target="_blank"
        >
            <code>scipy.special.zeta</code>
        </a>
    </td>
</tr>
<tr id="def-boxcox" class="row-double-top">
    <td>
        <a
            href="https://wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation"
            target="_blank"
            title="Box–Cox transformation - Power transform - Wikipedia"
        >
            Box–Cox transform
        </a>
    </td>
    <td>\[ \boxcox{y}{\lambda} \]</td>
    <td>
        \[
            = \begin{cases}
                (y^\lambda - 1) / \lambda & \text{if } \lambda \neq 0 \\
                \ln(y) & \text{if } \lambda = 0
            \end{cases}
        \]
    </td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html"
            target="_blank"
        >
            <code>scipy.special.boxcox</code>
        </a>
    </td>
</tr>
<tr id="def-coxbox">
    <td>
        <a
            href="https://wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation"
            target="_blank"
            title="Box–Cox transformation - Power transform - Wikipedia"
        >
            Inverse Box–Cox transform
        </a>
    </td>
    <td>\[ \coxbox{x}{\lambda} \]</td>
    <td>
        \[
            = \begin{cases}
                (\lambda y + 1)^{1 / \lambda} & \text{if } \lambda \neq 0 \\
                e^{x} & \text{if } \lambda = 0
            \end{cases}
        \]
    </td>
    <td>
        <a
            href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html"
            target="_blank"
        >
            <code>scipy.special.inv_boxcox</code>
        </a>
    </td>
</tr>
</table>
