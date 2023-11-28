# L-moments of common probability distributions

This page lists theoretical L-moments of popular probability distributions.

All distributions are in the "standardized" form, similar to the convention
used in the `scipy.stats` distribution documentation.
Shifting a distribution only affects the L-location \( \tlmoment{s,t}{1} \),
just like the expectation and the median.
Scaling a distribution simply scales all L-moments
\( \tlmoment{s,t}{r}, \; r \ge 1 \) analogous to e.g. the
standard deviation or MAD.
Note that neither shifting nor scaling affects the L-moment ratio's
 \( \tlratio{s,t}{r} \).

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
    <th>Shape</th>
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
        \( [0, 1] \)
        <br>
        <code>uniform</code>
    </td>
    <td></td>
    <td>\[ \frac 1 2 \]</td>
    <td>\[ \frac 1 6 \]</td>
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
    <td></td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{1}{\sqrt \pi} \]
        \( \approx 0.5642 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 30 \ \frac{\theta_m}{\pi} - 9 \]
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
        <code>logistic</code>
    </td>
    <td></td>
    <td>\( 0 \)</td>
    <td>\( 1 \)</td>
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
    <td></td>
    <td>\( 0 \)</td>
    <td>\[ \frac 3 4 \]</td>
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
    <td></td>
    <td>\( 1 \)</td>
    <td>\[ \frac 1 2 \]</td>
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
            href="https://wikipedia.org/wiki/Half-normal_distribution"
            target="_blank"
            title="Half-normal distribution - Wikipedia"
        >
            Half-normal
        </a>
        <br>
        <code>halfnorm</code>
    </td>
    <td></td>
    <td>\( 1 \)</td>
    <td>
        \[ \sqrt 2 - 1 \]<br>
        \( \approx 0.4142 \)
    </td>
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
    <td></td>
    <td>
        \[ \frac 1 2 \sqrt{2 \pi} \]
        \( \approx 1.253 \)
    </td>
    <td>
        \[ \frac {\sqrt 2 - 1}{2} \sqrt{\pi} \]
        \( \approx 0.3671 \)
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
        see eq. \( \eqref{eq:lr_burr12} \) for \( \lmoment{r} \)
    </td>
    <td>\( \alpha > 0  \)</td>
    <td>
        \[ \frac{\alpha}{\alpha - 1} \]
    </td>
    <td>
        \[ \frac{\alpha}{\alpha - 1} \frac{1}{2 \alpha - 1} \]
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
        <th>Shape</th>
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
        \( [0, 1] \)
        <br>
        <code>uniform</code>
    </td>
    <td></td>
    <td>\[ \frac 1 2 \]</td>
    <td>\[ \frac{1}{10} \]</td>
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
    <td></td>
    <td>\( 0 \)</td>
    <td>
        \[
            \frac{6}{\sqrt \pi} \left(
                1
                - 3 \frac{\theta_m}{\pi}
            \right)
        \]
        \( \approx 0.2970 \)
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
    <td></td>
    <td>0</td>
    <td>\[ \frac 1 2 \]</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{1}{12} \]
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
    <td></td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{11}{32} \]
        \( = 0.34375 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{3}{22} \]
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
        <br>
        <code>cauchy</code>
        /
        <code>t(2)</code>
    </td>
    <td>\( \nu = 1 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ \frac{18 \vphantom{)}}{\pi^3 \vphantom{)}} \ \zeta(3) \]
        \( \approx 0.6978 \)
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
    <td></td>
    <td>\[ \frac 5 6 \]</td>
    <td>\[ \frac 1 4 \]</td>
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
        \[
            \frac{\sqrt \pi}{6}
            \bigl( 9 - 2 \sqrt 6 \bigr)
        \]
        \( \approx 1.211  \)
    </td>
    <td>
        \[
            \frac{\sqrt \pi}{4}
            \bigl( 6 - 4 \sqrt 6 + 3 \sqrt 2 \bigr)
        \]
        \( \approx 0.1970 \)
    </td>
    <td>
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

### Bernoulli

Surprisingly, the L-moments of the discrete
[Bernoulli distribution](https://wikipedia.org/wiki/Bernoulli_distribution),
can't be expressed as easily as the distribution itself:

\[
    \begin{equation}
    \tlmoment{s, t}{r} =
        \frac{(-1)^r}{r}
        (1 - p)^{s + 1}
        \jacobi{r + t - 1}{s + 1}{-t - 1}{2p - 1}
        + \ffact{1}{r}
    \label{eq:lr_bernoulli}
    \end{equation}
\]

Here, \( \jacobi{n}{\alpha}{\beta}{x} \) is a
[Jacobi polynomial](#def-jacobi) (although it's not orthogonal, since
\( \beta > -1 \) does not hold).


### Gompertz

The [Gompertz distribution](https://wikipedia.org/wiki/Gompertz_distribution)
with shape parameter \( \alpha > 0 \) and \( x \ge 0 \), has the following CDF
and PPF:


\[
    \begin{align*}
        F(x) &= 1 - e^{\alpha (1 - e^x)} \\
        x(F) &= \ln\left( 1 - \frac{\ln(1-F)}{\alpha} \right)
    \end{align*}
\]

The general trimmed L-moments of the Gompertz distribution are:

\[
    \begin{equation}
    \tlmoment{s, t}{r} =
        \frac{1}{r}
        \sum_{k = t + 1}^{r + s + t}
            (-1)^{k - t - 1}
            \binom{r + k - 2}{r + t - 1}
            \binom{r + s + t}{k}
            e^{\alpha k} \
            \Gamma(0,\ \alpha k)
    \label{eq:lr_gompertz}
    \end{equation}
\]

### GEV

The [*generalized extreme value* (GEV)
](https://wikipedia.org/wiki/Generalized_extreme_value_distribution)
distribution unifies the
[Gumbel](https://wikipedia.org/wiki/Gumbel_distribution),
[Fréchet](https://wikipedia.org/wiki/Fr%C3%A9chet_distribution),
and [Weibull](https://wikipedia.org/wiki/Weibull_distribution) distributions.
It has one shape parameter \( \alpha \in \mathbb{R} \), and the following
distribution functions:

\[
    \begin{align*}
        F(x) &= e^{-\qexp{1 - \alpha}{-x}} \\
        x(F) &= -\qlog{1 - \alpha}{-\ln(F)}
    \end{align*}
\]

Here, \( \qexp{q}{y} \) and \( \qlog{q}{y} \) are the
[Tsallis](https://wikipedia.org/wiki/Tsallis_statistics)
[\( q \)-exponential](#def-qexp) and the [\( q \)-logarithm](#def-qlog),
respectively.

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

Note that the GEV is effectively a reparametrized
\( q \)-[Gumbel](https://wikipedia.org/wiki/Gumbel_distribution)
[Tsallis distribution](https://wikipedia.org/wiki/Tsallis_distribution), with
\( q = 1 - \alpha \).

### GLO

The *generalized logistic distribution* (GLO), also known as the [shifted
log-logistic distribution
](https://wikipedia.org/wiki/Shifted_log-logistic_distribution), with shape
parameter \( \alpha \in \mathbb{R} \), is characterized by the following
distribution functions:

\[
    \begin{align*}
        F(x) &= \frac{1}{1 + \qexp{1 - \alpha}{x}} \\
        x(F) &= -\qlog{1 - \alpha}{\frac{1 - F}{F}}
    \end{align*}
\]

For \( -1 < \alpha < 1 \), the general trimmed L-moments of the GLO are:

\[
    \begin{equation}
        \tlmoment{s, t}{r} = \begin{cases}
            \displaystyle
                \psi(s + 1) - \psi(t + 1)
            & \text{if } \alpha = 0 \wedge r = 1 \\
            \displaystyle
                \frac{(-1)^r}{r} \B(r - 1,\ s + 1)
                + \frac 1 r \B(r - 1,\ t + 1)
            & \text{if } \alpha = 0 \\
            \displaystyle
                \frac{\ffact{1}{r}}{\alpha}
                + \sum_{k = s + 1}^{r + s + t}
                    (-1)^{r + s - k }
                    \binom{r + k - 2}{r + s - 1}
                    \binom{r + s + t}{k}
                    \B(\alpha,\ k - \alpha)
            & \text{if } -1 < \alpha < 1
        \end{cases}
    \label{eq:lr_glo}
    \end{equation}
\]

Where \( \psi(z) \) is the [digamma function](#def-digamma).

The corresponding `scipy.stats` implementation is
[`kappa4`][scipy.stats.kappa4], with `h = -1` and `k` set to \( \alpha \);
**not** [`genlogistic`][scipy.stats.genlogistic].

Note that the GLO is effectively a reparametrized
\( q \)-[logistic](https://wikipedia.org/wiki/Logistic_distribution)
[Tsallis distribution](https://wikipedia.org/wiki/Tsallis_distribution), with
\( q = 1 - \alpha \).

### GPD

The [*generalized Pareto distribution*
](https://wikipedia.org/wiki/Generalized_Pareto_distribution) (GPD), with
shape parameter \( \alpha \in \mathbb{R} \), has for \( x \ge 0 \) the
distribution functions:

\[
    \begin{align*}
        F(x) &= 1 - \qexp{1 + \alpha}{-x} \\
        x(F) &= -\qlog{1 + \alpha}{1 - F}
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
                & \text{if } \alpha = 0 \\
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

Note that the GPD is a reparametrized [\( q \)-exponential distribution
](https://wikipedia.org/wiki/Q-exponential_distribution), where
\( q = (2 \alpha + 1) / (\alpha + 1) \) and \( \lambda = 1 / (2 - q) \) s.t.
\( \alpha \neq -1 \) and \( q < 2 \).

### Burr III / Dagum

The *Burr III* distribution, also known as the
[*Dagum distribution*](https://wikipedia.org/wiki/Dagum_distribution), has two
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

The Burr III distribution is implemented in
[`scipy.stats.burr`][scipy.stats.burr], where the shape parameters `c` and `d`
correspond to  \( \alpha \) and \( \beta \), respectively.
Equivalently, [`scipy.stats.mielke`][scipy.stats.mielke] can be used, by
setting `k` and `s` to \( \alpha \beta \) and \( \alpha \), respectively.

The special case where \( \beta = 1 \) is known as the
[*log-logistic*](https://wikipedia.org/wiki/Log-logistic_distribution)
distribution

### Burr XII / Pareto IV

The
[*Burr XII distribution*](https://wikipedia.org/wiki/Burr_distribution)
has two shape parameters \( \alpha \) and \( \beta \), both restricted to the
positive reals. It is also known as the *Singh-Maddala distribution*.
The alternative parametrization \( \alpha \mapsto 1 / \gamma \), where
\( \gamma > 0 \), is known as the (standard) type IV
[*Pareto distribution*](https://wikipedia.org/wiki/Pareto_distribution)


The distribution functions for \( x > 0 \) are defined as:

\[
\begin{align*}
    F(x) &= 1 - (1 + x^\alpha)^{-\beta} \\
    x(F) &= \bigl((1 - F)^{-1/\beta} - 1 \bigr)^{1/\alpha}
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

This distribution is implemented in
[`scipy.stats.burr12`][scipy.stats.burr12], where the shape parameters `c`
and `d` correspond to  \( \alpha \) and \( \beta \), respectively.

The Burr XII and Burr III distributions are related as \( Y = 1 / X \), where
\( X \) and \( Y \) are random variables with Burr XII \( (\alpha, \beta) \)
and Burr III \( (1 / \alpha, \beta) \)
distributions (or vice-versa), respectively.

In the special case where \( \alpha = 1 \) is known as the
[*Lomax distribution*](https://wikipedia.org/wiki/Lomax_distribution). This
has been implemented as [scipy.stats.lomax][scipy.stats.lomax], where the
parameter `c` corresponds to \( \beta \).

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

### Wakeby

The [*Wakeby distribution*](https://wikipedia.org/wiki/Wakeby_distribution)
is quantile-based, without closed-form expressions for the PDF and CDF, whose
quantile function (PPF) is defined to be

\[
x(F) =
    \frac \alpha \beta \bigl(1 - (1 - F)^\beta\bigr)
    - \frac \gamma \delta \bigl(1 - (1 - F)^{-\delta}\bigr)
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
](https://wikipedia.org/wiki/Tukey_lambda_distribution) can be extended to
the *generalized lambda distribution*, which has two scale parameters
\( \alpha, \gamma \), and two shape parameters \( \beta, \delta \).

Like the Wakeby distribution, the generalized lambda has no closed-form PDF
or CDF. Instead, it is defined through its PPF:

\[
x(F) = \alpha \qlog{1 - \beta}{F} - \gamma \qlog{1 - \delta}{1 - F}
\]

Although its central product moments have no closed-form expression, when
\( \beta > -1 \) and \( \delta > -1 \), the general trimmed L-moments can be
compactly expressed as:

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


## Constants and special functions

An overview of the (non-obvious) mathematical notation of special functions
and constants.

<table markdown="span">
    <tr>
        <th>Name</th>
        <th>Notation</th>
        <th>Definition</th>
        <th>Python</th>
    </tr>
    <tr id="const-euler">
        <td>
            [Euler–Mascheroni constant
            ](https://wikipedia.org/wiki/Euler-Mascheroni_constant)
        </td>
        <td>\( \gamma_e \)</td>
        <td>
            $$
            \begin{align\*}
                &= \int_1^\infty
                    \left(
                        \frac{1}{\lfloor x \rfloor} - \frac 1 x
                    \right) \
                    \mathrm{d} x \\\\
                &= \lim_{x \to 0} \left( \frac 1 x - \Gamma(x) \right) \\\\
                &\approx 0.5772 \vphantom{\frac 1 1}
            \end{align\*}
            $$
        </td>
        <td>[`numpy.euler_gamma`][numpy.euler_gamma]</td>
    </tr>
    <tr id="const-theta_m">
        <td>
            [Magic angle](https://wikipedia.org/wiki/Magic_angle)
        </td>
        <td>\( \theta_m \)</td>
        <td>
            $$
            \begin{align\*}
                &= \arctan \left( \sqrt 2 \right) \\\\
                &= \arccos \left( 1 / \sqrt 3 \right)
            \end{align\*}
            $$
        </td>
        <td>[`lmo.constants.theta_m`][lmo.constants.theta_m]</td>
    </tr>
    <tr id="def-factorial" class="row-double-top">
        <td>
            [Factorial](https://wikipedia.org/wiki/Factorial)
        </td>
        <td>$$ n! \vphantom{\prod_{k=1}^n k} $$</td>
        <td>
            $$
            \begin{align\*}
                &= \prod_{k=1}^n k \\\\
                &= 1 \times 2 \times \ldots \times n
            \end{align\*}
            $$
        </td>
        <td>[`scipy.special.factorial`][scipy.special.factorial]</td>
    </tr>
    <tr id="def-falling">
        <td>
            [Falling factorial
            ](https://wikipedia.org/wiki/Falling_and_rising_factorials)
        </td>
        <td>\( \ffact{x}{n} \)</td>
        <td>
            $$
            \begin{align\*}
                &= \frac{\Gamma(x + 1)}{\Gamma(x - n + 1)} \\\\
                &= \prod_{k=0}^{n-1} (x - k)
                    \quad (\text{if } n \in \mathbb{n})
            \end{align\*}
            $$
        </td>
        <td>[`lmo.special.fpow`][lmo.special.fpow]</td>
    </tr>
    <tr id="def-rising">
        <td>
            [Rising factorial
            ](https://wikipedia.org/wiki/Falling_and_rising_factorials)
        </td>
        <td>\( \rfact{x}{n} \)</td>
        <td>
            $$
            \begin{align\*}
                &= \frac{\Gamma(x + n)}{\Gamma(x)} \\\\
                &= \prod_{k=0}^{n-1} (x + k)
                    \quad (\text{if } n \in \mathbb{n})
            \end{align\*}
            $$
        </td>
        <td>[`scipy.special.poch`][scipy.special.poch]</td>
    </tr>
    <tr id="def-binom">
        <td>
            [Binomial coefficient
            ](https://wikipedia.org/wiki/Binomial_coefficient)
        </td>
        <td>$$ \binom n k $$</td>
        <td>
            $$
            \begin{align\*}
                &= \frac{n!}{k! \ (n - k)!} \\\\
                &= \frac{\ffact{n}{k}}{k!}
            \end{align\*}
            $$
        </td>
        <td>[`scipy.special.comb`][scipy.special.comb]</td>
    </tr>
    <tr id="def-gamma" class="row-double-top">
        <td>
            [Gamma function](https://wikipedia.org/wiki/Gamma_function)
        </td>
        <td>\( \Gamma(z) \)</td>
        <td>\( = \int_0^\infty t^{z-1} e^{-t} \, \mathrm{d} t \)</td>
        <td>[`scipy.special.gamma`][scipy.special.gamma]</td>
    </tr>
    <tr id="def-gammainc" class="row-double-top">
        <td>
            [Incomplete Gamma function
            ](https://wikipedia.org/wiki/Incomplete_gamma_function)
        </td>
        <td>\( \Gamma(a,\ x) \)</td>
        <td>\( = \int_x^\infty t^{a - 1} e^{-t} \, \mathrm{d} t \)</td>
        <td>[`lmo.special.gamma2`][lmo.special.gamma2]</td>
    </tr>
    <tr id="def-digamma">
        <td>
            [Digamma function](https://wikipedia.org/wiki/Digamma_function)
        </td>
        <td>\( \psi(z) \)</td>
        <td>
            $$
            = \frac{\mathrm{d}}{\mathrm{d}z} \ln \Gamma(z)
            $$
        </td>
        <td>[`scipy.special.digamma`][scipy.special.digamma]</td>
    </tr>
    <tr id="def-beta">
        <td>
            [Beta function](https://wikipedia.org/wiki/Beta_function)
        </td>
        <td>\( \B(x,\ y) \)</td>
        <td>
            $$
            = \frac{\Gamma(x) \Gamma(y)}{\Gamma(x + y)}
            $$
        </td>
        <td>[`scipy.special.beta`][scipy.special.beta]</td>
    </tr>
    <tr id="def-zeta">
        <td>
            [Riemann zeta function
            ](https://wikipedia.org/wiki/Riemann_zeta_function)
        </td>
        <td>\( \zeta(z) \)</td>
        <td>$$ = \sum_{n = 1}^{\infty} n^{-z} $$</td>
        <td>[`scipy.special.zeta`][scipy.special.zeta]</td>
    </tr>
    <tr id="def-jacobi" class="row-double-top">
        <td>
            [Jacobi polynomial](https://wikipedia.org/wiki/Jacobi_polynomials)
        </td>
        <td>\( \jacobi{n}{\alpha}{\beta}{x} \)</td>
        <td>
            $$
            = \frac{1}{2^n} \sum_{k=0}^n
                \binom{n + \alpha}{k}
                \binom{n + \beta}{n - k}
                (x + 1)^{n + k}
                (x - 1)^{n - k}
            $$
        </td>
        <td>[`scipy.special.eval_jacobi`][scipy.special.eval_jacobi]</td>
    </tr>
    <tr id="def-qexp" class="row-double-top">
        <td>
            [*q*-exponential](https://wikipedia.org/wiki/Tsallis_statistics)
        </td>
        <td>\( \qexp{1 - q}{x} \)</td>
        <td>
            $$
            = \begin{cases}
                \displaystyle e^x
                    & \text{if } q = 0 \\\\
                \displaystyle (1 + q x)^{1 / q}
                    & \text{otherwise}
            \end{cases}
            $$
        </td>
        <td>[`scipy.special.inv_boxcox`][scipy.special.boxcox]</td>
    </tr>
    <tr id="def-qlog">
        <td>
            [*q*-logarithm](https://wikipedia.org/wiki/Tsallis_statistics)
        </td>
        <td>\( \qlog{1 - q}{y} \)</td>
        <td>
            $$
            = \begin{cases}
                \displaystyle \ln(y)
                    & \text{if } q = 0 \\\\
                \displaystyle \left( y^q - 1 \right) / q
                    & \text{otherwise}
            \end{cases}
            $$
        </td>
        <td>[`scipy.special.boxcox`][scipy.special.boxcox]</td>
    </tr>
</table>
