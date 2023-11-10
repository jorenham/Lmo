# L-moments of common probability distributions

## Continuous univariate

### L-moments

<table style="overflow: hidden">
<tr>
    <th>Name / <code>scipy.stats</code></th>
    <th>Params</th>
    <th>\( \lmoment{1} \)</th>
    <th>\( \lmoment{2} \)</th>
    <th>\( \lratio{3} = \lmoment{3}/\lmoment{2} \)</th>
    <th>\( \lratio{4} = \lmoment{4}/\lmoment{2} \)</th>
</tr>
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
    <td>\[ (b - a) / 6 \]</td>
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
    <td>\( \mu \)<br>\( s>0 \)</td>
    <td>\( \mu \)</td>
    <td>\( s \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ 1 / 6 \]
        \( \approx 0.1667 \)
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
        \[ 3b / 4 \]
        \( = 0.75 \ b\)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 17 / 72 \]
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
    <td>\( \nu := 2 \)</td>
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
    <td>\( \nu := 3 \)</td>
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
    <td>\( \nu := 4 \)</td>
    <td>\( 0 \)</td>
    <td>
        \[ 15 \pi / 64 \]
        \( \approx 0.7363 \)
    </td>
    <td>\( 0 \)</td>
    <td>
        \[ 111 / 512 \]
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
    <td>\[ \frac 1 3 \]</td>
    <td>\[ \frac 1 6 \]</td>
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
            \frac{\vphantom{3 / \sqrt 2}\sqrt \pi}{\vphantom{1 / \sqrt 2}\sqrt 2}
            \sigma 
        \]
        \( \approx 1.2533 \ \sigma \)
    </td>
    <td>
        \[ 
            \frac{\vphantom{3 / \sqrt 2}\sqrt{\pi}}{\vphantom{1 / \sqrt 2}2} 
            \left(\sqrt 2 - 1\right) 
            \sigma
        \]
        \( \approx 0.3671 \ \sigma \)
    </td>
    <td>
        \[ 
            \frac{1 - 3 / \sqrt 2 + 2 / \sqrt 3}{1 - 1 / \sqrt 2}
        \]
        \( \approx 0.1140 \)
    </td>
    <td>
        \[ 
            \frac
                {1 - 6 / \sqrt 2 + 10 / \sqrt 3 - 5 / \sqrt 4}
                {1 - 1 / \sqrt 2}
        \]
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
        <code>gumbel_l</code>
    </td>
    <td>\( \beta > 0 \)</td>
    <td>
        \[ \gamma_e \beta \]
        \( \approx 0.5772 \ \beta \)
    </td>
    <td>
        \[ \ln(2) \beta \]
        \( \approx 0.6931 \ \beta \)
    </td>
    <td>
        \[ 2 \log_2(3) - 3 \]
        \( \approx -0.1699 \)
    </td>
    <td>
        \[ 16 - 10 \log_2(3) \]
        \( \approx 0.1504 \)
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Generalized_extreme_value_distribution"
            target="_blank"
            title="Generalized extreme value distribution - Wikipedia"
        >
            GEV
        </a>
        <br>
        <code>genextreme</code>
    </td>
    <td>
        \( \kappa > -1 \)<br>
        \( \beta > 0 \)
    </td>
    <td>
        \[ \frac{1 - \Gamma(1 + \kappa)}{\kappa} \ \beta \]
    </td>
    <td>
        \[\Gamma(1 + \kappa) \ \boxcox{2}{-\kappa} \ \beta \]
    </td>
    <td>
        \[ 2 \frac{\boxcox{3}{-\kappa}}{\boxcox{2}{-\kappa}} - 3 \]
    </td>
    <td>
        \[ 
            6 
            + 5 \frac{\boxcox{4}{-\kappa}}{\boxcox{2}{-\kappa}} 
            - 10 \frac{\boxcox{3}{-\kappa}}{\boxcox{2}{-\kappa}} 
        \]
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Pareto_distribution"
            target="_blank"
            title="Pareto distribution - Wikipedia"
        >
            Pareto I
        </a>
        <br>
        <code>pareto</code>
    </td>
    <td>
        \( b \)<br>
        \( \sigma > 1 \)
    </td>
    <td>
        \[ \frac{b}{b - 1} \ \sigma \]
    </td>
    <td>
        \[ \frac{b}{b - 1} \frac{1}{2b - 1} \ \sigma \]
    </td>
    <td>
        \[ \frac{b + 1}{3b - 1} \]
    </td>
    <td>
        \[ \frac{b + 1}{3b - 1} \frac{2b + 1}{4b - 1} \]
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Pareto_distribution#Pareto_types_I%E2%80%93IV"
            target="_blank"
            title="Pareto distribution - Wikipedia"
        >
            Pareto II
        </a>
        <br>
        <code>lomax</code>
    </td>
    <td>
        \( b, \ \mu \)<br>
        \( \sigma > 1 \)
    </td>
    <td>
        \[ \frac{1}{b - 1} \ \sigma + \mu \]
    </td>
    <td>
        \[ \frac{b}{b - 1} \frac{1}{2b - 1} \ \sigma \]
    </td>
    <td>
        \[ \frac{b + 1}{3b - 1} \]
    </td>
    <td>
        \[ \frac{b + 1}{3b - 1} \frac{2b + 1}{4b - 1} \]
    </td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Pareto_distribution#Pareto_types_I%E2%80%93IV"
            target="_blank"
            title="Pareto distribution - Wikipedia"
        >
            Pareto III
        </a>
        <br>
    </td>
    <td>
        \( c, \ \mu \)<br>
        \( \sigma > 1 \)
    </td>
    <td>
        \[ c \ \Gamma(c) \ \Gamma(1 - c) \ \sigma + \mu \]
    </td>
    <td>
        \[ c^2 \ \Gamma(c) \ \Gamma(1 - c) \ \sigma \]
    </td>
    <td>
        \[ c \vphantom{c^2 \Gamma(c)} \]
    </td>
    <td>
        \[ \frac{1 + 5 c^2}{6} \]
    </td>
</tr>

<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Kumaraswamy_distribution"
            target="_blank"
            title="Kumaraswamy distribution - Wikipedia"
        >
            Kumaraswamy
        </a>
        <br>
    </td>
    <td>
        \( \alpha > 0, \ \beta > 0 \)<br>
        <!-- \( \eta \stackrel{\text{def}}{=} 1 + 1 / a \) -->
        \( 
            \omega_n := \B \bigl( \frac{1 + \alpha}{\alpha}, n \beta \bigr) 
        \)
    </td>
    <td>
        \[ \omega_1 \beta \]
    </td>
    <td>
        \[ \left( \omega_1 - 2 \omega_2 \right) \beta \]
    </td>
    <td>
        \[ 
            \frac
                {\omega_1 - 6 \omega_2 + 6 \omega_3}
                {\omega_1 - 2 \omega_2} 
        \]
        <!-- \[
            1 - \frac
                {4 \B(\eta, 2b) - 6 \B(\eta, 3b)}
                {\B(\eta, b) - 2 \B(\eta, 2b)} 
        \] -->
    </td>
    <td>
        \[
            \frac
                {\omega_1 - 12 \omega_2 + 30 \omega_3 - 20 \omega_4}
                {\omega_1 - 2 \omega_2}
        \]
        <!-- \[
            1 - 10 \frac
                {\B(\eta, 2b) - 3 \B(\eta, 3b) + 2 \B(\eta, 4b)}
                {\B(\eta, b) - 2 \B(\eta, 2b)}
        \] -->
    </td>
</tr>

</table>


Constants

<table style="overflow: hidden">
<tr>
    <th>Name</th>
    <th>Symbol</th>
    <th>Definition</th>
    <th>Value</th>
    <th>Python</th>
</tr>
<tr>
    <td>
        <a
            href="https://en.wikipedia.org/wiki/Pi"
            target="_blank"
            title="Pi - Wikipedia"
        >
            Pi
        </a>
    </td>
    <td>\( \pi \)</td>
    <td>\[ \int_{-1}^1 \frac{\mathrm{d} x}{\sqrt{1 - x^2}} \]</td>
    <td>\( \approx 3.1416 \)</td>
    <td><code>numpy.pi</code></td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Euler%27s_constant"
            target="_blank"
            title="Euler's constant"
        >
            Euler–Mascheroni constant
        </a>
    </td>
    <td>\( \gamma_e \)</td>
    <td>
        \[
            \int_1^\infty 
            \left( 
                \frac{1}{\lfloor x \rfloor} - \frac 1 x 
            \right) \ 
            \mathrm{d} x 
        \]
    </td>
    <td>\( \approx 0.5772 \)</td>
    <td><code>numpy.euler_gamma</code></td>
</tr>
</table>


Special functions

<table style="overflow: hidden">
<tr>
    <th>Name</th>
    <th>Notation</th>
    <th>Definition</th>
    <th>Python</th>
</tr>
<tr id="def-gamma">
    <td>
        <a
            href="https://wikipedia.org/wiki/Gamma_function"
            target="_blank"
            title="Gamma function - Wikipedia"
        >
            Gamma function
        </a>
    </td>
    <td>\( \Gamma(z) \)</td>
    <td>\[ \int_0^\infty t^{z-1} e^{-t} \, \mathrm{d} t \]</td>
    <td><code>scipy.special.gamma</code></td>
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
    <td>\( \B(z_1, z_2) \)</td>
    <td>
        \[ 
            \frac{\Gamma(z_1) \Gamma(z_2)}{\Gamma(z_1 + z_2)}
        \]
    </td>
    <td><code>scipy.special.beta</code></td>
</tr>
<tr id="def-bcox">
    <td>
        <a
            href="https://wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation"
            target="_blank"
            title="Box–Cox transformation - Power transform - Wikipedia"
        >
            Box–Cox transform
        </a>
    </td>
    <td>\( \boxcox{y}{\lambda} \)</td>
    <td>
        \[
            \begin{cases}
                (y^\lambda - 1) / \lambda & \text{if } \lambda \neq 0 \\
                \ln(y) & \text{if } \lambda = 0
            \end{cases}
        \]
    </td>
    <td><code>scipy.special.boxcox</code></td>
</tr>
</table>
