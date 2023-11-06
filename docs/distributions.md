# L-moments of common probability distributions

## Univariate

### Continuous

<table style="overflow: hidden">
<tr>
    <th>Name</th>
    <th>Params</th>
    <th><code>scipy.stats._</code></th>
    <th>\( \lambda_1 \)</th>
    <th>\( \lambda_2 \)</th>
    <th>\( \tau_3 \)</th>
    <th>\( \tau_4 \)</th>
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
    </td>
    <td>\( a < b \)</td>
    <td><code>uniform(a, b-a)</code></td>
    <td>\( (a + b) / 2 \)</td>
    <td>\( (b - a) / 6 \)</td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0">\( 0 \)</td>
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
    </td>
    <td>\( \mu,\, \sigma>0 \)</td>
    <td><code>norm(μ, σ)</code></td>
    <td>\( \mu \)</td>
    <td>
        \( \sigma / \sqrt{\pi} \)<br>\( \approx 0.5642 \sigma \)
    </td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.1226">
        \( 30 \, \theta_m / \pi - 9 \)<br>\( \approx 0.1226 \)
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
    </td>
    <td>\( \mu,\, s>0 \)</td>
    <td><code>logistic(μ, s)</code></td>
    <td>\( \mu \)</td>
    <td>\( s \)</td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.1667">
        \( \frac{1}{6} \)<br>\( \approx 0.1667 \)
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
    </td>
    <td>\( \mu,\, b>0 \)</td>
    <td><code>laplace(μ, b)</code></td>
    <td>\( \mu \)</td>
    <td>\( \frac{3}{4} b \)<br>\( = 0.75 b\)</td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.2361">
        \( \frac{17}{72} \)<br>\( \approx 0.2361 \)
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
    </td>
    <td>\( \nu=2 \)</td>
    <td><code>t(2)</code></td>
    <td>\( 0 \)</td>
    <td>
        \( \frac{1}{2 \sqrt{2}} \pi\)<br>\( \approx 1.1107 \)
    </td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.375">
        \( \frac{3}{8} \)<br>\( = 0.375 \)
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
    </td>
    <td>\( \nu=3 \)</td>
    <td><code>t(3)</code></td>
    <td>\( 0 \)</td>
    <td>
        \( \frac{3 \sqrt{3}}{2} / \pi \)<br>\( \approx 0.8270 \)
    </td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.2612">
        \( 1 - \frac{175}{24} / \pi^2 \)<br>\( \approx 0.2612 \)
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
    </td>
    <td>\( \nu=4 \)</td>
    <td><code>t(4)</code></td>
    <td>\( 0 \)</td>
    <td>\( \frac{15}{64} \pi \)<br>\( \approx 0.7363 \)</td>
    <td data-sort-value="0">\( 0 \)</td>
    <td data-sort-value="0.2612">
        \( \frac{111}{512} \)<br>\( \approx 0.2168 \)
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
    </td>
    <td>\( \lambda>0 \)</td>
    <td><code>expon(0, 1/λ)</code></td>
    <td>\( \frac{1}{\lambda} \)</td>
    <td>\( \frac{1}{2 \lambda} \)</td>
    <td data-sort-value="0.3333">
        \( 1/3 \)<br>\( \approx 0.3333 \)
    </td>
    <td data-sort-value="0.1667">
        \( \frac{1}{6} \)<br>\( \approx 0.1667 \)
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
    </td>
    <td>\( \sigma > 0 \)</td>
    <td><code>rayleigh(0, σ)</code></td>
    <td>
        \( \sqrt{\frac{1}{2} \pi} \, \sigma \)<br>
        \( \approx 1.2533 \sigma \)
    </td>
    <td>
        \( \frac{1}{2} \left(\sqrt{2} - 1\right) \sqrt{\pi} \, \sigma \)<br>
        \( \approx 0.3671 \sigma \)
    </td>
    <td>
        \( \frac{1 - 3 / \sqrt{2} + 2 / \sqrt{3}}{1 - 1 / \sqrt{2}} \)<br>
        \(\approx 0.1140 \)
    </td>
    <td>
        \( \frac{1 - 6 / \sqrt{2} + 10 / \sqrt{3} - 5 / \sqrt{4}}{1 - 1 / \sqrt{2}} \)<br>
        \(\approx 0.1054 \)
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
    </td>
    <td>\( \beta>0 \)</td>
    <td><code>gumbel_r(0, β)</code></td>
    <td>
        \( \gamma_e \beta \)<br>
        \( \approx 0.5772 \beta \)
    </td>
    <td>
        \( \ln(2) \beta \)<br>\( \approx 0.6931 \beta \)
    </td>
    <td data-sort-value="0.1699">
        \( 3 - 2 \log_2(3) \)<br>\( \approx 0.1699 \)
    </td>
    <td data-sort-value="0.1504">
        \( 16 - 10 \log_2(3) \)<br>\( \approx 0.1504 \)
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
    </td>
    <td>\( \kappa > -1,\, \beta > 0 \)</td>
    <td><code style="white-space: nowrap;">genextreme(κ, 0, β)</code></td>
    <td>
        \( \left(\frac{1}{\kappa} - \Gamma(\kappa)\right) \beta \)
    </td>
    <td>
        \( \kappa\, \xi(2, -\kappa) \Gamma(\kappa) \beta \)
    </td>
    <td>
        <!-- \( 2 (1 - 3^{-\kappa}) / (1 - 2^{-\kappa}) - 3 \) -->
        \( \frac{2 \xi(3, -\kappa) - 3 \xi(2, -\kappa)}{\xi(2, -\kappa)} \)
    </td>
    <td>
        <!-- \( 6 + 5 ((1 - 4^{-\kappa}) - 2 (1 - 3^{-\kappa})) / (1 - 2^{-\kappa}) \) -->
        \( \frac{5 \xi(4, -\kappa) - 10 \xi(3, -\kappa) + 6 \xi(2, -\kappa)}{\xi(2, -\kappa)} \)
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
    </td>
    <td>\( b,\, \sigma > 1 \)</td>
    <td><code>pareto(b, 0, σ)</code></td>
    <td>\( \frac{b}{b - 1} \sigma \)</td>
    <td>\( \frac{b}{b - 1} \frac{1}{2b - 1} \sigma \)</td>
    <td>\( \frac{b + 1}{3b - 1} \)</td>
    <td>\( \frac{b + 1}{3b - 1} \frac{2b + 1}{4b - 1}  \)</td>
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
    </td>
    <td>\( b,\, \mu,\, \sigma > 1 \)</td>
    <td><code>lomax(b, μ, σ)</code></td>
    <td>\( \mu + \frac{1}{b - 1} \sigma \)</td>
    <td>\( \frac{b}{b - 1} \frac{1}{2b - 1} \sigma \)</td>
    <td>\( \frac{b + 1}{3b - 1} \)</td>
    <td>\( \frac{b + 1}{3b - 1} \frac{2b + 1}{4b - 1}  \)</td>
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
    </td>
    <td>\( c,\, \mu,\, \sigma > 1 \)</td>
    <td></td>
    <!-- <td>\( \mu + \frac{1}{\mathrm{sinc}(c)} \sigma \)</td> -->
    <td>\( \mu + c\,\Gamma(c) \Gamma(1-c) \sigma \)</td>
    <!-- <td>\( \frac{c}{\mathrm{sinc}(c)} \sigma \)</td> -->
    <td>\( c^2 \, \Gamma(c) \Gamma(1-c) \sigma \)</td>
    <td>\( c \)</td>
    <td>\( \frac{1}{6} + \frac{5}{6} c^2 \)</td>
</tr>
<!-- 
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Kumaraswamy_distribution"
            target="_blank"
            title="Kumaraswamy distribution - Wikipedia"
        >
            Kumaraswamy
        </a>
    </td>
    <td>\( a > 0,\, b > 0\)<br> \(\, \eta = 1 + 1/a \)</td>
    <td></td>
    <td>\( b B(\eta, b) \)</td>
    <td>\( b B(\eta, b) - 2b B(\eta, 2b) \)</td>
    <td>\( \frac{B(\eta, b) - 6 B(\eta, 2b) + 6 B(\eta, 3b)}{B(\eta, b) - 2 B(\eta, 2b)} \)</td>
    <td>\( \frac{B(\eta, b) - 12 B(\eta, 2b) + 30 B(\eta, 3b) - 40 B(\eta, 4b)}{B(\eta, b) - 2 B(\eta, 2b)} \)</td>
</tr>
-->
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
    <td>\( = \int_{-1}^1 1/\sqrt{1-x^2} \, \mathrm{d} x \)</td>
    <td>\( \approx 3.1416 \)</td>
    <td><code>numpy.pi</code></td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Magic_angle"
            target="_blank"
            title="Magic angle - Wikipedia"
        >
            Magic Angle
        </a>
    </td>
    <td>\( \theta_m \)</td>
    <td>\( = \tan^{-1}(\sqrt{2}) = \sec^{-1}(\sqrt{3}) \)</td>
    <td>\( \approx 0.9553 \)</td>
    <td><code>numpy.arctan(numpy.sqrt(2))</code></td>
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
    <td>\( = \int_1^\infty (1/\lfloor x \rfloor - 1/x) \, \mathrm{d} x \)</td>
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
<tr>
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
    <td>\( = \int_0^\infty t^{z-1} e^{-t} \, \mathrm{d} t \)</td>
    <td><code>scipy.special.gamma</code></td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Beta_function"
            target="_blank"
            title="Beta function - Wikipedia"
        >
            Beta function
        </a>
    </td>
    <td>\( \mathrm{B}(z_1, z_2) \)</td>
    <td>\( = \Gamma(z_1) \Gamma(z_2) / \Gamma(z_1 + z_2) \)</td>
    <td><code>scipy.special.beta</code></td>
</tr>
<tr>
    <td>
        <a
            href="https://wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation"
            target="_blank"
            title="Box–Cox transformation - Power transform - Wikipedia"
        >
            Box–Cox transform
        </a>
    </td>
    <td>\( \xi(y, k) \)</td>
    <td>
        \[
            =
            \begin{cases}
                (y^k - 1) / k & \text{if } k \neq 0 \\
                \ln(y) & \text{if } k = 0
            \end{cases}
        \]
    </td>
    <td><code>scipy.special.boxcox</code></td>
</tr>
</table>
