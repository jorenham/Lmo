<!--overview-start-->

<img src="https://jorenham.github.io/lmo/img/lmo.svg" alt="jorenham/lmo" width="128" align="right">

# Lmo - Trimmed L-moments and L-comoments

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/jorenham/lmo/CI.yml?branch=master&style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/lmo?style=flat-square)](https://pypi.org/project/lmo/)
[![versions](https://img.shields.io/pypi/pyversions/lmo?style=flat-square)](https://github.com/jorenham/lmo)
[![license](https://img.shields.io/github/license/jorenham/lmo?style=flat-square)](https://github.com/jorenham/lmo/blob/master/LICENSE?)

~~~

Is your tail too heavy? 
Can't find a moment? 
Are the swans black? 
The distribution pathological?

... then look no further: Lmo's got you covered!

Uniform or multi-dimensional, Lmo can summarize it all with one quick glance!

~~~

Unlike the legacy [moments](https://wikipedia.org/wiki/Moment_(mathematics)),
[L-moments](https://wikipedia.org/wiki/L-moment) **uniquely describe** a 
probability distribution, and are more robust and efficient.
The "L" stands for Linear; it is a linear combination of order statistics.
So Lmo is as fast as sorting your samples (in terms of time-complexity).

## Key Features:

- Calculates trimmed L-moments and L-*co*moments, from data or a distribution
  function.
- Complete support for trimmed L-moment (TL-moments): 
  `lmo.l_moment(..., trim=(1 / 137, 3.1416))`.
- Fast estimation of L-*co*moment matrices from your multidimensional data.
- A fully non-parametric `scipy.stats`-like distribution, `lmo.l_rv`.
  It's very efficient, robust, fast, and requires only some L-mo's!
- Exact (co)variance structure of the L-moment estimates.
- Complete [docs](https://jorenham.github.io/lmo/), including overly 
  complex $\TeX$ spaghetti equations.
- Clean Pythonic syntax for ease of use.
- Vectorized functions for very fast fitting.
- Fully typed, tested, and tickled.


## Quick example


Even if your data is pathological like 
[Cauchy](https://wikipedia.org/wiki/Cauchy_distribution), and the L-moments 
are not defined, the trimmed L-moments (TL-moments) can be used instead:

```pycon
>>> import numpy as np
>>> import lmo
>>> rng = np.random.default_rng(1980)
>>> x = rng.standard_cauchy(96)  # pickle me, Lmo
>>> x.mean(), x.std()  # don't try this at home
(-1.7113440959133905, 19.573507308373326)
>>> lmo.l_loc(x, trim=(1, 1)), lmo.l_scale(x, (1, 1)) 
(-0.17937038148581977, 0.6828766469913776)
```

For reference; the theoretical TL-location and TL-scale of the standard 
Cauchy distribution are $\lambda^{(1, 1)}_{1} = 0$ and 
$\lambda^{(1, 1)}_2 \approx 0.7$ 
([Elamir & Seheult, 2003](https://doi.org/10.1016/S0167-9473(02)00250-5)).



---

See the [documentation](https://jorenham.github.io/lmo/) for more examples and
the API reference.


## Roadmap:

- Add methods to all `scipy.stats` univariate distributions, for finding 
  the theoretical/population L-mo's.
- Robust distribution fitting, using the (generalized) method of L-moments.
- Theoretical L-moment variance structure calculation.
- A generic goodness-of-fit test.
- Automatic trim-length selection.
- Plotting utilities (deps optional), e.g. for L-moment ratio diagrams.
- Extended multivariate support, e.g. theoretical L-comoments, and 
  L-regression.

## Installation

Lmo is on [PyPI](https://pypi.org/project/lmo/), so you can do something like:

```shell
pip install lmo
```


## Dependencies

- `python >= 3.10`
- `numpy >= 1.22`
- `scipy >= 1.9`


## Foundational Literature

- [*J.R.M. Hosking* (1990) &ndash; L-moments: Analysis and Estimation of 
  Distributions using Linear Combinations of Order Statistics
  ](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)
- [*E.A.H. Elamir & A.H. Seheult* (2003) &ndash; Trimmed L-moments
  ](https://doi.org/10.1016/S0167-9473(02)00250-5)
- [*E.A.H. Elamir & A.H. Seheult* (2004) &ndash; Exact variance structure of 
  sample L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)
- [*J.R.M. Hosking* (2007) &ndash; Some theory and practical uses of trimmed 
  L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
- [*R. Serﬂing & P. Xiao* (2007) &ndash; A contribution to multivariate 
  L-moments: L-comoment matrices](https://doi.org/10.1016/j.jmva.2007.01.008)
- [*W.H. Asquith* (2011) &ndash; Univariate Distributional Analysis with 
  L-moment Statistics](https://hdl.handle.net/2346/ETD-TTU-2011-05-1319)


<!--overview-end-->
