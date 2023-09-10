<!--overview-start-->

<img src="https://jorenham.github.io/Lmo/img/lmo.svg" alt="jorenham/lmo" width="128" align="right">

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


## Key Features

- Calculates trimmed L-moments and L-*co*moments, from samples or any
  `scipy.stats` distribution.
- Full support for trimmed L-moment (TL-moments), e.g.
  `lmo.l_moment(..., trim=(1/137, 3.1416))`.
- Method of L-moments for robust distribution fitting.
- Fast estimation of L-*co*moment matrices from your multidimensional data.
- Goodness-of-fit test, using L-moment or L-moment ratio's.
- Non-parametric estimation of continuous distributions 
  with `lmo.l_rv_nonparametric`
- Exact (co)variance structure of the sample- and population L-moments.
- Theoretical influence functions of the L-moment (ratio's) of distributions.
- Complete [docs](https://jorenham.github.io/lmo/), including overly 
  complex $\TeX$ spaghetti equations.
- Clean Pythonic syntax for ease of use.
- Vectorized functions for very fast fitting.
- Fully typed, tested, and tickled.


## Quick example


Even if your data is pathological like 
[Cauchy](https://wikipedia.org/wiki/Cauchy_distribution), and the L-moments 
are not defined, the trimmed L-moments (TL-moments) can be used instead.
Let's calculate the TL-location and TL-scale of a small amount of samples:

```pycon
>>> import numpy as np
>>> import lmo
>>> rng = np.random.default_rng(1980)
>>> x = rng.standard_cauchy(96)  # pickle me, Lmo
>>> lmo.l_moment(x, [1, 2], trim=(1, 1)).
array([-0.17937038,  0.68287665])
```

Now compare with the theoretical standard Cauchy TL-moments:

```pycon
>>> from scipy.stats import cauchy
>>> cauchy.l_moment([1, 2], trim=(1, 1))
array([0.        , 0.69782723])
```

---

See the [documentation](https://jorenham.github.io/lmo/) for more examples and
the API reference.


## Roadmap

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
