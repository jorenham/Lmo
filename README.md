<!--overview-start-->

# Lmo - Trimmed L-moments and L-comoments

![Lmo - License][IMG-BSD]
[![Lmo - PyPI][IMG-PYPI]](https://pypi.org/project/Lmo/)
[![Lmo - Versions][IMG-VER]](https://github.com/jorenham/Lmo)
![Lmo - CI][IMG-CI]
[![Lmo - Pre-commit][IMG-PC]](https://github.com/pre-commit/pre-commit)
[![Lmo - Ruff][IMG-RUFF]](https://github.com/astral-sh/ruff)
[![Lmo - BassedPyright][IMG-BPR]](https://detachhead.github.io/basedpyright)

[IMG-CI]: https://img.shields.io/github/actions/workflow/status/jorenham/Lmo/ci.yml?branch=master
[IMG-BSD]: https://img.shields.io/github/license/jorenham/Lmo
[IMG-PYPI]: https://img.shields.io/pypi/v/Lmo
[IMG-VER]: https://img.shields.io/pypi/pyversions/Lmo
[IMG-pC]: https://img.shields.io/badge/pre--commit-enabled-orange?logo=pre-commit
[IMG-RUFF]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[IMG-BPR]: https://img.shields.io/badge/basedpyright-checked-42b983

Unlike the legacy
[product-moments](https://wikipedia.org/wiki/Moment_(mathematics)), the
[*L-moments*](https://wikipedia.org/wiki/L-moment) **uniquely describe** a
probability distribution, and are more robust and efficient.

The "L" stands for Linear; it is a linear combination of order statistics.
So Lmo is as fast as sorting your samples (in terms of time-complexity).

## Key Features

- Calculates trimmed L-moments and L-*co*moments, from samples or any
  `scipy.stats` distribution.
- Full support for trimmed L-moment (TL-moments), e.g.
  `lmo.l_moment(..., trim=(1/137, 3.1416))`.
- Generalized Method of L-moments: robust distribution fitting that beats MLE.
- Fast estimation of L-*co*moment matrices from your multidimensional data
  or multivariate distribution.
- Goodness-of-fit test, using L-moment or L-moment ratio's.
- Exact (co)variance structure of the sample- and population L-moments.
- Theoretical & empirical influence functions of L-moments & L-ratio's.
- Complete [docs](https://jorenham.github.io/lmo/), including detailed API
reference with usage examples and with mathematical $\TeX$ definitions.
- Clean Pythonic syntax for ease of use.
- Vectorized functions for very fast fitting.
- Fully typed, tested, and tickled.
- Optional Pandas integration.

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

## Installation

Lmo is on [PyPI](https://pypi.org/project/lmo/), so you can do something like:

```shell
pip install lmo
```

### Dependencies

These are automatically installed by your package manager when installing Lmo.

| Package      | Supported versions |
| ------------ | ------------------ |
| [Python][PY] | `>=3.10`           |
| [NumPy][NP]  | `>=1.23`           |
| [SciPy][SP]  | `>=1.9`            |

Additionally, Lmo supports the following optional packages:

| Package      | Supported versions | Installation              |
| ------------ | ------------------ | ------------------------- |
| [Pandas][PD] | `>=1.5`            | `pip install Lmo[pandas]` |

See [SPEC 0][SPEC0] for more information.

[PY]: https://github.com/python/cpython
[NP]: https://github.com/numpy/numpy
[SP]: https://github.com/scipy/scipy
[PD]: https://github.com/pandas-dev/pandas
[SPEC0]: https://scientific-python.org/specs/spec-0000/

## Foundational Literature

- [*J.R.M. Hosking* (1990) &ndash; L-moments: Analysis and Estimation of
  Distributions using Linear Combinations of Order Statistics
  ](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)
- [*E.A.H. Elamir & A.H. Seheult* (2003) &ndash; Trimmed L-moments
  ](https://doi.org/10.1016/S0167-9473(02)00250-5)
- [*J.R.M. Hosking* (2007) &ndash; Some theory and practical uses of trimmed
  L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
- [*R. Serﬂing & P. Xiao* (2007) &ndash; A contribution to multivariate
  L-moments: L-comoment matrices](https://doi.org/10.1016/j.jmva.2007.01.008)

<!--overview-end-->
