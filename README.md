<!--overview-start-->

# Lmo - Trimmed L-moments and L-comoments

[![Lmo - PyPI][IMG-PYPI]][PYPI]
[![Lmo - Python versions][IMG-PY]][REPO]
[![Lmo - SPEC 0][IMG-SPEC0]][SPEC0]
[![Lmo - BSD License][IMG-BSD]][BSD]
[![Lmo - CI][IMG-CI]][CI]
[![Lmo - pre-commit][IMG-PC]][PC]
[![Lmo - ruff][IMG-RUFF]][RUFF]
[![Lmo - bassedpyright][IMG-BPR]][BPR]

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
- Complete [docs][DOCS], including detailed API
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
>>> lmo.l_moment(x, [1, 2], trim=(1, 1))
array([-0.17937038,  0.68287665])
```

Now compare with the theoretical standard Cauchy TL-moments:

```pycon
>>> from scipy.stats import cauchy
>>> cauchy.l_moment([1, 2], trim=(1, 1))
array([0.        , 0.69782723])
```

---

See the [documentation][DOCS] for more examples and the API reference.

## Installation

Lmo is available on [PyPI][PYPI], and can be installed
with:

```shell
pip install lmo
```

If you care about static typing, then it is recommended to install Lmo as
`Lmo[typing]`, i.e.:

```shell
pip install Lmo[typing]
```

## Roadmap

- Automatic trim-length selection.
- Plotting utilities (deps optional), e.g. for L-moment ratio diagrams.

### Dependencies

These are automatically installed by your package manager when installing Lmo.

|                | version  |
| -------------: | -------- |
| [`python`][PY] | `>=3.11` |
| [`numpy`][NP]  | `>=1.24` |
| [`scipy`][SP]  | `>=1.10` |

Additionally, Lmo supports the following optional packages:

|                      | version      | `pip install _` | extra requirements |
| -------------------: | ------------ | --------------- | ------------------ |
| [`scipy-stubs`][SPT] | `>=1.14.1.0` | `Lmo[typing]`   | `scipy >= 1.14.1`  |
| [`pandas`][PD]       | `>=2.0`      | `Lmo[pandas]`   |                    |

See [SPEC 0][SPEC0] for more information.

## Foundational Literature

- [*J.R.M. Hosking* (1990) &ndash; L-moments: Analysis and Estimation of Distributions using Linear Combinations of Order Statistics](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)
- [*E.A.H. Elamir & A.H. Seheult* (2003) &ndash; Trimmed L-moments](https://doi.org/10.1016/S0167-9473(02)00250-5)
- [*J.R.M. Hosking* (2007) &ndash; Some theory and practical uses of trimmed L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
- [*R. Serﬂing & P. Xiao* (2007) &ndash; A contribution to multivariate L-moments: L-comoment matrices](https://doi.org/10.1016/j.jmva.2007.01.008)

[IMG-PYPI]: https://img.shields.io/pypi/v/Lmo
[IMG-PY]: https://img.shields.io/pypi/pyversions/Lmo
[IMG-SPEC0]: https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[IMG-BSD]: https://img.shields.io/github/license/jorenham/Lmo
[IMG-CI]: https://img.shields.io/github/actions/workflow/status/jorenham/Lmo/ci.yml?branch=master
[IMG-PC]: https://img.shields.io/badge/pre--commit-enabled-orange?logo=pre-commit
[IMG-RUFF]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[IMG-BPR]: https://img.shields.io/badge/basedpyright-checked-42b983

[PYPI]: https://pypi.org/project/Lmo/
[REPO]: https://github.com/jorenham/Lmo
[DOCS]: https://jorenham.github.io/Lmo/
[BSD]: https://github.com/jorenham/Lmo/blob/master/LICENSE
[CI]: https://github.com/jorenham/Lmo/actions/workflows/ci.yml?query=branch%3Amaster

[PC]: https://github.com/pre-commit/pre-commit
[RUFF]: https://github.com/astral-sh/ruff
[BPR]: https://github.com/detachhead/basedpyright/
[PY]: https://github.com/python/cpython
[NP]: https://github.com/numpy/numpy
[SP]: https://github.com/scipy/scipy
[PD]: https://github.com/pandas-dev/pandas
[SPT]: https://github.com/jorenham/scipy-stubs
[SPEC0]: https://scientific-python.org/specs/spec-0000/

<!--overview-end-->
