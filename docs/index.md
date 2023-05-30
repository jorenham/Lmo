{%
   include-markdown "../README.md"
   start="<!--head-start-->"
   end="<!--head-end-->"
%}


Lmo is a lightweight library with pythonic syntax. Some features include:

 - Robust alternatives to conventional [moments](https://wikipedia.org/wiki/Moment_(mathematics)): even the [Cauchy distribution](https://wikipedia.org/wiki/Cauchy_distribution) poses no threat!
 - Handles both univariate and multivariate cases
 - Support for custom sample weights
 - Lightweight; it only requires [numpy](https://numpy.org/doc/stable/index.html)
 - Scales to millions of samples, with $O(n \\log(n))$ time-complexity and $O(n)$ space-complexity. 
 - Clean code style: linted with [ruff](https://github.com/charliermarsh/ruff), and valid in [pyright](https://github.com/microsoft/pyright)'s strict mode
 - Thoroughly tested with [hypothesis](https://hypothesis.readthedocs.io/en/latest/)
 - Red and fluffy

## Installation

```shell
pip install lmo
```

## Roadmap

- [x] Sample L-, and TL-moment estimators
- [x] Sample L- and TL- co-moments (multivariate) estimators
- [x] Support for observation weights.
- [x] L-moments (co)variance structure [#4](https://github.com/jorenham/lmo/issues/4)
- [ ] Fitting of distributions with known L-moments [#5](https://github.com/jorenham/lmo/issues/5)
- [ ] Population L-moment estimation from quantile functions [#6](https://github.com/jorenham/lmo/issues/6)
