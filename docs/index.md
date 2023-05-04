# Lmo

{%
   include-markdown "../README.md"
   start="<!--badges-start-->"
   end="<!--badges-end-->"
%}


Lmo is a lightweight library with pythonic syntax. Some features include:

 - Lightweight; it only requires [numpy](https://numpy.org/doc/stable/index.html)
 - Clean code style: linted with [ruff](https://github.com/charliermarsh/ruff)
 - Fully type-annotated, valid in [pyright](https://github.com/microsoft/pyright)'s strict mode.
 - [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)-tested
 - Flat functions, no classes (scipy > scikit) 
 - Red and fluffy

## Why (T)L-moments?

!!! note info "Coming soon. For now, see [Wikipedia](https://wikipedia.org/wiki/L-moment)"


## Roadmap

- [x] Sample L-, and TL-moment estimators
- [x] Multivariate L- and TL- co-moments.
- [ ] Standard error / covariance matrices of the sample (T)L-moments (Elamir et al. 2002, 4.1.)
- [ ] Parameter estimatation for known probability distributions.
- [ ] Better docs: introduction, motivation, examples, etc.
- [ ] Numerical tools for estimating (T)L-moments of unknown distributions.
- [ ] Optional numba JIT support.
- [ ] Generic (T)L method-of-moments implementation
- [ ] LQ-moments (Mudholkar and Hutson, 1998)
