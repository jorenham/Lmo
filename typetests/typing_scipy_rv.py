from scipy.stats import distributions

import lmo.typing.scipy as lspt  # noqa: TCH001


D_uniform: lspt.AnyRV = distributions.uniform
D_binomial: lspt.AnyRV = distributions.binom
