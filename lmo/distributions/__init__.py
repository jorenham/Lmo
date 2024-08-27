# pyright: reportAbstractUsage=false

"""
Probability distributions, compatible with [`scipy.stats`][scipy.stats].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import lmo.typing.scipy as lspt
from . import _lm
from ._genlambda import genlambda_gen
from ._kumaraswamy import kumaraswamy_gen
from ._lm import *  # noqa: F403
from ._nonparametric import l_poly
from ._wakeby import wakeby_gen


__all__ = ['l_poly', 'kumaraswamy', 'wakeby', 'genlambda']
__all__ += _lm.__all__


# mkdocstring workaround
if not TYPE_CHECKING:
    wakeby = wakeby_gen(name='wakeby', a=0.0)
    r"""A Wakeby random variable, a generalization of
    [`scipy.stats.genpareto`][scipy.stats.genpareto].

    [`wakeby`][wakeby] takes \( b \), \( d \) and \( f \) as shape parameters.

    For a detailed description of the Wakeby distribution, refer to
    [Distributions - Wakeby](../distributions.md#wakeby).
    """
else:
    wakeby: Final = cast(lspt.RVContinuous, wakeby_gen(a=0.0, name='wakeby'))

# mkdocstring workaround
if not TYPE_CHECKING:
    kumaraswamy = kumaraswamy_gen(name='kumaraswamy', a=0.0, b=1.0)
    r"""
    A Kumaraswamy random variable, similar to
    [`scipy.stats.beta`][scipy.stats.beta].

    The probability density function for
    [`kumaraswamy`][lmo.distributions.kumaraswamy] is:

    \[
        f(x, a, b) = a x^{a - 1} b \left(1 - x^a\right)^{b - 1}
    \]

    for \( 0 < x < 1,\ a > 0,\ b > 0 \).

    [`kumaraswamy`][kumaraswamy] takes \( a \) and \( b \) as shape parameters.

    See Also:
        - [Theoretical L-moments: Kumaraswamy](../distributions.md#kumaraswamy)
    """
else:
    kumaraswamy: Final = cast(
        lspt.RVContinuous,
        kumaraswamy_gen(a=0.0, b=1.0, name='kumaraswamy'),
    )

# mkdocstring workaround
if not TYPE_CHECKING:
    genlambda = genlambda_gen(name='genlambda')
    r"""A generalized Tukey-Lambda random variable.

    `genlambda` takes `b`, `d` and `f` as shape parameters.
    `b` and `d` can be any float, and `f` requires `-1 <= f <= 1`.

    If `f == 0` and `b == d`, `genlambda` is equivalent to
    [`scipy.stats.tukeylambda`][scipy.stats.tukeylambda], with `b` (or `d`) as
    shape parameter.

    For a detailed description of the GLD, refer to
    [Distributions - GLD](../distributions.md#gld).
    """
else:
    genlambda: Final = cast(
        lspt.RVContinuous,
        genlambda_gen(name='genlambda'),
    )
