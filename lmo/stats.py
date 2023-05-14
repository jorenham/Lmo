from math import factorial as fact

from lmo._utils import expand_trimming
from lmo.typing import Trimming


def tl_ratio_max(
    r: int,
    /,
    k: int = 2,
    trim: Trimming = 1,
) -> float:
    """
    The theoretical upper bound on the absolute TL-ratios, i.e.::

        abs(tl_ratio(a, r, k, (tl, tr))) <= tl_ratio_max(r, k, tl, tr)

    is True for all samples `a`.

    References:
        * Hosking, J.R.M., Some Theory and Practical Uses of Trimmed L-moments.
          Page 6, equation 14.

    """

    # the zeroth (TL-)moment is 1. I.e. the total area under the pdf (or the
    # sum of the ppf if discrete) is 1.
    if r in (0, k):
        return 1.0
    if not k:
        return float('inf')

    if r < 0:
        raise ValueError(f'expected r >= 0, got {r} < 0')
    if k < 0:
        raise ValueError(f'expected k >= 0, got {k} < 0')

    tl, tr = expand_trimming(trim)

    m = min(tl, tr)
    # disclaimer: the `k` instead of a `2` here is just a guess
    return (
        k * fact(m + k - 1) * fact(tl + tr + r) /
        (r * fact(m + r - 1) * fact(tl + tr + k))
    )
