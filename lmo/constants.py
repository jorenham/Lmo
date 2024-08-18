"""Mathematical constants."""
from typing import Final


__all__ = 'theta_m', 'theta_m_bar'


theta_m: Final[float] = 0.955316618124509278163857102515757754
r"""
Magic angle \( \theta_m = \arctan \sqrt 2 \).

See also:
    - [Magic angle - Wikipedia](https://wikipedia.org/wiki/Magic_angle)
"""

theta_m_bar: Final[float] = 0.152043361992348182457286110194392272
r"""
Magic number of turns \( \bar{\theta}_m = \theta_m / (2 \pi) \).

See also:
    - [`lmo.constants.theta_m`][lmo.constants.theta_m]
    - [Magic angle - Wikipedia](https://wikipedia.org/wiki/Magic_angle)
    - [Turn (angle) - Wikipedia](https://wikipedia.org/wiki/Turn_(angle))
"""
