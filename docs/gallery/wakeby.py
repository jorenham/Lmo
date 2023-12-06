# pyright: reportUnknownMemberType=false
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lmo.distributions import wakeby

PARAMS = [
    (5, 1, 0.5),
    (5, 1, 0.8),
    (5, -0.5, 0.5),
    # (-0.5, 1, 0.5),
]
LABEL_TEMPLATE = r'$\beta={:.1f}, \delta={:.1f}, \phi={:.1f}$'
GALLERY_PATH = Path(__file__).resolve().parent

X_MAX = 2
TEXT_COLOR = 226/255, 228/255, 233/255, 0.82

if __name__ == '__main__':

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 16,
        'text.color': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    plt.figure(figsize=(16, 9), dpi=240)

    y_max = 0
    for i, (b, d, f) in enumerate(PARAMS):
        x = np.linspace(0, min(wakeby.support(b, d, f)[1] - 1e-3, X_MAX), 1000)
        y = wakeby.pdf(x, b, d, f)
        y_max = max(y.max(), y_max)
        plt.plot(x, y, label=LABEL_TEMPLATE.format(b, d, f))
        plt.fill_between(x, y, alpha=0.1)

    plt.legend()
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.xlim(0, X_MAX)
    plt.ylim(0, 1.01 * y_max)
    plt.title('Estimated PDF of some standard Wakeby distributions')
    plt.tight_layout()
    plt.savefig(
        GALLERY_PATH / 'wakeby.svg',
        transparent=True,
        format='svg',
        bbox_inches='tight'
    )

