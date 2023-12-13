# pyright: reportUnknownMemberType=false
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lmo.distributions import genlambda

GALLERY_PATH = Path(__file__).resolve().parent
TEX_LABEL_TEMPLATE = r'$\beta = {},\ \delta = {}$'

X_MIN, X_MAX = -2, 2

if __name__ == '__main__':
    plt.style.use([
        'dark_background',
        'seaborn-v0_8-poster',
        GALLERY_PATH.parent / 'styles' / 'gallery.mplstyle',
    ])

    plt.figure(figsize=(16, 9), dpi=240)
    palette = mpl.color_sequences['Paired']

    b_denom = 2
    f = 0

    y_max = 0
    for b_numer, linestyle in zip([1, -1], ['-', '--']):
        for i, d in enumerate(range(-2, 3)):
            b = b_numer / b_denom
            xa, xb = genlambda.support(b, d, f)

            x = np.linspace(max(X_MIN, xa), min(X_MAX, xb), 1000)
            y = genlambda.pdf(x, b, d, f)
            y_max = max(y.max(), y_max)

            plt.plot(
                x,
                y,
                linestyle,
                label=TEX_LABEL_TEMPLATE.format(
                    f'{b_numer:+d}/{b_denom}',
                    f'{d:+d}',
                ).replace('+', r'\quad\;'),  # (\vphantom requires amsmath)
                alpha=0.8,
                color=palette[i]
            )

    plt.legend()
    plt.ylabel('$f(X)$', rotation=0)
    plt.xlabel('$X$')
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(0, 1.01 * y_max)
    plt.title(r'PDF of $X \sim \mathrm{GLD}(\beta, \delta, \phi = 0)$')
    plt.tight_layout()
    plt.savefig(
        GALLERY_PATH / 'genlambda.svg',
        transparent=True,
        format='svg',
        bbox_inches='tight'
    )

