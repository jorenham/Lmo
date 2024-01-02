# pyright: reportUnknownMemberType=false
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lmo.distributions import kumaraswamy

GALLERY_PATH = Path(__file__).resolve().parent
TEX_LABEL_TEMPLATE = r'$\alpha = {},\ \beta = {}$'

X_MIN, X_MAX = 0, 1
Y_MAX = 3.01

if __name__ == '__main__':
    plt.style.use([
        'dark_background',
        'seaborn-v0_8-poster',
        GALLERY_PATH.parent / 'styles' / 'gallery.mplstyle',
    ])

    plt.figure(figsize=(16, 9), dpi=240)
    palette = mpl.color_sequences['Paired']

    for i, (a, a_str, b, b_str) in enumerate([
        (1 / 2, '1/2', 1 / 2, '1/2'),
        (5, '5', 1, '1'),
        (1, '1', 3, '3'),
        (2, '2', 2, '2'),
        (2, '2', 5, '5'),
    ]):
        x = np.linspace(X_MIN, X_MAX, 1000)
        y = kumaraswamy.pdf(x, a, b)

        plt.plot(
            x,
            y,
            label=TEX_LABEL_TEMPLATE.format(f'{a_str}', f'{b_str}'),
            alpha=0.8,
            color=palette[i],
            )

    plt.legend()
    plt.ylabel('$f(X)$', rotation=0)
    plt.xlabel('$X$')
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(0, Y_MAX)
    plt.title(r'PDF of $X \sim \mathrm{Kumaraswamy}(\alpha, \beta)$')
    plt.tight_layout()
    plt.savefig(
        GALLERY_PATH / 'kumaraswamy.svg',
        transparent=True,
        format='svg',
        bbox_inches='tight'
    )

