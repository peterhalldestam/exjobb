#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

import utils

SCALE = .1

def main():


    utils.setFigureFonts()
    fig, ax = plt.subplots(figsize=utils.FIGSIZE_1X1)

    x = np.linspace(0, 5, 100)

    xc = 2.5
    # ax.plot([0, xc], [maxwell.pdf(xc), maxwell.pdf(xc)], '--', c='k')
    ax.plot([xc, xc], [0, SCALE * maxwell.pdf(xc)], '--', c='k')

    # fill in runaway region
    x_re = x[x>xc]
    ax.fill_between(x_re, SCALE * maxwell.pdf(x_re), np.zeros(x_re.shape), color='bisque')


    ax.plot(x, SCALE * maxwell.pdf(x), 'k')


    ax.set_yticks([])
    ax.set_xticks([xc])
    ax.set_xticklabels([r'$v_c$'])

    ax.set_xlabel(r'${\rm Electron\:speed}$')
    ax.set_ylabel(r'${\rm Distrbution\:function}$')
    # ax.yaxis.set_label_coords(-.08, .5)


    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=6, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=6, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
