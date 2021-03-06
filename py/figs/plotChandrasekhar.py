#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import erf

import utils

def erfDer(x):
    ''' Derivative of the error function '''
    return 2 * np.exp(-x**2) / np.sqrt(np.pi)

def chskr(x):
    ''' Chandrasekhar function '''
    return (erf(x) - x * erfDer(x)) / (2 * x**2)

def main():


    utils.setFigureFonts()
    fig, ax = plt.subplots(figsize=utils.FIGSIZE_1X1)



    x0, x1 = .01, 8
    x = np.linspace(x0, x1, 1000)
    ax.plot(x, (1/x**2) * chskr(x1) * x1**2, 'grey')
    ax.plot(x, x * chskr(x0) / x0, 'grey')
    ax.plot(x, chskr(x), 'k')
    ax.plot([0, x0], [0, chskr(x0)], 'k')

    # show critical velocity + eE_parallel
    xc1 = 3
    ax.plot([0, xc1], [chskr(xc1), chskr(xc1)], '--', c='k')
    ax.plot([xc1, xc1], [0, chskr(xc1)], '--', c='k')

    # show critical field E_c
    xc2 = 8
    ax.plot([0, xc2], [chskr(xc2)/2, chskr(xc2)/2], '--', c='k')
    # ax.plot([xc2, xc2], [0, chskr(xc2)+.006], c='k')

    # fill in runaway region
    x_re = x[(xc1<x)*(x<xc2)]
    ax.fill_between(x_re, chskr(x_re), np.zeros(x_re.shape), color='bisque')
    # ax.text(xc1+.25, .011, r'${\rm runaway\;region}$')

    # show thermal velocity + 0.21eE_D
    ax.plot([0, 1], [chskr(1), chskr(1)], '--', c='k')
    ax.plot([1, 1], [0, chskr(1)], '--', c='k')



    ax.set_xlim(0, xc2+.1)
    ax.set_ylim(0, .25)

    ax.set_xticks([1, xc1])
    ax.set_xticklabels([r'$v_{\rm th}$', r'$v_{\rm c}$'])
    ax.set_yticks([chskr(xc1), chskr(xc2)/2, chskr(1)])
    ax.set_yticklabels([r'$eE_\parallel$', r'$eE_{\rm c}$', r'$0.21eE_{\rm D}$'])

    # plt.xlabel(r'$v$')
    # plt.ylabel(r'$F_{drag}$')
    ax.set_xlabel(r'${\rm velocity}$')
    ax.set_ylabel(r'${\rm Friction\;force}$')
    ax.yaxis.set_label_coords(-.08, .5)


    ax.text(.75, .24, r'$\propto v$')
    ax.text(1.9, .15, r'$\propto \displaystyle\frac{ 1}{ v^2}$')

    ax.text(4.2, .04, r'$\rm runaway\,region$')

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
