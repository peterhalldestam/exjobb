#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

import alphashape


SHOW_BORDER = False
SHOW_POINTS = False
LOG_PATH = 'scan2.log'


def main():

    nD, nNe, I_re = [], [], []
    nD_, nNe_, tCQ_ = [], [], []

    # Set up data log
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as fp:
            while line := fp.readline():
                data = line.rstrip().replace(',', '').split()[-5:]

                nNe.append(float(data[0]))
                nD.append(float(data[1]))
                I_re.append(float(data[4]))

                if not data[3] == 'inf':
                    nNe_.append(float(data[0]))
                    nD_.append(float(data[1]))
                    tCQ_.append(float(data[3]))


    fig, ax = plt.subplots()

    # Current plot
    ax.tricontour(nD, nNe, I_re, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(nD, nNe, I_re, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax)

    # Current quench time plot
    ax.tricontour(nD_, nNe_, tCQ_, levels=[50e-3, 150e-3], linewidths=2, linestyles=['dashed', 'dotted'])


    if SHOW_BORDER:
        points = [(n1, n2) for n1, n2 in zip(nD_, nNe_)]
        alpha = 0.95 * alphashape.optimizealpha(points)
        hull = alphashape.alphashape(points, alpha)
        border = hull.exterior.coords.xy

        nD_inf, nNe_inf = [], []
        for n1, n2 in zip(border[0], border[1]):
            print(n1, n2)
            if n1 < 1.5e22 and n2 < 8e18:
                nD_inf.append(n1)
                nNe_inf.append(n2)

        ax.plot(nD_inf[:-1], nNe_inf[:-1], 'r')

    if SHOW_POINTS:
        for t, n1, n2 in zip(tCQ, nD, nNe):
            if np.isinf(t):
                ax.plot(n1, n2, 'ro', ms=3)
            else:
                ax.plot(n1, n2, 'ko', ms=2, alpha=.5)



    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
                # sys.exit()

    return 0

if __name__ == '__main__':
    sys.exit(main())
