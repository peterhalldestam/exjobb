#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

import alphashape


SHOW_BORDER = False
SHOW_POINTS = True #False
LOG_PATH = 'scan_zoomed2.log'


def main():

    nD, nNe, tCQ, I_re = [], [], [], []
    nD_, nNe_, tCQ_ = [], [], []

    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f'The log file {LOG_PATH} does not exist!')

    # Set up data log
    with open(LOG_PATH) as fp:
        fp.readline()
        while line := fp.readline():
            data = line.rstrip().replace(',', ' ').split()[-5:]

            nNe.append(float(data[0]))
            nD.append(float(data[1]))
            tCQ.append(np.inf if data[2]=='inf' else float(data[2]))
            I_re.append(float(data[3]))

            if not data[2] == 'inf':
                nNe_.append(float(data[0]))
                nD_.append(float(data[1]))
                tCQ_.append(float(data[2]))


    if len(nD) == 0:
        raise EOFError(f'No data in {LOG_PATH}')

    #print(np.array(nD).reshape((20,20)))
    #print(nNe)
    #print(np.array(nNe).reshape((20,20)))
    #print(np.array(I_re).reshape((20,20)))

    fig, ax = plt.subplots()

#    fig2, ax2 = plt.subplots()

    # Current plot
    ax.tricontour(nD, nNe, I_re, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(nD, nNe, I_re, levels=14, cmap="RdBu_r")
#    cntr2 = ax2.contourf(np.array(nD).reshape((20,20)), np.array(nNe).reshape((20,20)), np.log10(np.array(I_re).reshape((20,20))), levels=14, cmap="RdBu_r")
#    ax2.contour(np.array(nD).reshape((20,20)), np.array(nNe).reshape((20,20)), np.log10(np.array(I_re).reshape((20,20))), levels=14, linewidths=0.5, colors='k')
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
