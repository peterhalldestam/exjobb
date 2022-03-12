#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt



SHOW_POINTS = True
LOG_PATH = 'scan2.log'


def main():

    nD, nNe, tCQ, I_re = [], [], [], []

    # Set up data log
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as fp:
            while line := fp.readline():
                data = line.rstrip().replace(',', '').split()[-5:]

                nNe.append(float(data[0]))
                nD.append(float(data[1]))

                tCQ.append(np.inf if data[3] == 'inf' else float(data[3]))
                I_re.append(float(data[4]) * 1e-6)


    fig, ax = plt.subplots()

    ax.tricontour(nD, nNe, I_re, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(nD, nNe, I_re, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax)



    yesCQ = np.isfinite(tCQ)

    nD_, nNe_, tCQ_ = [], [], []
    for t, n1, n2 in zip(tCQ, nD, nNe):
        if np.isfinite(t):
            nD_.append(n1)
            nNe_.append(n2)
            tCQ_.append(t)

    print(nD_, nNe_, tCQ_)
    ax.tricontour(nD_, nNe_, tCQ_, levels=[50e-3, 150e-3], linewidths=3, colors='k')

    if SHOW_POINTS:
        for t, n1, n2 in zip(tCQ, nD, nNe):
            if np.isinf(t):
                pass
                # ax.plot(n1, n2, 'ko', ms=3)
            elif :
                ax.plot(n1, n2, 'ko', ms=2, alpha=.5)



    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
                # sys.exit()

    return 0

if __name__ == '__main__':
    sys.exit(main())
