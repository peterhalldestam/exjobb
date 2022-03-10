#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt


LOG_PATH = 'scan.log'


def main():

    nNe, nD, tCQ, I_re = [], [], [], []

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
    for t, n1, n2 in zip(tCQ, nD, nNe):
        if not np.isinf(t):
            ax.plot(n1, n2, 'ko', ms=3)
# ax2.set(xlim=(-2, 2), ylim=(-2, 2))
# ax2.set_title('tricontour (%d points)' % npts)

# plt.subplots_adjust(hspace=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
                # sys.exit()

    return 0

if __name__ == '__main__':
    sys.exit(main())
