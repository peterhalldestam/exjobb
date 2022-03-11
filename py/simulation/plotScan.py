#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt


LOG_PATH = 'scan1.log'

CURRENT_QUENCH_TIME_MAX = 150e-3
CURRENT_QUENCH_TIME_MIN = 50e-3

def main():

    nNe, nD2, tCQ, I_re = [], [], [], []

    # Set up data log
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as fp:
            while line := fp.readline():
                data = line.rstrip().replace(',', '').split()[-5:]

                nNe.append(float(data[0]))
                nD2.append(float(data[1]))
                I_re.append(float(data[4]) * 1e-6)
                tCQ.append(np.inf if data[3] == 'inf' else float(data[3]))


    fig, ax = plt.subplots()
    ax.tricontour(nD2, nNe, I_re, levels=14, linewidths=0.5, colors='k')

    ax.tricontour(nD2, nNe, I_re, levels=3, linewidths=2, colors='k', linestyles=['dashed', 'dotted'])

    cntr2 = ax.tricontourf(nD2, nNe, I_re, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax)

    plt.legend([f'{CURRENT_QUENCH_TIME_MAX * 1e3} ms (dotted)', f'{CURRENT_QUENCH_TIME_MIN * 1e3} ms (dashed)'], title='CQ time')

    ax.set_ylabel(r'$n_{Ne}$ [$10^{20}$ m$^{-3}$]')
    ax.set_xlabel(r'$n_{D}$ [$10^{20}$ m$^{-3}$]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
                # sys.exit()

    return 0

if __name__ == '__main__':
    sys.exit(main())
