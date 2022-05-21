#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

import opt.objective as obj
import utils


LOG_PATH = '../scans/data/expDecay2.log'

LOW, HIGH = -8, 2
LEVELS = np.linspace(LOW, HIGH, HIGH - LOW + 1, dtype=np.int32)

def plotScan(x, y, z, t, ax=None):

    if ax is None:
        ax = plt.axes()

    # plot current
    cntr = ax.tricontourf(x, y, np.log10(z), levels=LEVELS, cmap=cc.cm.diverging_bwr_40_95_c42)

    # show CQ time boundaries
    ax.tricontour(x, y, t, levels=[50e-3, 150e-3], linewidths=1, colors='k', linestyles=['-', '--'])

    ax.set_yscale('log')
    ax.set_xscale('log')
    return cntr

def main():

    nD, nNe, tCQ, I_re, I_ohm = [], [], [], [], []

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
            I_re.append(float(data[3]) * 1e-6)
            I_ohm.append(float(data[4]) * 1e-6)

    if len(nD) == 0:
        raise EOFError(f'No data in {LOG_PATH}')


    utils.setFigureFonts()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=utils.FIGSIZE_2X1, sharey=True)

    runaway = plot_scan(ax1, LEVELS)

    ohmic = plotScan(nD, nNe, I_ohm, tCQ, ax=ax2)



    cbar_ax = fig.add_axes([.9, 0.2, utils.COLOURBAR_WIDTH, 0.7])
    cbar = fig.colorbar(ohmic, cax=cbar_ax)
    cbar.ax.set_title(r'$I\,({\rm MA})$')

    levels = [lvl for lvl in LEVELS if lvl%2]
    cbar.ax.set_yticks(levels)
    cbar.ax.set_yticklabels(['$10^{' + str(lev) + '}$' for lev in levels])

    ax1.set_yticks([1e16, 1e18,1e20])
    ax1.set_xticks([1e18, 1e20, 1e22])
    ax2.set_xticks([1e18, 1e20, 1e22])

    ax1.set_ylabel(r'$n_{\rm Ne}\,({\rm m}^{-3})$')
    ax1.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')
    ax2.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')

    x, y = 1.5e18, 3e19

    ax1.text(x, y, r'${\rm (a)\;maximal\;RE\;current}$')
    ax2.text(x, y, r'${\rm (b)\;final\;Ohmic\;current}$')


    plt.tight_layout()
    plt.subplots_adjust(wspace=.1, right=.85)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
