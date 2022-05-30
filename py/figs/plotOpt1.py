#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

import sklearn.gaussian_process as gp

import utils
import opt.objective as objective

SCAN_LOG_PATH   = '../scans/data/expDecay2.log'
BAYES_LOG_PATH  = '../opt/bayes/data/expDecay4.json'
POWELL_LOG_PATH = '../opt/powell/data/log_expDecay.json'

bounds = {'log_nD': (18, 22.2), 'log_nNe': (15, 20)}

NX, NY = 40, 40


def plot_scan(ax, lvls, bounds=None):

    with open(SCAN_LOG_PATH) as file:
        log = np.loadtxt(file, skiprows=1, dtype=np.str_)[:,-1]

    # Load the entire log file
    nNe, nD, tCQ, I_re, I_ohm = np.array([sample.split(',')[1:] for sample in log], dtype=np.float32).T
    obj = objective._baseObjective(I_re, I_ohm, tCQ)

    cntr = ax.tricontourf(nD, nNe, np.log10(obj), levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1e18, 1e20, 1e22])
    ax.set_yticks([1e16, 1e18, 1e20])
    ax.set_xticklabels([r'$10^{18}$', r'$10^{20}$', r'$10^{22}$'])
    ax.set_yticklabels([r'$10^{16}$', r'$10^{18}$', r'$10^{20}$'])
    return cntr



def plot_bayes(ax, lvls):

    with open(BAYES_LOG_PATH) as file:
        log = list(map(json.loads, file))

    input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log], dtype=np.float32)
    output  = -1 * np.array([sample['target'] for sample in log], dtype=np.float32)

    input = input[:200,:]
    output = output[:200]

    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(length_scale=[1., 1.], nu=2.5),
        # alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=420,
    )
    reg.fit(input, output)

    xmin, xmax = input[:,0].min(), input[:,0].max()
    ymin, ymax = input[:,1].min(), input[:,1].max()
    x = np.logspace(xmin, xmax, NX)
    y = np.logspace(ymin, ymax, NY)

    xy = np.log10([[xx, yy] for xx in x for yy in y])
    mu = reg.predict(xy)

    nD  = 10 ** xy[:,0]
    nNe = 10 ** xy[:,1]

    cntr = ax.tricontourf(nD, nNe, np.log10(mu), levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)

    # add each sampled point
    nD  = 10 ** input[:,0]
    nNe = 10 ** input[:,1]

    ax.scatter(nD, nNe, c='k', s=1)

    nD_, nNe_, obj_ = utils.get_optimum(10**input[:,0], 10**input[:,1], output)
    ax.scatter(nD_, nNe_, c='r', marker='*', s=60)
    print(utils.get_optimum(input[:,0], input[:,1], output))


    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xticks([1e18, 1e20, 1e22])
    ax.set_yticks([1e16, 1e18, 1e20])
    ax.set_xticklabels([r'$10^{18}$', r'$10^{20}$', r'$10^{22}$'])
    ax.set_yticklabels([r'$10^{16}$', r'$10^{18}$', r'$10^{20}$'])



    return cntr

def plot_powell(ax):

    with open(POWELL_LOG_PATH) as file:
        log = json.load(file)

    input = np.array(log['P'])
    nD = input[:,0]
    nNe = input[:,1]
    ax.plot(nD, nNe, 'c', lw=2, zorder=1)

    input = np.array(log['allP'])
    output = np.array(log['allFun'])
    nD = input[:,0]
    nNe = input[:,1]
    ax.scatter(nD[0], nNe[0], marker='p', c='c', s=100, zorder=1)
    ax.scatter(nD, nNe, c='k', s=1, zorder=1)

    nD_, nNe_, obj_ = utils.get_optimum(nD, nNe, output)
    ax.scatter(nD_, nNe_, c='r', marker='*', s=100)
    print(utils.get_optimum(nD, nNe, output))

def main():


    utils.setFigureFonts()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=utils.FIGSIZE_2X1, sharey=True)


    levels = np.linspace(-1.5, 2.4, 100)

    # plot scan + optimizations
    cntr1 = plot_scan(ax1, levels)
    cntr2 = plot_bayes(ax2, levels)
    plot_powell(ax1)



    # colourbar settings
    ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([.9, 0.2, utils.COLOURBAR_WIDTH, 0.7])
    cbar = fig.colorbar(cntr1, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(r'$\mathcal{L}_1$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])

    # add text
    x, y = 1.5e18, 3e19
    ax1.text(x, y, r"${\rm (a)\,scan}+{\rm Powell's}$")
    ax2.text(x, y, r"${\rm (b)\,BayesOpt}$")


    ax1.set_ylabel(r'$n_{\rm Ne}\,({\rm m}^{-3})$')
    ax1.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')
    ax2.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')


    plt.tight_layout()
    fig.subplots_adjust(wspace=.1, right=.85)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
