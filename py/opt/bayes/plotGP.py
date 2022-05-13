#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from mpl_toolkits import mplot3d
import colorcet as cc

import sklearn.gaussian_process as gp


from opt1 import blackBoxFunction
from opt.objective import CRITICAL_RE_CURRENT, CRITICAL_OHMIC_CURRENT
from opt.objective import CQ_TIME_MIN, CQ_TIME_MAX, SLOPE_LEFT, SLOPE_RIGHT, CQ_WEIGHT

plt.rcParams['text.usetex'] = True

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


SCAN_LOG_PATH   = '../../scans/data/expDecay2.log'
BAYES_LOG_PATH  = 'data/expDecay4.json'
POWELL_LOG_PATH = '../powell/data/log_expDecay.json'

bounds = {'log_nD': (18, 22.2), 'log_nNe': (15, 20)}

NX, NY = 40, 40

def sigmoid(x, x0=0, k=1):
    """
    Hyperbolic tangent sigmoid function.
    """
    return 1/2 + 1/2*np.tanh(k*(x-x0))


def baseObjective(I_re, I_ohm, tCQ):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM.
    """
    obj1 = I_re / CRITICAL_RE_CURRENT
    obj2 = I_ohm / CRITICAL_OHMIC_CURRENT
    obj3 = sigmoid(-tCQ, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(tCQ, CQ_TIME_MAX, SLOPE_RIGHT)
    return obj1 + obj2 + CQ_WEIGHT * (obj3 + obj4)


def plot_scan(ax, lvls):

    with open(SCAN_LOG_PATH) as fp:
        log = np.loadtxt(fp, skiprows=1, dtype=np.str_)[:,-1]

    nNe, nD, tCQ, I_re, I_ohm = np.array([sample.split(',')[1:] for sample in log], dtype=np.float32).T
    obj = baseObjective(I_re, I_ohm, tCQ)

    cntr = ax.tricontourf(nD, nNe, np.log10(obj), levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)

    ax.set_yscale('log')
    ax.set_xscale('log')

    return cntr



def plot_bayes(ax, lvls):

    with open(BAYES_LOG_PATH) as fp:
        log = list(map(json.loads, fp))

    input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log], dtype=np.float32)
    output  = -1 * np.array([sample['target'] for sample in log], dtype=np.float32)

    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(nu=2.5),
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

    ax.scatter(nD, nNe, c='k', s=.1)

    ax.set_yscale('log')
    ax.set_xscale('log')

    print(output.min(), mu.min())

    return cntr

def plot_powell(ax):

    with open(POWELL_LOG_PATH) as fp:
        log = list(map(json.loads, f))

    print(log)


def main():

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

    sys.exit(plot_powell(ax2))

    levels = np.linspace(-1.2, 2.25, 100)

    cntr1 = plot_scan(ax1, levels)
    cntr2 = plot_bayes(ax2, levels)

    ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.7])
    cbar = fig.colorbar(cntr1, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(r'$\mathcal{L}$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])

    ax1.set_ylabel(r'$n_{\rm Ne}\, ({\rm m}^{-3})$')
    ax1.set_xlabel(r'$n_{\rm D}\, ({\rm m}^{-3})$')
    ax2.set_xlabel(r'$n_{\rm D}\, ({\rm m}^{-3})$')

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
