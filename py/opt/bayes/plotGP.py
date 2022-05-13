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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from opt1 import blackBoxFunction
from opt.objective import CRITICAL_RE_CURRENT, CRITICAL_OHMIC_CURRENT
from opt.objective import CQ_TIME_MIN, CQ_TIME_MAX, SLOPE_LEFT, SLOPE_RIGHT, CQ_WEIGHT

plt.rcParams['text.usetex'] = True

SMALL_SIZE = 15
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

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

def get_optimum(x, y, z):
    return x[z.argmin()], y[z.argmin()], z.min()

def plot_scan(ax, lvls, bounds=None):

    with open(SCAN_LOG_PATH) as file:
        log = np.loadtxt(file, skiprows=1, dtype=np.str_)[:,-1]

    # Load the entire log file
    nNe, nD, tCQ, I_re, I_ohm = np.array([sample.split(',')[1:] for sample in log], dtype=np.float32).T
    obj = baseObjective(I_re, I_ohm, tCQ)

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

    ax.scatter(nD, nNe, c='k', s=1)

    nD_, nNe_, obj_ = get_optimum(10**input[:,0], 10**input[:,1], output)
    ax.scatter(nD_, nNe_, c='r', marker='*', s=60)

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
    ax.scatter(nD[0], nNe[0], c='c', s=30, zorder=1)
    ax.scatter(nD, nNe, c='k', s=1, zorder=1)

    nD_, nNe_, obj_ = get_optimum(nD, nNe, output)
    ax.scatter(nD_, nNe_, c='r', marker='*', s=60)
    print(get_optimum(nD, nNe, output))

def main():

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))


    levels = np.linspace(-1.5, 2.3, 100)

    # plot scan + optimizations
    cntr1 = plot_scan(ax1, levels)
    cntr2 = plot_bayes(ax2, levels)
    plot_powell(ax1)



    # colourbar settings
    ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.7])
    cbar = fig.colorbar(cntr1, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(r'$\mathcal{L}$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])

    # add text
    ax1.text(1.5e18, 4e19, r"${\rm (a)\,scan}+{\rm Powell's\;method}$")
    ax2.text(1.5e18, 4e19, r"${\rm (b)\,Bayesian\;optimization}$")
    #

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
