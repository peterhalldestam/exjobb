#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt

from bayes_opt.util import load_logs
from bayes_opt import BayesianOptimization
import sklearn.gaussian_process as gp



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

BAYES_LOG_PATH  = 'data/expDecayTest_0.01.json'

# bounds = {'log_nD': (18, 22.2), 'log_nNe': (15, 20)}

NX, NY = 300, 300

def get_optimum(x, y, z):
    return x[z.argmin()], y[z.argmin()], z.min()


def plot_bayes(ax, select=False):

    with open(BAYES_LOG_PATH) as file:
        log = list(map(json.loads, file))

    input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log], dtype=np.float32)
    output  = np.array([sample['target'] for sample in log], dtype=np.float32)

    if select:
        select = output<1
        output = output[select]
        input = input[select]


    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=20,
        random_state=420,
    )
    reg.fit(input, output)

    xmin, xmax = input[:,0].min(), input[:,0].max()
    ymin, ymax = input[:,1].min(), input[:,1].max()
    x = np.linspace(xmin, xmax, NX)
    y = np.linspace(ymin, ymax, NY)

    xy = np.array([[xx, yy] for xx in x for yy in y])
    mu = reg.predict(xy)

    nD  = xy[:,0]
    nNe = xy[:,1]

    # sys.exit(print(mu))

    # ax.scatter(nD, nNe, mu, c='k', alpha=.1)
    cntr = ax.tricontourf(nD, nNe, mu, levels=40, cmap=cc.cm.diverging_bwr_40_95_c42)
    ax.scatter(input[:,0], input[:,1], output, c='r', s=10)


    print(get_optimum(input[:,0], input[:,1], output))
    print(get_optimum(input[:,0], input[:,1], -output))


    # ax.scatter(nD, nNe, c='k', s=1)


    # ax.scatter(nD_, nNe_, c='r', marker='*', s=60)

    # ax.set_yscale('log')
    # ax.set_xscale('log')
    #
    # ax.set_xticks([1e18, 1e20, 1e22])
    # ax.set_yticks([1e16, 1e18, 1e20])
    # ax.set_xticklabels([r'$10^{18}$', r'$10^{20}$', r'$10^{22}$'])
    # ax.set_yticklabels([r'$10^{16}$', r'$10^{18}$', r'$10^{20}$'])



    return cntr


def main():

    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')



    # levels = np.linspace(-1.5, 2.3, 100)

    cntr2 = plot_bayes(ax)



    # colourbar settings
    ticks = np.linspace(-1, 4, 6)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.7])
    cbar = fig.colorbar(cntr2, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(r'$\mathcal{L}$')
    # cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
