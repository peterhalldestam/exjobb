#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from mpl_toolkits import mplot3d

import sklearn.gaussian_process as gp

import utils
import opt.objective as objective

LOG_PATH = '../opt/bayes/data/expDecay4.json'


NX, NY = 100, 100

def plot_bayes(log, ax, lvls):

    # create GP regressor
    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(length_scale=[1., 1.], nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=20,
        random_state=420,
    )

    # train the GP on sampled data
    input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log])
    output  = np.log10([sample['target'] for sample in log])

    N = 93
    input = input[:N,:]
    output = output[:N]

    input = input[output<2]
    output = output[output<2]

    # sys.exit(print(output))
    # print(output.max())

    reg.fit(input, output)

    # get posterior mean function on a grid of points
    xmin, xmax = input[:,0].min(), input[:,0].max()
    ymin, ymax = input[:,1].min(), input[:,1].max()
    x = np.logspace(xmin, xmax, NX)
    y = np.logspace(ymin, ymax, NY)
    xy = np.log10([[xx, yy] for xx in x for yy in y])
    mu = 10 ** reg.predict(xy)
    nD  = 10 ** (xy[:,0] - 21)
    nNe = 10 ** (xy[:,1] - 19)
    # _, _, obj = utils.get_optimum(input[:,0], input[:,1], -output)
    mu = (mu - mu.min()) / (mu.max() - mu.min())
    cntr = ax.tricontourf(nD, nNe, mu, levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)
    ax.set_xlim(nD.min(), nD.max())
    ax.set_ylim(nNe.min(), nNe.max())

    output = 10 ** output

    # show each sampled point
    nD  = 10 ** (input[:,0] - 21)
    nNe = 10 ** (input[:,1] - 19)
    ax.scatter(nD, nNe, c='k', s=10)

    # indicate minimum
    ax.scatter(nD[0], nNe[0], c='r', marker='*', s=100)

    # indicate maximum
    nD, nNe, obj = utils.get_optimum(nD, nNe, -output)
    ax.scatter(nD, nNe, c='b', marker='*', s=100)
    print(-obj)

    ax.set_xlabel(r'$n_{\rm D}\;(10^{21}\,{\rm m}^{-3})$')
    ax.set_ylabel(r'$n_{\rm Ne}\;(10^{19}\,{\rm m}^{-3})$')

    return cntr

# def plot_bayes(ax, lvls):
#
#     with open(BAYES_LOG_PATH) as file:
#         log = list(map(json.loads, file))
#
#     input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log], dtype=np.float32)
#     output  = -1 * np.array([sample['target'] for sample in log], dtype=np.float32)
#
#     input = input[:200,:]
#     output = output[:200]
#
#     reg = gp.GaussianProcessRegressor(
#         kernel=gp.kernels.Matern(length_scale=[1., 1.], nu=2.5),
#         # alpha=1e-6,
#         normalize_y=True,
#         n_restarts_optimizer=5,
#         random_state=420,
#     )
#     reg.fit(input, output)
#
#     xmin, xmax = input[:,0].min(), input[:,0].max()
#     ymin, ymax = input[:,1].min(), input[:,1].max()
#     x = np.logspace(xmin, xmax, NX)
#     y = np.logspace(ymin, ymax, NY)
#
#     xy = np.log10([[xx, yy] for xx in x for yy in y])
#     mu = reg.predict(xy)
#
#     nD  = 10 ** xy[:,0]
#     nNe = 10 ** xy[:,1]
#
#     cntr = ax.tricontourf(nD, nNe, np.log10(mu), levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)
#
#     # add each sampled point
#     nD  = 10 ** input[:,0]
#     nNe = 10 ** input[:,1]
#
#     ax.scatter(nD, nNe, c='k', s=1)
#
#     nD_, nNe_, obj_ = get_optimum(10**input[:,0], 10**input[:,1], output)
#     ax.scatter(nD_, nNe_, c='r', marker='*', s=60)
#
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#
#     ax.set_xticks([1e18, 1e20, 1e22])
#     ax.set_yticks([1e16, 1e18, 1e20])
#     ax.set_xticklabels([r'$10^{18}$', r'$10^{20}$', r'$10^{22}$'])
#     ax.set_yticklabels([r'$10^{16}$', r'$10^{18}$', r'$10^{20}$'])
#
#
#
#     return cntr



def main():

    # create figure with 4 subplots
    utils.setFigureFonts()
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=utils.FIGSIZE_2X2)
    axs = [ax for row in axs for ax in row]

    levels = 100#np.logspace(.7, 1.9, 100)

    with open(LOG_PATH) as file:
        log = list(map(json.loads, file))


    for n in (10, 50, 100, 150):
        cntr = plot_bayes(log, ax, levels)






    # colourbar settings
    # ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([.9, 0.2, utils.COLOURBAR_WIDTH, 0.7])
    cbar = fig.colorbar(cntr, cax=cbar_ax)
    cbar.ax.set_title(r'$\widetilde{\mathcal{L}}_2$')
    cbar.ax.set_yticks([0, .25, .5, .75, 1.])
    #
    # # add text
    # x, y = 1.5e18, 3e19
    # ax1.text(x, y, r"${\rm (a)\,scan}+{\rm Powell's}$")
    # ax2.text(x, y, r"${\rm (b)\,BayesOpt}$")
    #
    #
    # ax1.set_ylabel(r'$n_{\rm Ne}\,({\rm m}^{-3})$')
    # ax1.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')
    # ax2.set_xlabel(r'$n_{\rm D}\,({\rm m}^{-3})$')




    plt.tight_layout()
    fig.subplots_adjust(right=.85)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
