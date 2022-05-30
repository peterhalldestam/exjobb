#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from mpl_toolkits import mplot3d

import sklearn.gaussian_process as gp

import utils

LOG_PATH = '../sens/data/expDecay_0.1.json'
# LOG_PATH = '../opt/bayes/data/expDecay4.json'

NX, NY = 100, 100
XOFF, YOFF = 22, 16

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

    input = np.insert(input, 0, [np.log10([1.1652767024506405e22, 6.99511456882493e16])], axis=0)
    output = np.insert(output, 0, np.log10(0.07803045))

    # N = 70
    # input = input[:N,:]
    # output = output[:N]

    # input = input[output<2]
    # output = output[output<2]

    # sys.exit(print(output))
    # print(output.max())

    reg.fit(input, output)

    # get posterior mean function on a grid of points
    xmin, xmax = input[:,0].min(), input[:,0].max()
    ymin, ymax = input[:,1].min(), input[:,1].max()
    x = np.logspace(xmin, xmax, NX)
    y = np.logspace(ymin, ymax, NY)
    xy = np.log10([[xx, yy] for xx in x for yy in y])
    mu = reg.predict(xy)
    nD  = 10 ** (xy[:,0] - XOFF)
    nNe = 10 ** (xy[:,1] - YOFF)
    # _, _, obj = utils.get_optimum(input[:,0], input[:,1], -output)
    # mu = (mu - mu.min()) / (mu.max() - mu.min())
    cntr = ax.tricontourf(nD, nNe, mu, levels=lvls, cmap=cc.cm.diverging_bwr_40_95_c42)
    ax.set_xlim(nD.min(), nD.max())
    ax.set_ylim(nNe.min(), nNe.max())

    output =  output

    # show each sampled point
    nD  = 10 ** (input[:,0] - XOFF)
    nNe = 10 ** (input[:,1] - YOFF)
    ax.scatter(nD, nNe, c='k', s=5)

    # indicate minimum
    ax.scatter(nD[0], nNe[0], c='r', marker='*', s=100)

    # indicate maximum
    nD, nNe, obj = utils.get_optimum(nD, nNe, -output)
    ax.scatter(nD, nNe, c='b', marker='*', s=100)

    return cntr


def main():

    # create figure with 4 subplots
    utils.setFigureFonts()
    fig, ax = plt.subplots(figsize=utils.FIGSIZE_1X1)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    levels = np.linspace(-2.1, 1.1, 100)

    with open(LOG_PATH) as file:
        log = list(map(json.loads, file))

    cntr = plot_bayes(log, ax, levels)






    # colourbar settings
    ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([.9, 0.2, utils.COLOURBAR_WIDTH, 0.7])
    cbar = fig.colorbar(cntr, cax=cbar_ax)
    cbar.ax.set_title(r'$\mathcal{L}_1$')
    cbar.ax.set_yticks([-2, -1, 0, .9999])
    cbar.ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
    #
    # # add text
    # x, y = 1.5e18, 3e19
    # ax1.text(x, y, r"${\rm (a)\,scan}+{\rm Powell's}$")
    # ax2.text(x, y, r"${\rm (b)\,BayesOpt}$")
    #
    #
    ax.set_ylabel(r'$n_{\rm Ne}\,(10^{' + str(XOFF) + r'}\;{\rm m}^{-3})$')
    ax.set_xlabel(r'$n_{\rm D}\,(10^{' + str(YOFF) + r'}\;{\rm m}^{-3})$')


    plt.tight_layout()
    fig.subplots_adjust(right=.85)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
