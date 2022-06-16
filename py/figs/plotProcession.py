#!/usr/bin/env python3
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

import sklearn.gaussian_process as gp

import utils
import opt.objective as objective

BAYES_LOG_PATH  = '../opt/bayes/data/expDecay4.json'

N_SAMPLES = (100, 115, 130, 200)

NX, NY = 40, 40
XMIN, XMAX = 18, 22.2
YMIN, YMAX = 15, 20

LEVELS = np.linspace(-1.5, 2.4, 100)


XTEXT, YTEXT = 1.5e18, 3e19


def plot_bayes(ax, input, output):



    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(length_scale=[1., 1.], nu=2.5),
        # alpha=1e-6,
        normalize_y=True,

        n_restarts_optimizer=5,
        random_state=420,
    )
    reg.fit(input, output)

    x = np.logspace(XMIN, XMAX, NX)
    y = np.logspace(YMIN, YMAX, NY)

    xy = np.log10([[xx, yy] for xx in x for yy in y])
    mu = reg.predict(xy)

    nD  = 10 ** xy[:,0]
    nNe = 10 ** xy[:,1]

    cntr = ax.tricontourf(nD, nNe, np.log10(mu), levels=LEVELS, cmap=cc.cm.diverging_bwr_40_95_c42)

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

def main():

    with open(BAYES_LOG_PATH) as file:
        log = list(map(json.loads, file))

    input   = np.array([[sample['params']['log_nD'], sample['params']['log_nNe']] for sample in log], dtype=np.float32)
    output  = -1 * np.array([sample['target'] for sample in log], dtype=np.float32)

    global XMIN, XMAX, YMIN, YMAX
    XMIN, XMAX = input[:,0].min(), input[:,0].max()
    YMIN, YMAX = input[:,1].min(), input[:,1].max()

    print(output.min())


    utils.setFigureFonts()
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=utils.FIGSIZE_2X2, sharey=True, sharex=True)
    axs = [ax for row in axs for ax in row]

    for n_samples, ax, alph in zip(N_SAMPLES, axs, ('a', 'b', 'c', 'd')):

        x = input[:n_samples,:]
        y = output[:n_samples]

        cntr = plot_bayes(ax, x, y)
        ax.set_xlim(10**XMIN, 10**XMAX)
        ax.set_ylim(10**YMIN, 10**YMAX)

        ax.text(XTEXT, YTEXT, r'$\rm (' + alph +')$')
        # ax.set_title(r'$' + str(n_samples) + r'{\rm \:samples}$')

    # # colourbar settings
    ticks = np.linspace(-1, 2, 4)
    cbar_ax = fig.add_axes([.9, 0.2, utils.COLOURBAR_WIDTH, 0.7])
    cbar = fig.colorbar(cntr, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(r'$\mathcal{L}_1$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])

    # add text
    # x, y = 1.5e18, 3e19
    # ax1.text(x, y, r"${\rm (a)\,scan}+{\rm Powell's}$")
    # ax2.text(x, y, r"${\rm (b)\,BayesOpt}$")
    # ax1.set_title(r"${\rm scan}+{\rm Powell's\:method}$")
    # ax2.set_title(r"${\rm Bayesian\:optimization}$")

    fig.supylabel(r'$n_{\rm Ne}\;({\rm m}^{-3})$', x=.04, y=.55)
    fig.supxlabel(r'$n_{\rm D}\;({\rm m}^{-3})$', y=.04)
    plt.tight_layout()




    # axs[0].set_ylabel(r'$n_{\rm Ne}\;({\rm m}^{-3})$')
    # axs[2].set_ylabel(r'$n_{\rm Ne}\;({\rm m}^{-3})$')
    # axs[2].set_xlabel(r'$n_{\rm D}\;({\rm m}^{-3})$')
    # axs[3].set_xlabel(r'$n_{\rm D}\;({\rm m}^{-3})$')

    fig.subplots_adjust(wspace=.1, hspace=.05, right=.85)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
