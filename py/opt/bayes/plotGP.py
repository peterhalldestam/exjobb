#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from mpl_toolkits import mplot3d
import colorcet as cc

from opt1 import blackBoxFunction

LOG_PATH = 'data/expDecay4.json'

bounds = {'log_nD': (1e19, 2e22), 'log_nNe': (1e15, 1e19)}

NX, NY = 30, 30

def posterior(opt, input, output, grid):
    opt.set_gp_params(optimizer=None)
    opt._gp.fit(input, output)
    mu, sigma = opt._gp.predict(grid, return_std=True)
    return mu, sigma



def plot_gp(opt, inp):
    fig = plt.figure()
    steps = len(opt.space)
    fig.suptitle(
        'Gaussian Process After {} Steps'.format(steps),
        fontdict={'size':30}
    )

    input = np.array([[res['params']['log_nD'], res['params']['log_nNe']] for res in opt.res])
    output = -np.array([res["target"] for res in opt.res])
    
    #input = input[:110]
    #output = output[:110]
    #########
   # output[output==1e6] = 200.
    #########


    # input = np.log10(input)
    # input = np.log10(input[-10:,:])
    # output = np.log10(output)

    mu, sigma = posterior(opt, input, output, inp)



    np.append(inp, input, axis=0)
    input = 10**input
    inp = 10**inp

    ax = plt.axes()
    # ax = plt.axes(projection='3d')


    cntr = ax.tricontourf(inp[:,0], inp[:,1], np.log10(mu), levels=30, cmap="RdBu_r")
    fig.colorbar(cntr, ax=ax, label='Objective function')

    print(cntr.levels)

    # surf = ax.plot_trisurf(inp[:,0], inp[:,1], mu, linewidth=0.1, alpha=.25)
    # fig.colorbar(surf)
    # ax.scatter(inp[:,0], inp[:,1], mu, 'g')

    ax.scatter(input[:10,0], input[:10,1], c='k', s=10, alpha=.3, label='Initial random samples')
    ax.scatter(input[10:,0], input[10:,1], c='k', s=2, alpha=.4, label='From acquisition function')



    print(opt.max)
    ax.scatter(10**opt.max['params']['log_nD'], 10**opt.max['params']['log_nNe'], c='y', s=30, marker='*', label='Optimal input parameters')

    ax.scatter(1e20, 2e18, c='r', s=20, marker='*', label='Bad input parameters')

    ax.set_xlabel('injected deuterium $(m^{-3})$')
    ax.set_ylabel('injected neon $(m^{-3})$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='lower left')
    plt.show()

def main():

    opt = BayesianOptimization(
        f=blackBoxFunction,
        pbounds=bounds,
        random_state=1
    )

    load_logs(opt, logs=LOG_PATH)
    print(opt.max)
    input = np.array([[res['params']['log_nD'], res['params']['log_nNe']] for res in opt.res])

    # opt._gp.fit(input_data, output_data)
    xmin, xmax = input[:,0].min(), input[:,0].max()
    ymin, ymax = input[:,1].min(), input[:,1].max()
    x = np.logspace(xmin, xmax, NX)
    y = np.logspace(ymin, ymax, NY)
    xy = np.log10(np.array([[xx, yy] for xx in x for yy in y]))

    plot_gp(opt, xy)

if __name__ == '__main__':
    sys.exit(main())
