#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from mpl_toolkits import mplot3d

from opt1 import blackBoxFunction

LOG_PATH = 'data/logs_.json'

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


    # input = np.log10(input)
    # input = np.log10(input[-10:,:])
    # output = np.log10(output)

    mu, sigma = posterior(opt, input, output, inp)

    np.append(inp, input, axis=0)

    ax = plt.axes()
    # ax = plt.axes(projection='3d')

    cntr = ax.tricontourf(inp[:,0], inp[:,1], mu, levels=30, cmap="RdBu_r")
    fig.colorbar(cntr, ax=ax)

    print(cntr.levels)

    # surf = ax.plot_trisurf(inp[:,0], inp[:,1], mu, linewidth=0.1, alpha=.25)
    # fig.colorbar(surf)
    # ax.scatter(inp[:,0], inp[:,1], mu, 'g')
    ax.scatter(input[10:,0], input[10:,1], c='b', s=1, alpha=.3)
    ax.scatter(input[:10,0], input[:10,1], c='k', s=20, alpha=.3)


    ax.scatter(opt.max['params']['log_nD'], opt.max['params']['log_nNe'], c='r', s=20)



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
