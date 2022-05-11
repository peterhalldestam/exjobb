#!/usr/bin/env python3
import os, sys, glob
import numpy as np
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import colorcet as cc

plt.rcParams['text.usetex'] = True

SMALL_SIZE = 17
MEDIUM_SIZE = 19
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from mpl_toolkits import mplot3d
from bayes.opt1 import blackBoxFunction

BAYESIAN_PATH = 'bayes/data/transportData'
POWELL_PATH = 'powell/data/transportData'
N_STEPS = 110
FIG_NAME = None #'transport_opt.eps'

bounds = {'log_nD': (1e19, 2e22), 'log_nNe': (1e15, 1e19)}

NX, NY = 30, 30

def posterior(opt, input, output, grid):
    opt.set_gp_params(optimizer=None)
    opt._gp.fit(input, output)
    mu, sigma = opt._gp.predict(grid, return_std=True)
    return mu, sigma



def plot_gp(opt, inp, ax=None, show=False):

    input = np.array([[res['params']['log_nD'], res['params']['log_nNe']] for res in opt.res])
    output = -np.array([res["target"] for res in opt.res])
    
    input = input[:N_STEPS]
    output = output[:N_STEPS]

    input = np.power(10, input)
    inp = np.power(10, inp)


    mu, sigma = posterior(opt, np.log10(input), output, np.log10(inp))

    np.append(inp, input, axis=0)
        
    if not ax:
        ax = plt.axes()

    levels = np.linspace(1, 3, 50)

    cntr = ax.tricontourf(inp[:,0], inp[:,1], np.log10(mu), cmap=cc.cm.diverging_bwr_40_95_c42, levels=levels, extend='max')

    ax.scatter(input[:,0], input[:,1], c='dimgray', s=10, clip_on=False)

    optInd = output.argmin()
    ax.scatter(input[optInd,0], input[optInd,1], c='r', marker='*', s=70)


    if show:
        plt.show()
        
    return cntr, ax

def plot_powell(log, ax=None, show=False):

    P = np.array(log['P'])
    nD = P[:,0]
    nNe = P[:,1]
    
    if not ax:
        ax = plt.axes()
    
    ax.plot(nD, nNe, ':c', linewidth=2)
    ax.scatter(nD[0], nNe[0], c='c', marker='p', s=50)
    ax.scatter(nD[-1], nNe[-1], c='c', marker='*', s=70)

    if show:
        plt.show()

    return ax

def main():

    powFiles = sorted(glob.glob(POWELL_PATH+'/log_dBB*'), key=lambda x: float(x[len(POWELL_PATH)+8:-5]))
    bayFiles = sorted(glob.glob(BAYESIAN_PATH+'/log_dBB*'), key=lambda x: float(x[len(BAYESIAN_PATH)+8:-5]))
    
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True, constrained_layout=True)
    indicators = ['a)', 'b)', 'c)', 'd)']
    
    for i, (powfile, bayfile) in enumerate(zip(powFiles, bayFiles)):
        # Load bayesian log
        opt = BayesianOptimization(
            f=blackBoxFunction,
            pbounds=bounds,
            random_state=1
        )

        load_logs(opt, logs=bayfile)
        input = np.array([[res['params']['log_nD'], res['params']['log_nNe']] for res in opt.res])

        # opt._gp.fit(input_data, output_data)
        xmin, xmax = input[:,0].min(), input[:,0].max()
        ymin, ymax = input[:,1].min(), input[:,1].max()
        x = np.logspace(xmin, xmax, NX)
        y = np.logspace(ymin, ymax, NY)
        xy = np.log10(np.array([[xx, yy] for xx in x for yy in y]))

        # Load powell log
        with open(powfile, 'r') as fp:
            log = json.load(fp)

        # Plot logs
        ax = axes[int(i>1), i%2]
        
        cntr, _ = plot_gp(opt, xy, ax=ax)
        plot_powell(log, ax=ax)
        ax.text(1.5e17, 1.7e20, indicators[i], fontsize=22)
        
        ax.set_xlim(10**xmin, 10**xmax)#(xmin, xmax)
        ax.set_ylim(10**ymin, 10**ymax)#(ymin, ymax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1e18, 1e20, 1e22])

    ticks = np.linspace(1, 3, 3, dtype=int)
    labels = [r'$10^{1}$', r'$10^{2}$', r'$\geq 10^{3}$']'
    cbar = fig.colorbar(cntr, ax=axes, shrink=0.95, label='objective function', ticks=ticks)
    
    cbar.ax.set_yticklabels(labels)
    
    fig.supxlabel(r'injected deuterium ($\rm{m}^{-3}$)')
    fig.supylabel(r'injected neon ($\rm{m}^{-3}$)')

    if FIG_NAME:
        plt.savefig(FIG_NAME)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
