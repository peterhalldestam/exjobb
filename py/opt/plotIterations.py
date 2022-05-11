#!/usr/bin/env python3

import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes.opt1 import blackBoxFunction

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'

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


POWELL_DIRECTORY = 'powell/data/transportData'
BAYESIAN_DIRECTORY = 'bayes/transportData'
FIG_NAME = None #'iterations_v2.eps'

N_MAX = 110

powFiles = sorted(glob.glob(POWELL_DIRECTORY+'/log_dBB*'), key=lambda x: float(x[len(POWELL_DIRECTORY)+8:-5]))
bayFiles = sorted(glob.glob(BAYESIAN_DIRECTORY+'/log_dBB*'), key=lambda x: float(x[len(BAYESIAN_DIRECTORY)+8:-5]))

bounds = {'log_nD': (1e19, 2e22), 'log_nNe': (1e15, 1e19)}

fun1, dBB1 = [], []
fun2, dBB2 = [], []

fig, ax = plt.subplots(figsize=(9,6))


for bayFile, powFile in zip(bayFiles, powFiles):
    opt = BayesianOptimization(f=blackBoxFunction, pbounds=bounds, random_state=1)
    #with open(file, 'r') as fp:
    load_logs(opt, logs=bayFile)

    targets = -np.array([res["target"] for res in opt.res])[:N_MAX]
    opts = np.array([targets[:i+1].min() for i in range(len(targets))])
    #bayLine = ax.plot(np.arange(1, len(opts)+1), opts-opts[-1], '--')
    bayLine = ax.plot(np.arange(1, len(opts)+1), opts, '--')

    with open(powFile, 'r') as fp:
        log = json.load(fp)
    
    dBB = float(bayFile[len(BAYESIAN_DIRECTORY)+8:-5])
    
    allFun = np.array(log['allFun'])
    allSim = allFun[allFun < 1e10]
    allOpt = np.array([allSim[:i+1].min() for i in range(len(allSim))])
    fun = np.repeat(log['fun'], 2)
    fun = np.delete(fun, -1)
    nSim = np.repeat(log['nSim'], 2)
    nSim = np.delete(nSim, 0)
    
    powLine = ax.plot(np.arange(1, len(allOpt)+1), allOpt, color=bayLine[0].get_color(), label=rf'{dBB*100:.2}\%')
    ax.scatter(len(allSim), fun[-1], color=bayLine[0].get_color(), marker='x')


custom_lines = [Line2D([0], [0], color='k', linestyle='-', lw=1.5),
                Line2D([0], [0], color='k', linestyle='--', lw=1.5)]
         
custom_legend = plt.legend(custom_lines, ['Powell\'s method', 'Bayesian method'], loc='upper center')
ax.legend(title=r'$\delta B/B$', loc=1)
ax.add_artist(custom_legend)

ax.set_xlabel('number of simulations')
ax.set_ylabel('current minimum')
ax.set_yscale('log')
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])

if FIG_NAME:
    plt.savefig(FIG_NAME)
plt.show()
