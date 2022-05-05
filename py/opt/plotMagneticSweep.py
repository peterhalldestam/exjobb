#!/usr/bin/env python3

import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes.opt1 import blackBoxFunction

POWELL_DIRECTORY = 'powell/data'
BAYESIAN_DIRECTORY = 'bayes/dataNew'

powFiles = sorted(glob.glob(POWELL_DIRECTORY+'/log_dBB*'), key=lambda x: float(x[len(POWELL_DIRECTORY)+8:-5]))
bayFiles = sorted(glob.glob(BAYESIAN_DIRECTORY+'/log_dBB*'), key=lambda x: float(x[len(BAYESIAN_DIRECTORY)+8:-5]))

bounds = {'log_nD': (1e19, 2e22), 'log_nNe': (1e15, 1e19)}

P1, fun1, dBB1 = [], [], []
P2, fun2, dBB2 = [], [], []

# Read results from powell optimization
for file in powFiles:
    with open(file, 'r') as fp:
        log = json.load(fp)

    P1.append(log['P'][-1])
    fun1.append(log['fun'][-1])
    dBB1.append(float(file[len(POWELL_DIRECTORY)+8:-5]))

P1 = np.array(P1)

for file in bayFiles:
    opt = BayesianOptimization(f=blackBoxFunction, pbounds=bounds, random_state=1)
    #with open(file, 'r') as fp:
    load_logs(opt, logs=file)

    P2.append([10**opt.max['params']['log_nD'], 10**opt.max['params']['log_nNe']])
    fun2.append(-opt.max['target'])
    dBB2.append(float(file[len(BAYESIAN_DIRECTORY)+8:-5]))

P2 = np.array(P2)

fig, ax = plt.subplots()

funAll = np.concatenate([fun1, fun2], axis=0)
fmin, fmax = np.log10(funAll.min()), np.log10(funAll.max())
norm = LogNorm(vmin=0.1, vmax=100)

powPoints = ax.scatter(P1[:,0], P1[:,1], c=fun1, norm=norm, cmap='inferno', label='Powell\'s method')
bayPoints = ax.scatter(P2[:,0], P2[:,1], marker='X', c=fun2, norm=norm,  cmap='inferno', label='Bayesian method')

fig.colorbar(bayPoints, ax=ax, label='objective function')

for i, P in enumerate(P1):
    ax.annotate(f'{dBB1[i]*100:.3}%', P, xytext=P+(0.3e20, 0.2e17))

for i, P in enumerate(P2):
    ax.annotate(f'{dBB2[i]*100:.3}%', P, xytext=P+(-2.5e20, 0.2e17))

ax.set_xlabel(r'injected deuterium ($\rm{m}^{-3}$)')
ax.set_ylabel(r'injected neon ($\rm{m}^{-3}$)')
ax.legend()
plt.show()
