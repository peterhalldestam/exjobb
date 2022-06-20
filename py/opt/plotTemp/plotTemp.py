#!/usr/bin/env python3

import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from DREAM import DREAMOutput
from sim.DREAM.transport import TransportSimulation
from sim.DREAM.expDecay import ExponentialDecaySimulation

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

EXPDECAY_DIRECTORY = 'expDecayOutputs' ### REMOVE FILES OFF
TRANSPORT_DIRECTORY = 'transportOutputs' ### REMOVE FILES OFF

FIG_NAME = None #'temperature.eps'

expDecayFiles = sorted(glob.glob(EXPDECAY_DIRECTORY+'/*'))[:]
transportFiles = sorted(glob.glob(TRANSPORT_DIRECTORY+'/*'))[:]

expDecayOutput = ExponentialDecaySimulation.Output(*[DREAMOutput(fp) for fp in expDecayFiles])
transportOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in transportFiles])

fig, axes = plt.subplots(1, 2, figsize=(8,5), sharey=True, sharex=True)#, constrained_layout=True)

indicators = [r'$\rm a)$', r'$\rm b)$']
for i, output in enumerate([expDecayOutput, transportOutput]):
    output.t *= 1e3
    print(output.t[399])

    axes[i].plot(output.t, output.T_cold[:, 0])
    axes[i].plot(output.t, output.T_cold[:, -1])
    
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    
    axes[i].set_xlim(transportOutput.t[0], transportOutput.t[-1])
    axes[i].set_ylim(5e-1, 1e5)
    axes[i].text(2e-5, 3.5e4, indicators[i], fontsize=22)

axes[1].legend([r'$\rm center$', r'$\rm edge$'])
fig.supxlabel(r'$t\, \rm (ms)$', y=0.06, x=0.55)
fig.supylabel(r'$T_{\rm cold}\, \rm (eV)$', x=0.04, y=0.57)

plt.tight_layout()
if FIG_NAME:
    plt.savefig(FIG_NAME)
plt.show()
