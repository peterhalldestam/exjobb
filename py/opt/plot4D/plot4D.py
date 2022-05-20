#!/usr/bin/env python3
import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes.opt1 import blackBoxFunction

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

from sim.DREAM.transport import TransportSimulation
import tokamaks.ITER as Tokamak
from utils import getDensityProfile
from DREAM import DREAMOutput

BASE_OUTPUT_DIRECTORY = 'baseOutputs' ###STÄNG AV REMOVE FILES!!!
POWELL_OUTPUT_DIRECTORY = 'powOutputs' ###STÄNG AV REMOVE FILES!!!
BAYESIAN_OUTPUT_DIRECTORY = 'bayOutputs' ###STÄNG AV REMOVE FILES!!! 

BASE_OPT_LOG = '../powell/data/transportData/log_dBB50e-4.json'
POWELL_OPT_LOG = '../powell/data/log_4D_dBB50e-4.json'     
BAYESIAN_OPT_LOG = '../bayes/data/new_log_4D_dBB50e-4.json'

FIG_NAME = None #'profile_v2.eps'

# Load output from files
baseFiles = sorted(glob.glob(BASE_OUTPUT_DIRECTORY+'/*'))
powFiles = sorted(glob.glob(POWELL_OUTPUT_DIRECTORY+'/*'))
bayFiles = sorted(glob.glob(BAYESIAN_OUTPUT_DIRECTORY+'/*'))

baseOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in baseFiles])
powOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in powFiles])
bayOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in bayFiles])
do = DREAMOutput(baseFiles[0])

# Extract powell density profiles
with open(POWELL_OPT_LOG, 'r') as fp:
    profLog = json.load(fp)
    
nD2Temp, nNeTemp, cD2, cNe = profLog['P'][-1]   

r, nD2Pow = getDensityProfile(do, nD2Temp, cD2)
_, nNePow = getDensityProfile(do, nNeTemp, cNe)

# Extract bayesian density profiles
bounds = {'log_nD': (1e19, 2e22), 'log_nNe': (1e15, 1e19), 'cD2': (-12., 12.), 'cNe': (-12., 12.)}
opt = BayesianOptimization(
    f=blackBoxFunction,
    pbounds=bounds,
    random_state=1
)

load_logs(opt, logs=BAYESIAN_OPT_LOG)

input = np.array([[10**res['params']['log_nD'], 10**res['params']['log_nNe'], res['params']['cD2'], res['params']['cNe']] for res in opt.res])
output = -np.array([res["target"] for res in opt.res])

optInd = output.argmin()

nD2Temp, nNeTemp, cD2, cNe = input[optInd,0], input[optInd,1], input[optInd,2], input[optInd,3]

_, nD2Bay = getDensityProfile(do, nD2Temp, cD2)
_, nNeBay = getDensityProfile(do, nNeTemp, cNe)

# Create uniform profiles
with open(BASE_OPT_LOG, 'r') as fp:
    baseLog = json.load(fp)
    
nD2Norm, nNeNorm = baseLog['P'][-1]      
nD2Base = np.repeat(nD2Norm, len(r))
nNeBase = np.repeat(nNeNorm, len(r))

# Normalise profiles using base density
nD2Pow /= nD2Norm; nNePow /= nNeNorm
nD2Bay /= nD2Norm; nNeBay /= nNeNorm
nD2Base /= nD2Norm; nNeBase /= nNeNorm

# Figure
fig, ax = plt.subplots(1, 2, figsize=(9,5))#, constrained_layout=True)

# Plot density profiles
ax[0].plot(r, nD2Pow, '-', c='orange', label=r'$\rm deuterium$')
ax[0].plot(r, nD2Bay, '--', c='orange')
ax[0].plot(r, nD2Base, 'k:')

ax[0].plot(r, nNePow,  'm-', label=r'$\rm neon$')
ax[0].plot(r, nNeBay,  'm--')
ax[0].plot(r, nNeBase, 'k:')


ax[0].set_xlim(r[0], r[-1])
ax[0].set_ylabel(r'$\tilde{n}$')
ax[0].set_xlabel(r'$r\, \rm (m)$')
ax[0].legend()

# Plot power
print(Tokamak.R0)
for output, linestyle in zip([powOutput, bayOutput, baseOutput], ['-', '--', ':']):

    t = output.t * 1e3
    P_trans = output.P_trans[:,0] * Tokamak.R0 * 1e-6
    P_rad =  output.P_rad * Tokamak.R0 * 1e-6
    ax[1].plot(t[1:], P_rad, 'g', linestyle=linestyle)
    ax[1].plot(t[1:], P_trans, 'r', linestyle=linestyle)
    
    print(output.P_rad)

    
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(t[1], t[-1])
ax[1].set_ylim(1e-1, 1e9)
ax[1].set_ylabel(r'$P\, \rm (MW)$')
ax[1].set_xlabel(r'$t\, \rm (ms)$')
ax[1].legend([r'$\rm radiated$', r'$\rm transported$'])

plt.tight_layout()

if FIG_NAME:
    plt.savefig(FIG_NAME)
plt.show()
