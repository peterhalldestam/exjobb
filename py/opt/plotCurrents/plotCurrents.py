#!/usr/bin/env python3
import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

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

from sim.DREAM.expDecay import ExponentialDecaySimulation
import tokamaks.ITER as Tokamak
from utils import getDensityProfile
from DREAM import DREAMOutput

BASE_OUTPUT_DIRECTORY = ###STÄNG AV REMOVE FILES!!! 'baseOutputs'
OPTIMA_OUTPUT_DIRECTORY = ###STÄNG AV REMOVE FILES!!! 'optOutputs'

FIG_NAME = None #'currents_v1.eps'

# Load output from files
baseFiles = sorted(glob.glob(BASE_OUTPUT_DIRECTORY+'/*'))
optFiles = sorted(glob.glob(OPTIMA_OUTPUT_DIRECTORY+'/*'))

baseOutput = ExponentialDecaySimulation.Output(*[DREAMOutput(fp) for fp in baseFiles])
optOutput = ExponentialDecaySimulation.Output(*[DREAMOutput(fp) for fp in optFiles])

"""
# Extract density profiles
with open(PROFILE_OPT_LOG, 'r') as fp:
    profLog = json.load(fp)
    
nD2Temp, nNeTemp, cD2, cNe = profLog['P'][-1]   

r, nD2Prof = getDensityProfile(do, nD2Temp, cD2)
_, nNeProf = getDensityProfile(do, nNeTemp, cNe)

# Create uniform profiles
with open(BASE_OPT_LOG, 'r') as fp:
    baseLog = json.load(fp)
    
nD2Temp, nNeTemp = baseLog['P'][-1]      
nD2Base = np.repeat(nD2Temp, len(r))
nNeBase = np.repeat(nNeTemp, len(r))

# Figure
fig, ax = plt.subplots(1, 2, figsize=(9,6), constrained_layout=True)

# Plot density profiles
ax[0].plot(r, nD2Prof/nD2Temp, '-', c='orange', label='deuterium')
ax[0].plot(r, nD2Base/nD2Temp, 'k--')

ax[0].plot(r, nNeProf/nNeTemp, 'm-', label='neon')
ax[0].plot(r, nNeBase/nNeTemp, 'k--')


ax[0].set_xlim(r[0], r[-1])
#ax[0].set_ylabel(r'$n/n_{\rm flat}$')
ax[0].set_ylabel('normalized density')
ax[0].set_xlabel('minor radial coordinate (m)')
#ax[0].set_yscale('log')
#ax[0].legend()
#profOutput.visualizeCurrents(ax=ax[1])

"""

# Figure
fig, ax = plt.subplots(figsize=(9,6))

# Plot current evolution
for output, linestyle in zip([optOutput, baseOutput], ['-', '--']):

    t = output.t * 1e3
    ax.plot(t, output.I_ohm * 1e-6, 'r', linestyle=linestyle)#, label=r'$\rm Ohmic$')
    ax.plot(t, output.I_re * 1e-6,  'b', linestyle=linestyle)#, label=r'$\rm REs$')
    ax.plot(t, output.I_tot * 1e-6, 'k', linestyle=linestyle)#, label=r'$\rm total$')
    


#ax.set_yscale('log')
#ax.set_ylim((1e-9, 1.))
#ax.set_ylim((1e-8, 1.))
ax.set_xlim((0, 150))
#axMain = plt.subplot(111)
#axMain.plot(xdomain, np.sin(xdomain))
#axMain.set_yscale('log')
#axMain.set_ylim((0.01, 0.5))
"""
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
divider = make_axes_locatable(ax)
axLin = divider.append_axes("top", size=2.0, pad=0)#, sharex=ax)
for output, linestyle in zip([optOutput, baseOutput], ['-', '--']):
    t = output.t * 1e3
    axLin.plot(t, output.I_ohm * 1e-6, 'r', linestyle=linestyle)#, label='Ohmic')
    axLin.plot(t, output.I_re * 1e-6,  'b', linestyle=linestyle)#, label='REs')
    axLin.plot(t, output.I_tot * 1e-6, 'k', linestyle=linestyle)#, label='total')
#axLin.plot(xdomain, np.sin(xdomain))
axLin.set_yscale('linear')
axLin.set_ylim((1., 16.))
axLin.set_xticklabels([])
axLin.set_xlim((0, 150))
# Removes bottom axis line
#axLin.spines['bottom'].set_visible(False)
#axLin.xaxis.set_ticks_position('top')
#plt.setp(axLin.get_xticklabels(), visible=False)

#ax[1].legend(title='Currents:')
"""
ax.set_xlabel(r'$t\, \rm (ms)$')
ax.set_ylabel(r'$I\, \rm (MA)$')
#ax.legend([r'$\rm Ohmic$', r'$\rm REs$', r'$\rm total$'])

if FIG_NAME:
    plt.savefig(FIG_NAME)
plt.show()
