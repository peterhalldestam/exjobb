#!/usr/bin/env python3
import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
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

from sim.DREAM.transport import TransportSimulation
import tokamaks.ITER as Tokamak
from utils import getDensityProfile
from DREAM import DREAMOutput

BASE_OUTPUT_DIRECTORY = ###STÄNG AV REMOVE FILES!!! 'baseOutputs'
PROFILE_OUTPUT_DIRECTORY = ###STÄNG AV REMOVE FILES!!! 'profOutputs'

BASE_OPT_LOG = '../data/transportData/log_dBB50e-4.json'
PROFILE_OPT_LOG = '../data/log_4D_dBB50e-4.json'     

FIG_NAME = None #'profile_v2.eps'

# Load output from files
baseFiles = sorted(glob.glob(BASE_OUTPUT_DIRECTORY+'/*'))
profFiles = sorted(glob.glob(PROFILE_OUTPUT_DIRECTORY+'/*'))

baseOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in baseFiles])
profOutput = TransportSimulation.Output(*[DREAMOutput(fp) for fp in profFiles])
do = DREAMOutput(baseFiles[0])

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
# Plot current evolution

for output, linestyle in zip([profOutput, baseOutput], ['-', '--']):

    t = output.t * 1e3
    P_trans = output.P_trans[:,0] * Tokamak.R0 * 1e-6
    P_rad =  output.P_rad * Tokamak.R0 * 1e-6
    ax[1].plot(t[1:], P_rad, 'g', linestyle=linestyle)
    ax[1].plot(t[1:], P_trans, 'r', linestyle=linestyle)
    #ax[1].plot(t * 1e3, output.I_ohm * 1e-6, 'r', linestyle=linestyle, label='Ohmic')
    #ax[1].plot(t * 1e3, output.I_re * 1e-6,  'b', linestyle=linestyle, label='REs')
    #ax[1].plot(t * 1e3, output.I_tot * 1e-6, 'k', linestyle=linestyle, label='total')
    
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(t[1], t[-1])
ax[1].set_ylabel('power (MW)')
ax[1].set_xlabel('time (ms)')
#ax[1].legend(['transported', 'radiated'])
"""
ax[1].set_yscale('log')
ax[1].set_ylim((1e-12, 1.))
#axMain = plt.subplot(111)
#axMain.plot(xdomain, np.sin(xdomain))
#axMain.set_yscale('log')
#axMain.set_ylim((0.01, 0.5))
ax[1].spines['top'].set_visible(False)
ax[1].xaxis.set_ticks_position('bottom')

divider = make_axes_locatable(ax[1])
axLin = divider.append_axes("top", size=2.0, pad=0, sharex=ax[1])

for output, linestyle in zip([profOutput, baseOutput], ['-', '--']):

    t = output.t
    axLin.plot(t * 1e3, output.I_ohm * 1e-6, 'r', linestyle=linestyle, label='Ohmic')
    axLin.plot(t * 1e3, output.I_re * 1e-6,  'b', linestyle=linestyle, label='REs')
    axLin.plot(t * 1e3, output.I_tot * 1e-6, 'k', linestyle=linestyle, label='total')

#axLin.plot(xdomain, np.sin(xdomain))
axLin.set_yscale('linear')
axLin.set_ylim((1., 16.))

# Removes bottom axis line
#axLin.spines['bottom'].set_visible(False)
#axLin.xaxis.set_ticks_position('top')
#plt.setp(axLin.get_xticklabels(), visible=False)

"""

#ax[1].legend(title='Currents:')
#ax[1].set_xlabel('time (ms)')
#ax[1].set_ylabel('current (MA)')

if FIG_NAME:
    plt.savefig(FIG_NAME)
plt.show()
