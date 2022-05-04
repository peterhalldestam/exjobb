#!/usr/bin/env python3

import sys, os
sys.path.append('/home/hannber/exjobb/py/simulation')
sys.path.append('/home/hannber/exjobb/py')
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import json
from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.transport import TransportSimulation
from objective import baseObjective, heatLossObjective


if len(sys.argv) == 2:

    with open(sys.argv[1], 'r') as fp:
        log = json.load(fp)

    nD2, nNe = log['P'][-1]
    
    objective = baseObjective
    s = ExponentialDecaySimulation(nD2=nD2, nNe=nNe, id='optim')

if len(sys.argv) == 3:
  
    objective = heatLossObjective
    
    with open(sys.argv[1], 'r') as fp:
        log = json.load(fp)

    nD2, nNe = log['P'][-1]
    dBB = float(sys.argv[2])

    s = TransportSimulation(nD2=nD2, nNe=nNe, TQ_initial_dBB0=dBB, id='optim')

    
s.run(handleCrash=True)

I_re_max = s.output.maxRECurrent *1e-6
t_CQ = s.output.currentQuenchTime*1e3

print('\nMaximal runaway current'.ljust(26) + ' = ' + f'{I_re_max:.3} MA'.rjust(10))
print('Current quench time'.ljust(25) + ' = ' + f'{t_CQ:.4} ms'.rjust(11))

if len(sys.argv) == 3:
    transFrac = s.output.transportedFraction*1e2
    print('Transported fraction'.ljust(25) + ' = ' + f'{transFrac:.3} %'.rjust(10))

print('Objective function'.ljust(25) + ' = ' + f'{objective(s.output)}'.rjust(10))

ax = s.output.visualizeCurrents(show=False)

ax.set_title(f'nD = {nD2:.2} m^-3    nNe = {nNe:.2} m^-3')
ax.text(30, 14, f'Maximal runaway current: {I_re_max:.3} MA', fontsize = 12)
ax.text(30, 13, f'Current quench time: {t_CQ:.4} ms', fontsize = 12)

if len(sys.argv) == 3:
    ax.text(30, 12, f'Transported fraction: {transFrac:.3} %', fontsize = 12)
    
plt.show()
