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

    #nD2, nNe, aD2, aNe = log['P'][-1]
    nD2, nNe = log['P'][-1]


    #s = DREAMSimulation(nD2=nD2, nNe=nNe, aD2=aD2, aNe=aNe, id='optim0')
    #s = ExponentialDecaySimulation(nD2=nD2, nNe=nNe, id='optim0')
    s = TransportSimulation(nD2=nD2, nNe=nNe, TQ_initial_dBB0=30e-4, id='optim')
    s.run(handleCrash=True)

    I_re_max = s.output.maxRECurrent *1e-6
    t_CQ = s.output.currentQuenchTime*1e3
    transFrac = s.output.transportedFraction*1e2

    print('\nMaximal runaway current'.ljust(26) + ' = ' + f'{I_re_max:.3} MA'.rjust(10))
    print('Current quench time'.ljust(25) + ' = ' + f'{t_CQ:.4} ms'.rjust(11))
    print('Transported fraction'.ljust(25) + ' = ' + f'{transFrac:.3} %'.rjust(10))

    print('Objective function'.ljust(25) + ' = ' + f'{heatLossObjective(s.output)}'.rjust(10))

    ax = s.output.visualizeCurrents()
    #ax.set_title(f'dB/B = 0.35%    nD = {nD2:.2} m^-3    nNe = {nNe:.2} m^-3    aD2 = {aD2:.2}    aNe = {aNe:.2}')
    ax.text(30, 14, f'Maximal runaway current: {I_re_max:.3} MA', fontsize = 12)
    ax.text(30, 13, f'Current quench time: {t_CQ:.4} ms', fontsize = 12)
    ax.text(30, 12, f'Transported fraction: {transFrac:.3} %', fontsize = 12)
    plt.show()
