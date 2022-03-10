#!/usr/bin/env python3

import sys, os
import numpy as np
import json
#sys.path.append(os.path.abspath('~/exjobb/optimization'))
sys.path.append('~/DREAM')

from DREAMSimulation import DREAMSimulation
from DREAM import DREAMException
from optimization import Powell
import subprocess

def logisticFunction(x, x0, k):
    return 1/2 + 1/2*np.tanh(k*(x-x0))

def densityFun(nD2, nNe, out='out'):
    #print(f'nD2={nD2}, nNe={nNe}')
   
    s = DREAMSimulation(id = f'{nD2:.4}_{nNe:.4}', nD2=nD2, nNe=nNe, verbose=False)
    
    try:
        s.run(handleCrash=True)
    except:
        DREAMException('')
        print(f'Simulation error obtained for values nD2={nD2}, nNe={nNe}')
        sys.exit()

    I_crit = 150e3      # [A]
    t_CQ_min = 50e-3    # [s]
    t_CQ_max = 150e-3   # [s]
    k = 3e2             # [s^-1]

    t_CQ = s.output.getCQTime()
    I_re_max = s.output.getMaxRECurrent()
    
    cost_I = (I_re_max - I_crit) / I_crit * np.heaviside(I_re_max - I_crit, 1)
    cost_t = logisticFunction(-t_CQ, -t_CQ_min, k) + logisticFunction(t_CQ, t_CQ_max, k)
    
    for file in os.listdir('outputs'):
        os.remove('outputs/'+file)
    
    return cost_I + 20*cost_t


#s = DREAMSimulation(nD2=6e+20, nNe=8e18, id='test')
#s.run(handleCrash=False)
#s.output.visualizeCurrents(show=True)
#[20e20, 1e17] gav väldigt bra resultat!#

#P0 = np.array([6e20, 8e18])
P0_list = [np.array([6e20, 8e18]), np.array([20e20, 1e17])]
lowerBound = (2e20, 0.)#(4e20,1e18)
upperBound = (40e20, 12e18)

for i, P0 in enumerate(P0_list):
    powOpt = Powell(densityFun, P0, lb=lowerBound, ub=upperBound, verbose=True, maxIter=2)
    r =powOpt.run()
    log = powOpt.getLog()
    print(r)
    print(log)
    with open(f'log{i}.json', 'w') as fp:
        tmp = {}
        tmp['P'] = log['P'].tolist()
        tmp['fun'] = log['fun'].tolist()
        json.dump(tmp, fp)






"""
s0 = DREAMSimulation(nD2=3.57e+20, nNe=2.93e+18, id='init')
s0.run(handleCrash=True)
s0.output.visualizeCurrents(show=True)

s1 = DREAMSimulation(nD2=4.016e+20, nNe=2.84e+18, id='final')
s1.run(handleCrash=True)
print(s1.output.tCQ)
print(s1.output.getMaxRECurrent())
s1.output.visualizeCurrents(show=True)
"""

