#!/usr/bin/env python3

import sys, os
import numpy as np
import json
sys.path.append('~/DREAM')

from DREAMSimulation import DREAMSimulation
from DREAM import DREAMException
from optimization import Powell

def logisticFunction(x, x0, k):
    return 1/2 + 1/2*np.tanh(k*(x-x0))

def densityFun(nD2, nNe, out='out'):

    s = DREAMSimulation(id = out, nD2=nD2, nNe=nNe, verbose=False)

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

    #cost_I = (I_re_max - I_crit) / I_crit * np.heaviside(I_re_max - I_crit, 1)
    cost_I = I_re_max / I_crit
    cost_t = logisticFunction(-t_CQ, -t_CQ_min, k) + logisticFunction(t_CQ, t_CQ_max, k)

    for file in os.listdir('outputs'):
        os.remove('outputs/'+file)

    return cost_I + 100*cost_t



#[20e20, 1e17] gav väldigt bra resultat!#
P0_list = [np.array([6e20, 8e18])]
lowerBound = (2e20, 0.)
upperBound = (3e22, 12e18)

for i, P0 in enumerate(P0_list):
    powOpt = Powell(densityFun, P0, lb=lowerBound, ub=upperBound, verbose=True, maxIter=5)
    r = powOpt.run()
    log = powOpt.getLog()
    print(f'Optimum found at: {r}')
    with open(f'logs/log{i}.json', 'w') as fp:
        tmp = {}
        tmp['P'] = log['P'].tolist()
        tmp['fun'] = log['fun'].tolist()
        json.dump(tmp, fp)

