#!/usr/bin/env python3

import sys
sys.path.append('../simulation')

import numpy as np
from DREAMSimulation import DREAMSimulation
from PowellOptimization import PowellOptimization

def logisticFunction(x, x0, k):
    return .5 + .5*np.tanh(k*(x-x0))

def obFun(output):
    t_CQ = output.getCQTime()
    I_re_max = output.getMaxRECurrent()

    I_crit = 150e3      # [A]
    t_CQ_min = 50e-3    # [s]
    t_CQ_max = 150e-3   # [s]
    k = 3e2             # [s^-1]

    cost_I = I_re_max / I_crit
    cost_t = logisticFunction(-t_CQ, -t_CQ_min, k) + logisticFunction(t_CQ, t_CQ_max, k)

    return cost_I + 100*cost_t

parameters = {'nD2': 1e22, 'nNe': 6e16, 'aD2': 0.1, 'aNe': 0.1}
lowerBound = (2e20, 0., -10., -10.)
upperBound = (3e22, 12e18, 10., 10.)

po = PowellOptimization(simulation=DREAMSimulation, parameters=parameters, verbose=True,
                        obFun=obFun, maxIter = 5,
                        upperBound=upperBound, lowerBound=lowerBound)

output = po.run()
print(output)
