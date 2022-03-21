#!/usr/bin/env python3

import sys
sys.path.append('../simulation')

import numpy as np
from DREAMSimulation import DREAMSimulation
from PowellOptimization import PowellOptimization
from PowellOptimization import LINEMIN_BRENT, LINEMIN_GSS
from PowellOptimization import POWELL_TYPE_RESET, POWELL_TYPE_DLD

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

parameters = {'nD2': 6e20, 'nNe': 8e18}#{'nD2': 1e22, 'nNe': 6e16}
lowerBound = (2e20, 0.)
upperBound = (3e22, 12e18)

po = PowellOptimization(simulation=DREAMSimulation, parameters=parameters, verbose=True,
                        obFun=obFun, maxIter = 3,
                        upperBound=upperBound, lowerBound=lowerBound,
                        linemin = LINEMIN_BRENT, powellType = POWELL_TYPE_DLD)

output = po.run()
print(output)
