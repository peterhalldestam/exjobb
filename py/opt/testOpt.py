#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
import json
from sim.DREAM.expDecay import ExponentialDecaySimulation
from powell.PowellOptimization import PowellOptimization
from powell.PowellOptimization import LINEMIN_BRENT, LINEMIN_GSS
from powell.PowellOptimization import POWELL_TYPE_RESET, POWELL_TYPE_DLD
from objective import baseObjective

if len(sys.argv) < 2:
    raise Exception('Insufficient number of arguments.\nOptimization log must be specified.')
elif len(sys.argv) > 3:
    raise Exception('An unexpected number of arguments found.\nOnly 1-2 arguments should be provided.')

try:
    with open(sys.argv[1], 'r') as fp:
        log = json.load(fp)
except:
    raise Exception('Invalid file name/format.')

if len(sys.argv) == 3:
    try:
        perc = float(sys.argv[2])/100
    except:
        raise Exception ('Invalid percentage specified.')
else:
    perc = .05

nD2, nNe = log['P'][-1]
parameters = {'nD2': nD2, 'nNe': nNe}
lowerBound = (nD2*(1.-perc), nNe*(1.-perc))
upperBound = (nD2*(1.+perc), nNe*(1.+perc))

po = PowellOptimization(simulation=ExponentialDecaySimulation, parameters=parameters, verbose=True,
                        obFun=baseObjective, maxIter=20, ftol=1.,
                        upperBound=upperBound, lowerBound=lowerBound,
                        linemin=LINEMIN_BRENT, powellType=POWELL_TYPE_DLD,
                        maximize=True, out=f'sensLog_{int(perc*100)}perc')

output = po.run()
print(output)

