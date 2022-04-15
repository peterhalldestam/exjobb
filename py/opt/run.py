#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
from sim.DREAM.transport import TransportSimulation
from sim.DREAM.expDecay import ExponentialDecaySimulation
from powell.PowellOptimization import PowellOptimization
from powell.PowellOptimization import LINEMIN_BRENT, LINEMIN_GSS
from powell.PowellOptimization import POWELL_TYPE_RESET, POWELL_TYPE_DLD
from objective import baseObjective, heatLossObjective

# Starting points and boundries
DEUTERIUM_START = 6e20
DEUTERIUM_MIN   = 2e20
DEUTERIUM_MAX   = 2e22

NEON_START  = 6e17
NEON_MIN    = 0.
NEON_MAX    = 12e18

# Transport settings
TRANSPORT   = True
dBB0        = 1.5e-3

# Ouptut file
OUTPUT = 'log_dBB15e-4'
OUTPUTDIR = 'logs'

def main():

    simArgs = {}

    if TRANSPORT:
        simulation = TransportSimulation
        obFun = heatLossObjective
        simArgs['TQ_initial_dBB0'] = dBB0
    else:
        simulation = ExponentialDecaySimulation
        obFun = baseObjective

    parameters = {'nD2': {'val': DEUTERIUM_START, 'min': DEUTERIUM_MIN, 'max': DEUTERIUM_MAX},
                  'nNe': {'val': NEON_START, 'min': NEON_MIN, 'max': NEON_MAX}}
#    lowerBound = (DEUTERIUM_MIN, NEON_MIN)
#    upperBound = (DEUTERIUM_MAX, NEON_MAX)

    po = PowellOptimization(simulation=simulation, parameters=parameters, simArgs=simArgs,
                            verbose=True, out=OUTPUTDIR+'/'+OUTPUT,
                            obFun=obFun, maxIter = 20, ftol = 1.,
#                            upperBound=upperBound, lowerBound=lowerBound,
                            linemin=LINEMIN_BRENT, powellType = POWELL_TYPE_DLD)

    output = po.run()
    print(output)

if __name__ == '__main__':
    sys.exit(main())
