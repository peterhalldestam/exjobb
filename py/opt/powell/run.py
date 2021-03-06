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
DEUTERIUM_START = 1e19 #6e20
DEUTERIUM_MIN   = 1e17 #2e20
DEUTERIUM_MAX   = 1e22

NEON_START      = 1e19 #5e18 #6e17
NEON_MIN        = 1e15
NEON_MAX        = 1e21 #12e18

# Profile settings
SHAPING     = False

CD2_START   = 0.
CD2_MIN     = -5.
CD2_MAX     = 5.

CNE_START   = 0.
CNE_MIN     = -5.
CNE_MAX     = 5.

# Transport settings
TRANSPORT   = True
dBB0        = 40e-4

# Ouptut file
OUTPUT = 'log_dBB40e-4'
OUTPUTDIR = 'data_NewStart'

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

    if SHAPING:
        parameters['cD2'] = {'val': CD2_START, 'min': CD2_MIN, 'max': CD2_MAX}
        parameters['cNe'] = {'val': CNE_START, 'min': CNE_MIN, 'max': CNE_MAX}

    po = PowellOptimization(simulation=simulation, parameters=parameters, simArgs=simArgs,
                            verbose=True, out=OUTPUTDIR+'/'+OUTPUT,
                            obFun=obFun, maxIter = 20, ftol = .1,
                            linemin=LINEMIN_BRENT, powellType = POWELL_TYPE_DLD)

    output = po.run()
    print(output)

if __name__ == '__main__':
    sys.exit(main())
