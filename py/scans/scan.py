#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
import utils
from DREAM import DREAMOutput
from DREAMSimulation import DREAMSimulation
from DREAMSimulation import MaximumIterationsException

OUTPUT_DIR = 'outputs/'
LOG_PATH = 'test.log'


N_NEON      = 1
N_DEUTERIUM = 1

# log10
MIN_DEUTERIUM, MAX_DEUTERIUM    = 18, 23
MIN_NEON, MAX_NEON              = 17, 21

DEUTERIUM_DENSITIES = np.logspace(MIN_DEUTERIUM, MAX_DEUTERIUM, N_DEUTERIUM)
NEON_DENSITIES      = np.logspace(MIN_NEON, MAX_NEON, N_NEON)

def constrain(x, y):
    return 45 < np.log10(x) + 3/2 * np.log10(y) < 53

def removeOutputFiles():
    """
    Removes all files in OUTPUT_DIR.
    """
    paths = [OUTPUT_DIR + path for path in os.listdir(OUTPUT_DIR)]
    for fp in paths:
        os.remove(fp)

def main():

    # Set up data log
    if os.path.exists(LOG_PATH):
        sys.exit(f'ERROR: {LOG_PATH} already exists!')
    logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.INFO,format='%(asctime)s :\t %(message)s')
    logging.info(f'format: current/total, nNe, nD, tCQ, I_re')

    try:
        removeOutputFiles()
    except FileNotFoundError:
        pass

    # Run simulations
    scanSpace = [(n1, n2) for n1 in DEUTERIUM_DENSITIES for n2 in NEON_DENSITIES if constrain(n1, n2)]

    for i, (nD, nNe) in enumerate(scanSpace):

        print(f'Running simulation {i+1}/{len(scanSpace)}')

        s = DREAMSimulation(verbose=True)
        s.configureInput(nNe=nNe, nD2=nD)

        try:
            s.run(handleCrash=True)
        except MaximumIterationsException:
            print('Skipping this simulation.')
        else:
            tCQ  = s.output.getCQTime()
            I_re = s.output.getMaxRECurrent()
            logging.info(f'{i+1}/{len(scanSpace)},{nNe},{nD},{tCQ},{I_re}')
        finally:
            removeOutputFiles()

    return 0

if __name__ == '__main__':
    sys.exit(main())
