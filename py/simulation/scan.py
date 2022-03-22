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
LOG_PATH = 'scan_test.log'


N_NEON      = 20
N_DEUTERIUM = 20

# log10
MIN_DEUTERIUM, MAX_DEUTERIUM    = 20, 22 + np.log10(1.6)
MIN_NEON, MAX_NEON              = 17, 20

DEUTERIUM_DENSITIES = np.logspace(MIN_DEUTERIUM, MAX_DEUTERIUM, N_DEUTERIUM)
NEON_DENSITIES      = np.logspace(MIN_NEON, MAX_NEON, N_NEON)

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
    logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.INFO,
                        format='%(asctime)s :\t %(message)s')

    try:
        removeOutputFiles()
    except FileNotFoundError:
        pass

    # Run simulations
    for iNe, nNe in enumerate(NEON_DENSITIES):
        for iD, nD in enumerate(DEUTERIUM_DENSITIES):

            i = iNe * N_DEUTERIUM + iD
            print(f'Running simulation {i+1}/{N_NEON * N_DEUTERIUM}')

            s = DREAMSimulation(verbose=False)
            s.configureInput(nNe=nNe, nD2=nD)

            try:
                s.run(handleCrash=True)
            except MaximumIterationsException:
                print('Skipping this simulation.')
            else:
                tCQ  = s.output.getCQTime()
                I_re = s.output.getMaxRECurrent()
                logging.info(f'{i},\t{nNe},\t{nD} => {tCQ:2.5},\t{I_re:10.3}')
            finally:
                removeOutputFiles()

    return 0

if __name__ == '__main__':
    sys.exit(main())
