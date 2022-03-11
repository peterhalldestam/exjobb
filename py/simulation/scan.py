#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
import utils
from DREAM import DREAMOutput
from DREAMSimulation import DREAMSimulation

OUTPUT_DIR = 'outputs/'
LOG_PATH = 'scan1.log'

N_NEON      = 2
N_DEUTERIUM = 2

NEON_DENSITIES      = [n * 1e20 for n in np.logspace(-3, 1, N_NEON)]
DEUTERIUM_DENSITIES = [n * 1e20 for n in np.logspace(0, 2, N_DEUTERIUM)]

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
    logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.INFO, format='%(asctime)s :\t %(message)s')

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
            s.run(handleCrash=True)

            tCQ  = s.output.getCQTime()
            I_re = s.output.getMaxRECurrent()

            logging.info(f'{i},\t{nNe},\t{nD} => {tCQ},\t{I_re}')

            removeOutputFiles()

    return 0

if __name__ == '__main__':
    sys.exit(main())
