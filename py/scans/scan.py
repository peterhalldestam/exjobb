#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
import utils
from DREAM import DREAMOutput
from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.DREAMSimulation import MaximumIterationsException

OUTPUT_DIR = 'outputs/'
LOG_PATH = 'data/expDecay2.log'


N_NEON      = 40
N_DEUTERIUM = 40

MIN_DEUTERIUM, MAX_DEUTERIUM    = 18, 22.2#5e21, 1.6e22
MIN_NEON, MAX_NEON              = 15, 20#1e16, 1e17

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
    logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.INFO,format='%(asctime)s :\t %(message)s')
    logging.info(f'format: current/total, nNe, nD, tCQ, I_re, I_ohm')

    try:
        removeOutputFiles()
    except FileNotFoundError:
        pass

    # Run simulations
    scanSpace = [(n1, n2) for n1 in DEUTERIUM_DENSITIES for n2 in NEON_DENSITIES]

    for i, (nD, nNe) in enumerate(scanSpace):

        print(f'Running simulation {i+1}/{len(scanSpace)}')

        s = ExponentialDecaySimulation(verbose=False)
        s.configureInput(nNe=nNe, nD2=nD)

        try:
            s.run(handleCrash=True)
        except MaximumIterationsException:
            print('Skipping this simulation.')
        else:
            tCQ  = s.output.currentQuenchTime
            I_re = s.output.maxRECurrent
            I_ohm = s.output.finalOhmicCurrent
            logging.info(f'{i+1}/{len(scanSpace)},{nNe},{nD},{tCQ},{I_re},{I_ohm}')
        finally:
            removeOutputFiles()

    return 0

if __name__ == '__main__':
    sys.exit(main())
