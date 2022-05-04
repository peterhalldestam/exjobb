#!/usr/bin/env python3
import sys, os, glob
import numpy as np

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



from sim.DREAM.transport import TransportSimulation, TransportException
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import baseObjective, heatLossObjective

LOG_PATH = 'dataNew/log_dBB60e-4.json'
LARGE_NUMBER = 500#1_000_000


dBB0 = 6e-3

def blackBoxFunction(log_nD, log_nNe):

    nD = np.power(10, log_nD)
    nNe = np.power(10, log_nNe)

    # Create simulation object
    sim = TransportSimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe, TQ_initial_dBB0=dBB0)

    try:
        # Run simulation
        sim.run(handleCrash=True)

    except (MaximumIterationsException, TransportException):
        # Penalize bad simulation runs (to avoid these regions)
        for file in glob.glob('outputs/*'):
            os.remove(file)
            
        return -LARGE_NUMBER


    return -heatLossObjective(sim.output)


def main():

    # Bounded region of parameter space
    bounds = {'log_nD': (17, 22.2), 'log_nNe': (15, 21)}    # previously (19, 22.2), (15, 19)

    optimizer = BayesianOptimization(f=blackBoxFunction, pbounds=bounds, verbose=2, random_state=4)

    logger = JSONLogger(path=LOG_PATH)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    print('lesgo')
    optimizer.maximize(init_points=10, n_iter=100, acq='ei') # previously n_iter=300


    print(optimizer.max)
if __name__ == "__main__":
    sys.exit(main())
