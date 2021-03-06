#!/usr/bin/env python3
import sys, os
import numpy as np

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import baseObjective

LOG_PATH = 'log_test_300.json'
LARGE_NUMBER = 1_000_000


def blackBoxFunction(log_nD, log_nNe):

    nD = np.power(10, log_nD)
    nNe = np.power(10, log_nNe)

    # Create simulation object
    sim = ExponentialDecaySimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe)

    try:
        # Run simulation
        sim.run(handleCrash=True)

    except MaximumIterationsException:
        # Penalize bad simulation runs (to avoid these regions)
        return -LARGE_NUMBER

    return -baseObjective(sim.output)


def main():

    # Bounded region of parameter space
    bounds = {'log_nD': (19, 22.2), 'log_nNe': (15, 19)}

    optimizer = BayesianOptimization(f=blackBoxFunction, pbounds=bounds, verbose=2, random_state=1)

    logger = JSONLogger(path=LOG_PATH)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    print('lesgo')
    optimizer.maximize(init_points=10, n_iter=300, acq='ei')


    print(optimizer.max)
if __name__ == "__main__":
    sys.exit(main())
