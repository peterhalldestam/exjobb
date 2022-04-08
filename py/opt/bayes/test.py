import sys, os
import numpy as np

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import baseObjective

OUTPUT_DIR = '/outputs/'
LARGE_NUMBER = 1e10

CRITICAL_CURRENT = 150e3
CQ_TIME_MIN      = 50e-3
CQ_TIME_MAX      = 150e-3
SLOPE            = 3e2


def objectiveFunction(nD, nNe):

    # Create simulation object
    sim = ExponentialDecaySimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe)

    try:
        # Run simulation
        sim.run(handleCrash=True)

    except MaximumIterationsException:
        # Penalize bad simulation runs (to avoid these regions)
        return LARGE_NUMBER

    return baseObjective(sim.output)


def main():

    # Bounded region of parameter space
    bounds = {'nD': (1e21, 1e22), 'nNe': (1e18, 1e19)}

    optimizer = BayesianOptimization(f=lambda nD, nNe: -objective(nD, nNe), pbounds=bounds, verbose=2, random_state=1)


    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=1, n_iter=5)


    print(optimizer.max)
if __name__ == "__main__":
    sys.exit(main())
