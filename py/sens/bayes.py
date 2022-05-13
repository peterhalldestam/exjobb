#!/usr/bin/env python3
import sys, os, glob
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import baseObjective

from tmp import TMP
# OPT_LOG_PATH = 'data/expDecay3.json'
PES_LOG_NAME = 'data/expDecay'

LARGE_NUMBER = 500


def objective_function(log_nD, log_nNe):

    nD = np.power(10, log_nD)
    nNe = np.power(10, log_nNe)

    # Create simulation object
    sim = ExponentialDecaySimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe)

    try:
        # Run simulation
        sim.run(handleCrash=True)

    except (MaximumIterationsException, TransportException):
        # Penalize bad simulation runs
        return LARGE_NUMBER

    return baseObjective(sim.output)


def main():

    print(TMP)

    log_nD_opt  = np.log10(1.17e22)
    log_nNe_opt = np.log10(7.00e16)

    dnns = (.01, .1, .5)
    for dnn in dnns:

        # set pessimization bounds
        perturb = np.log10([(1 - dnn), (1 + dnn)])
        ai, bi = log_nD_opt + perturb
        aj, bj = log_nNe_opt + perturb

        print(ai, bi, aj, bj)
        bounds = {'log_nD': (ai, bi), 'log_nNe': (aj, bj)}

        # initialize optimizer
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=bounds,
            verbose=2,
            random_state=420
        )
        logger = JSONLogger(path=f'{PES_LOG_NAME}_{dnn}')
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # run pessimization
        optimizer.maximize(init_points=0, n_iter=1, acq='ei') # previously n_iter=300


    print(optimizer.max)
if __name__ == "__main__":
    sys.exit(main())
