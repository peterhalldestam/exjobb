#!/usr/bin/env python3
import sys, os
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from sklearn.gaussian_process.kernels import Matern

from sim.DREAM.transport import TransportSimulation, TransportException
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import heatLossObjective

from tmp import TMP
LOG_NAME = 'data/transport'

LARGE_NUMBER = 500

DBBS = [.002, .003, .004, .005]
# OPTIMA = np.log10([(5.52e20, 4.3e19), (2.28e21, 2.04e19), (1.92e21, 2.78e19), (3.30e21, 1.32e19)])
OPTIMA = [(20.741554, 19.633368), (21.357925, 19.309378), (21.2833012, 19.444044), (21.51791, 19.1196)]
OPTIMA = [OPTIMA[2]]
DNNS = [.1]#[.005, .05, .2]


def objective_function(log_nD, log_nNe):

    nD = np.power(10, log_nD)
    nNe = np.power(10, log_nNe)

    # Create simulation object
    sim = TransportSimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe)

    try:
        # Run simulation
        sim.run(handleCrash=True)

    except (MaximumIterationsException, TransportException):
        # Penalize bad simulation runs
        return LARGE_NUMBER

    return heatLossObjective(sim.output)


def main():

    print(TMP)

    # for each magnetic perturbation
    for dBB, (log_nD_opt, log_nNe_opt) in zip(DBBS, OPTIMA):

        print(10**log_nD_opt, 10**log_nNe_opt)

        # for each variation in input space
        for dnn in DNNS:

            # set pessimization bounds
            perturb = np.log10([(1 - dnn), (1 + dnn)])
            ai, bi = log_nD_opt + perturb
            aj, bj = log_nNe_opt + perturb
            print(ai, bi, aj, bj)

            # initialize optimizer
            optimizer = BayesianOptimization(
                f=objective_function,
                pbounds={'log_nD': (ai, bi), 'log_nNe': (aj, bj)},
                verbose=2,
                random_state=420
            )
            optimizer.set_gp_params(kernel=Matern(length_scale=[1, 1.], nu=2.5))
            logger = JSONLogger(path=f'{LOG_NAME}_{dBB}_{dnn}')
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            # evaluate the minimum
            optimizer.probe(params={'log_nD': log_nD_opt, 'log_nNe': log_nNe_opt}, lazy=False)

            # run pessimization
            optimizer.maximize(init_points=80, n_iter=20, acq='ei')
            print(optimizer.max)


if __name__ == "__main__":
    sys.exit(main())
