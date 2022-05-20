#!/usr/bin/env python3
import sys, os, glob
import numpy as np

from sklearn.gaussian_process.kernels import Matern

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



from sim.DREAM.transport import TransportSimulation, TransportException
from sim.DREAM.DREAMSimulation import MaximumIterationsException

from opt.objective import baseObjective, heatLossObjective

LOG_PATH = 'data/new_log_4D_dBB50e-4.json'
LARGE_NUMBER = 500


dBB0 = 5e-3

def blackBoxFunction(log_nD, log_nNe, cD2, cNe):

    nD = np.power(10, log_nD)
    nNe = np.power(10, log_nNe)

    # Create simulation object
    sim = TransportSimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe, cD2=cD2, cNe=cNe, TQ_initial_dBB0=dBB0)

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
    bounds = {'log_nD': (17, 22), 'log_nNe': (15, 21), 'cD2': (-12, 12), 'cNe': (-12, 12)}    # previously (19, 22.2), (15, 19)

    nD_ = np.linspace(17, 22, 5)
    nD_ = np.tile(nD_, 5)

    nNe_ = np.linspace(18, 20.5, 5)
    nNe_ = np.repeat(nNe_, 5)

    #probe = [{'log_nD': nD, 'log_nNe': nNe, 'cD2': 0., 'cNe': 0.} for nD, nNe in zip(nD_, nNe_)]

    optimizer = BayesianOptimization(
        f=blackBoxFunction,
        pbounds=bounds,
        verbose=2,
        random_state=420
    )

    # 1 hyperparameter per input
    optimizer.set_gp_params(kernel=Matern(length_scale=[1., 1., 1., 1.], nu=2.5))

    # set json logger
    logger = JSONLogger(path=LOG_PATH)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    for i, (nD, nNe) in enumerate(zip(nD_, nNe_)):
        print(f'Probing point {i+1}/{len(nD_)}')
        optimizer.probe(params={'log_nD': nD, 'log_nNe': nNe, 'cD2': 0., 'cNe': 0.}, lazy=False)

    print('Probing complete, commencing sampling using acquisition function')
    optimizer.maximize(init_points=0, n_iter=500, acq='ei') # previously n_iter=300


    print(optimizer.max)
if __name__ == "__main__":
    sys.exit(main())
