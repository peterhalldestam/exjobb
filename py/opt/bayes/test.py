import sys, os
import numpy as np

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



from sim.DREAM.expDecay import ExponentialDecaySimulation
from sim.DREAM.DREAMSimulation import MaximumIterationsException
OUTPUT_DIR = '/outputs/'
LARGE_NUMBER = 1e10

CRITICAL_CURRENT = 150e3
CQ_TIME_MIN      = 50e-3
CQ_TIME_MAX      = 150e-3
SLOPE            = 3e2

def logisticFunction(x, x0, k):
    return 1/2 + 1/2*np.tanh(k*(x-x0))



def objective(nD, nNe):

    sim = ExponentialDecaySimulation(verbose=False)
    sim.configureInput(nD2=nD, nNe=nNe)

    try:
        sim.run(handleCrash=True)
    except MaximumIterationsException:
        return LARGE_NUMBER

    paths = [OUTPUT_DIR + path for path in os.listdir(OUTPUT_DIR)]
    print('test')
    for fp in paths:
        print(fp)
        os.remove(fp)

    maxRunawayCurrent = sim.output.maxRECurrent
    currentQuenchTime = sim.output.currentQuenchTime

    obj1 = maxRunawayCurrent / CRITICAL_CURRENT
    obj2 = 100*logisticFunction(-currentQuenchTime, -CQ_TIME_MIN, SLOPE)
    obj3 = 100*logisticFunction(currentQuenchTime, CQ_TIME_MAX, SLOPE)
    print(maxRunawayCurrent, currentQuenchTime)
    return obj1 + obj2 + obj3


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
