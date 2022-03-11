import sys
sys.path.append('../simulation')

import numpy as np
from dataclasses import dataclass
from types import FunctionType

from optimization import Optimization
from simulation import Simulation
from DREAMSimulation import DREAMSimulation

LINEMIN_BRENT = 1
LINEMIN_GS = 2


class PowellOptimization(Optimization):

    @dataclass
    class Settings(Optimization.Settings):
        """
        Settings parameters for the optimization algorithm.
        """
        # Objective function
        obFun:      FunctionType
        
        # Initial point and boundries
        P0:         np.ndarray
        lowerBound: tuple
        upperBound: tuple
        
        # Termination conditions
        ftol:       float       = 1e-2
        maxIter:    int         = 10
        
        # Linemin method
        linemin:    int         = LINEMIN_BRENT
        


    def __init__(self, simulation=None, verbose=True, **settings):
        """
        Constructor.
        """
        super().__init__(self, simulation=simulation, verbose=verbose, **settings)
        
        self.input = None
        self.log = None


    def run(self):

        assert isinstance(self.simulation, )
        self.input = self.simulation.Input()


    def log(self, out='log.dat'):
        """
        Log progress (...)
        """
        pass

    def isFinished(self):
        # return self.output i not None
        pass
