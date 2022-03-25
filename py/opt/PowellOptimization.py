import numpy as np
from dataclasses import dataclass

from optimization import Optimization
from simulation.simulation import Simulation
from simulation.DREAMSimulation import DREAMSimulation


class PowellOptimization(Optimization):

    @dataclass
    class Settings(Optimization.Settings):
        """
        Settings parameters for the optimization algorithm.
        """
        pass


    def __init__(self, simulation=None, verbose=True, **settings):
        """
        Constructor.
        """
        super().__init__(self, simulation=simulation, verbose=verbose, **settings)


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
