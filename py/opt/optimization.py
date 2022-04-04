import sys, os
import numpy as np
from dataclasses import dataclass

from sim.simulation import Simulation
from opt.objective import Objective

class Optimization:

    @dataclass
    class Settings:
        """
        Settings parameters for the optimization algorithm.
        """
        pass


    def __init__(self, sim=None, obj=None, verbose=True, **settings):
        """
        Constructor.
        """
        if sim is None:
            raise AttributeError('Expected an Simulation subclass to be provided.')
        elif not issubclass(sim, Simulation):
            raise TypeError(f'Expected sim to be a Simulation subclass, not {type(sim)}.')

        if obj is None:
            raise AttributeError('Expected an objective function to be provided.')
        elif not issubclass(obj, Objective):
            raise TypeError(f'Expected obj to be a Objective subclass, not {type(obj)}.')

        self.sim = sim
        self.obj = obj
        self.verbose = verbose

        # Configure settings provided by the user.
        try:
            self.settings = self.Settings(**settings)
        except TypeError as err:
            print(f'Provided settings must exist in {self.Settings().__dataclass_fields__.keys()}')
            raise err


    def run(self):
        pass



    def log(self, out='log.dat'):
        """
        Log progress (...)
        """
        pass

    def isFinished(self):
        # return self.output i not None
        pass
