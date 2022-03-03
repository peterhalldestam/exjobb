import numpy as np
from dataclasses import dataclass

from optimization import Optimization

class PowellOptimization(Optimization):

    @dataclass
    class Settings(Optimization.Settings):
        """
        Settings parameters for the optimization algorithm.
        """
        pass


    def __init__(self, objFun=None, verbose=True, **settings):
        """
        Constructor.
        """
        super().__init__(self, objFun=objFun, verbose=verbose, **settings)





    def log(self, out='log.dat'):
        """
        Log progress (...)
        """
        pass

    def isFinished(self):
        # return self.output i not None
        pass
