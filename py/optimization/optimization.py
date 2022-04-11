import sys, os
import numpy as np
from dataclasses import dataclass

sys.path.append(os.path.abspath('..'))
from simulation import Simulation

class Optimization:

    @dataclass
    class Settings:
        """
        Settings parameters for the optimization algorithm.
        """
        pass


    def __init__(self, simulation=None, parameters={}, verbose=True, **settings):
        """
        Constructor.
        """

        self.simulation = simulation
        self.verbose = verbose

        # Check if optimization parameters are valid
        for key in parameters.keys():
            if key not in simulation().input.__dataclass_fields__.keys():
                raise AttributeError(key+' is not an input parameter in the specified simulation.')

        self.parameters = parameters

        # Configure settings provided by the user.
        try:
            self.settings = self.Settings(**settings)
        except TypeError as err:
            print(f'Provided settings must exist in {Input().__dataclass_fields__.keys()}')
            raise err


    def getParameters(self):
        return np.array(list(self.parameters.values()))

    def setParameters(self, arr):
        assert len(arr) == len(self.parameters), 'Array length must be equalt to the number of optimization parameters!'

        for i, key in enumerate(self.parameters.keys()):
            self.parameters[key] = arr[i]

    def run(self):
        pass


    def writeLog(self, out='log.json'):
        """
        Log progress (...)
        """
        pass

    def isFinished(self):
        # return self.output i not None
        pass
