import numpy as np
from dataclasses import dataclass

@dataclass
class Parameter:
    """
    Input parameter going into the simulation.
    """
    val: float
    min: float
    max: float

    def __post_init__(self):
        if not self.min < self.max:
            raise AttributeError('The maximum must be strictly larger than the minimum.')
        if self.val < self.min or self.max < self.val:
            raise AttributeError(f'The value {self.val} is not within the domain interval.')



class Simulation:

    @dataclass
    class Input:
        """
        Input parameters for simulation object.
        """
        pass

    @dataclass
    class Output:
        """
        Output variables obtained when the simulation is finished.
        """
        pass


    def __init__(self, id=None, verbose=True, **inputs):
        """
        Constructor.
        """
        self.id = id
        self.verbose = verbose

        # Set input from any user provided input parameters.
        try:
            self.input = self.Input(**inputs)
        except TypeError as err:
            print(f'Provided inputs must exist in {Input().__dataclass_fields__.keys()}')
            raise err()

        self.output = None
        self.objFun = None




    def run(self, doubleIterations=None):
        """
        Run simulation and update output.
        """
        self.output = None
        pass

    def isFinished(self):
        return self.output is not None
