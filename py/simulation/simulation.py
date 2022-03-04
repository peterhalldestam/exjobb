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
        # if self.val < self.min or self.max < self.val:
        #     raise AttributeError(f'The value {self.val} is not within the domain interval.')

    def inDomain(self):
        return (self.val < self.max and self.min < self.val)

class Simulation:

    @dataclass
    class Input:
        """
        Input parameters for simulation object.
        """
        def __post_init__(self):
            for field in self.__dataclass_fields__.keys():
                if not isinstance(getattr(self, field), Parameter):
                    raise TypeError('Input object expected only Parameter attributes')

        def inDomain(self) -> bool:
            """
            Checks if each current input parameter is within its domain interval.
            """
            for field in self.__dataclass_fields__.keys():
                parameter = getattr(self, field)
                if parameter.inDomain():
                    return False
            return True

        def getArray():
            pass
        def fromVector():
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
        self.objFun = None

        # Set input from any user provided input parameters.
        try:
            self.input = self.Input(**inputs)
        except TypeError as err:
            print(f'Provided inputs must exist in {self.Input().__dataclass_fields__.keys()}')
            raise err

        self.output = None
        self.objFun = None




    def run(self, doubleIterations=None):
        """
        Run simulation and update output.
        """
        self.output = None
        pass

    def isFinished(self) -> bool:
        return self.output is not None
