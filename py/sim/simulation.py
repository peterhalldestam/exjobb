import numpy as np
import dataclasses




class Simulation:

    @dataclasses.dataclass
    class Input:
        """
        Input parameters for simulation object.
        """

        def getArray():
            pass
        def fromVector():
            pass

    @dataclasses.dataclass
    class Output:
        """
        Output variables obtained when the simulation is finished.
        """
        pass
        
    def __init__(self, id=None, verbose=True, **inputs):
        """
        Constructor.
        """
        self.configureInput(**inputs)

        self.id = id
        self.verbose = verbose
        self.objFun = None

        self.output = None
        self.objFun = None

    def configureInput(self, **inputs):
        """
        Sets input from any user provided input parameters.
        """
        try:
            self.input = self.Input(**inputs)
        except TypeError as err:
            print(f'Provided inputs must exist in {self.Input().__dataclass_fields__.keys()}')
            raise err

    def run(self, doubleIterations=None):
        """
        Run simulation and update output.
        """
        pass
