import sys
import types
import numpy as np
from dataclasses import dataclass
from multiprocessing import Process, Queue
from time import time

class Simulation:

    @dataclass
    class Input:
        """
        Input parameters for simulation object.
        """

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


    def __init__(self, id=None, verbose=True, timeout=1e2, **inputs):
        """
        Constructor.
        """
        self.configureInput(**inputs)

        self.id = id
        self.verbose = verbose
        self.timeout = timeout
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


    def run(self):
        pass

    def _run(self, runSimulation):
        """
        Run simulation until finished or raise TimeoutError if taking too long.
        """
        assert isinstance(runSimulation, types.FunctionType)

        queue = Queue()

        def worker():
            queue.put(runSimulation())
            sys.stdout.flush()

        p = Process(target=worker)
        p.start()

        time0 = time()
        while time() - time0 < self.timeout:
            print(time())
            # print(f'{time()- time0:3.6},\t{queue.get()}')
            if not p.is_alive():
                break
        else:
            p.terminate()
            p.join()
            raise TimeoutError(f'TIMEOUT ERROR: Simulation time has reached the maximum time of {self.timeout} s.' )
        return queue.get()

    def isFinished(self) -> bool:
        return self.output is not None
