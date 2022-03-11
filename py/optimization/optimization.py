import types
import sys, os
import numpy as np
from dataclasses import dataclass
from multiprocessing import Process, Queue
from time import time
# print(help('modules'))
# from ..simulation import DREAMSimulation
# from . import simulation

sys.path.append(os.path.abspath('..'))
from simulation.simulation import Simulation

class Optimization:

    @dataclass
    class Settings:
        """
        Settings parameters for the optimization algorithm.
        """
        pass


    def __init__(self, simulation=None, verbose=True, **settings):
        """
        Constructor.
        """
        if objFun is None:
            raise AttributeError('Expected an objective function to be provided.')

        self.simulation = simulation
        self.verbose = verbose

        # Configure settings provided by the user.
        try:
            self.settings = self.Settings(**settings)
        except TypeError as err:
            print(f'Provided settings must exist in {Input().__dataclass_fields__.keys()}')
            raise err

    def _runSimulation(self, **inputs):


        queue = Queue()
        p = Process(target=lambda : queue.put(self.simulation(**))
        p.start()

        time0 = time()
        while time() - time0 < self.timeout:
            if not p.is_alive():
                return queue.get()
        else:
            print(f'TIMEOUT ERROR: Simulation time has reached the maximum time of {self.timeout} s.' )


        timer = Process(target=lambda: time.sleep(self.timeout))
        timer.start()
        process.start()

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
