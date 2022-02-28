import numpy as np

class Simulation:

    def __init__(self, baseline, id=None, verbose=True, **inputs):
        """
        Constructor.
        """
        self.id = id
        self.verbose = verbose

        self.input = baseline
        self.output = None
        self.objFun = None

        # Set input from any user provided input parameters.
        if inputs:
            if verbose:
                print('User provided the following input arguments:')
            for i, (key, value) in enumerate(inputs.items()):
                if key in baseline.keys():
                    self.input[key] = value
                    if verbose:
                        print(f'\t({i+1}) {key} \t= {value}')
                else:
                    raise Exception(f'Did not expect keyword argument: {key}={value}')


    def run(self, doubleIterations=None):
        """
        Run simulation and update output.
        """
        self.output = None
        pass

    def isFinished(self):
        return self.output is not None
