import sys
import json
import numpy as np
from numpyencoder import NumpyEncoder
from dataclasses import dataclass
from types import FunctionType

from optimization import Optimization
from linemin import Brent, goldenSectionSearch

LINEMIN_BRENT = 1
LINEMIN_GS = 2

class PowellOptimization(Optimization):

    @dataclass
    class Settings(Optimization.Settings):
        """
        Settings parameters for the optimization algorithm.
        """
        # Objective function
        obFun:      FunctionType

        # Domain boundries
        lowerBound: tuple
        upperBound: tuple

        # Termination conditions
        ftol:       float       = 1e-2
        maxIter:    int         = 10

        # Linemin method
        linemin:    int         = LINEMIN_BRENT

    def __init__(self, simulation=None, parameters={}, verbose=True, **settings):
        """
        Constructor.
        """
        super().__init__(simulation=simulation, parameters=parameters, verbose=verbose, **settings)
        self.log = None

    def _restrainedFun(self, P):
        """
        Modified base function that incorporates parameter boundries in the optimization process and runs simulation.
        Any configuration outside of the specified domain automatically returns an arbitrary large number (10^10).
        """
        if (P < self.settings.lowerBound).any() or (P > self.settings.upperBound).any():
            return 1e+10

        else:
            self.setParameters(P)
            s = self.simulation(verbose=False, **self.parameters)

            try:
                s.run(handleCrash=True) # handleCrash is currently specific to DREAMSimulation
            except Exception as err:
                self.writeLog()
                print(f'Simulation error obtained for parameters:\n {self.parameters}\n')
                raise err

            return self.settings.obFun(s.output)

    def _initBracket(self, P, u):
        """
        Method used to find two initial starting points when attempting to bracket a minimum.
        Additionaly, the upper and lower limits of the 1D line function are returned.
        """
        # Arbitrary small number used to avoid dividing by zero and avoid errors if the vector is already located at the minimum.
        eps = 1e-10

        # Computes the values xi for which P + xi*u lies on the upper and lower boundries of the respective parameters.
        upperCross = (self.settings.upperBound - P) / (u+eps)
        lowerCross = (self.settings.lowerBound - P) / (u+eps)
        allCross = np.append(upperCross, lowerCross)

        # Finds the positive and negative values for xi that minimize abs(xi), i.e. the points at which the first boundries are crossed.
        xin = np.max(allCross[allCross <= 0.])
        xip = np.min(allCross[allCross > 0.])

        lineBounds = (xin, xip)
        lp = xip * np.linalg.norm(u)

        return (-eps, lp/10), lineBounds

    def _findBracket(self, fun, b0, bounds, gamma=2, verbose=False):
        """
        Simple algorithm used to initially bracket a local minimum by expontentially increasing step sizes.

        :param fun:     function handle to a 1D function where the minimum is going to be bracketed.
        :param b0:      tuple or list object containing two of the points that are going to be included in the bracket.
        :param gamma:   constant factor by which the step size is increased.
        """

        ax = b0[0]; bx = b0[1]
        fa = fun(ax); fb = fun(bx)
        lx = bounds[0]; ux = bounds[1]

        swap = False

        if fb > fa:
            dum = bx
            bx = ax
            ax = dum

            dum = fb
            fb = fa
            fa = dum

            swap = True

        cx = bx + gamma*(bx-ax)
        fc = fun(cx)

        i=1
        while fc <= fb:
            i += 1

            gamma *= 2
            cx = bx + gamma*(bx-ax)

            if cx > ux:
                cx = ux
                break
            elif cx < lx:
                cx = lx
                break

            fc = fun(cx)

        if verbose:
            print(f'Minimum bracketed after {i} iterations.')

        if swap:
            return (cx, bx, ax)
        else:
            return (ax, bx, cx)

    def run(self):
        """
        Runs the main optimization routine.
        Returns the parameter configuration P and function value f at the time of termination.
        """
        if self.settings.linemin == LINEMIN_BRENT:
            linemin = Brent
        elif self.setting.linemin == LINEMIN_GS:
            linemin = goldenSectionSearch
        else:
            raise AttributeError('The specified 1D minimization method is invalid.')

        P0 = self.getParameters()
        P = np.copy(P0)
        nD = len(P0)

        f0 = np.inf
        fmin = self._restrainedFun(P0)

        self.log = {'P': np.copy(P), 'fun': np.array([fmin])}

        # Main loop that updates the basis to avoid linear dependence.
        i = 0
        while np.abs(f0-fmin) > self.settings.ftol:
            i += 1
            f0 = fmin

            if i > self.settings.maxIter:
                if self.verbose:
                    print(f'Maximum number of iterations exceeded (maxIter = {self.settings.maxIter}). Terminating Powell.')
                self.writeLog()
                return P, fmin

            basis = np.eye(nD) # Resets basis (other methods not yet implemented).

            # Iterates through nD cycles of the basis vectors until all have been updated.
            for _ in range(nD):

                # Finds the minimum of the function along each direction in the basis.
                for u in basis:
                    lineFun = lambda x: self._restrainedFun(P + x*u)

                    b0, lineBounds = self._initBracket(P, u)
                    bracket = self._findBracket(lineFun, b0, lineBounds, verbose=self.verbose)

                    xmin, fmin = linemin(lineFun, bracket, verbose=self.verbose, maxIter=20)
                    P += xmin*u

                    self.log['P'] = np.vstack((self.log['P'], P))
                    self.log['fun'] = np.append(self.log['fun'], fmin)

                # Creates a new basis vector from the total distance moved in the previous cycle.
                uN = P - P0
                basis[:-1] = basis[1:]
                basis[-1] = uN

                # Minimizes along new direction and sets new P0.
                lineFun = lambda x: self._restrainedFun(P + x*uN)

                b0, lineBounds = self._initBracket(P, uN)
                bracket = self._findBracket(lineFun, b0, lineBounds, verbose=self.verbose)

                xmin, fmin = linemin(lineFun, bracket, verbose=self.verbose, maxIter=20)
                P += xmin*uN
                P0 = np.copy(P)

                self.log['P'] = np.vstack((self.log['P'], P))
                self.log['fun'] = np.append(self.log['fun'], fmin)

        if self.verbose:
            print(f'Powell finished after {i} iterations.')

        self.writeLog()
        return P, fmin


    def writeLog(self, out='log.json'):
        """
        Saves the current log in as json file.
        """
        with open(out, 'w') as fp:
            json.dump(self.log, fp, cls=NumpyEncoder)

    def isFinished(self):
        # return self.output i not None
        pass
