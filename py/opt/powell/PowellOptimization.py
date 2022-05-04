import sys, os
sys.path.append(os.path.abspath('..'))
from sim.DREAM.transport import TransportException

import json
import numpy as np
from numpyencoder import NumpyEncoder
from dataclasses import dataclass, asdict
from types import FunctionType

from optimization import Optimization
from powell.linemin import Brent, goldenSectionSearch

# Arbitrary large and small numbers used in certain steps to represent infinity and avoid dividing by zero.
###### MAY BE PROBLEMATIC ######
BIG = 1e30
SMALL = 1e-30

LINEMIN_BRENT = 1
LINEMIN_GSS = 2

POWELL_TYPE_RESET = 1
POWELL_TYPE_DLD = 2

class PowellOptimization(Optimization):

    @dataclass
    class Settings(Optimization.Settings):
        """
        Settings parameters for the optimization algorithm.
        """
        # Objective function
        obFun:      FunctionType
        maximize:   bool        = False

        # Termination conditions
        ftol:       float       = 1e-1
        xtol:       float       = 1e-1
        maxIter:    int         = 20
        
        # Output file
        out:        str         = 'log'

        # Linemin method and Powell type
        linemin:    int         = LINEMIN_BRENT
        powellType: int         = POWELL_TYPE_RESET
        
        @property
        def asDict(self):
            """ Returns input data as dictionary. """
            tempDict = {key: val for key, val in asdict(self).items() if self.__dataclass_fields__[key].repr and key!='obFun'}
            tempDict['obFun'] = self.obFun.__name__
            return tempDict

    def __init__(self, simulation=None, parameters={}, simArgs={}, verbose=True, **settings):
        """
        Constructor.
        """
        super().__init__(simulation=simulation, parameters=parameters, simArgs=simArgs, verbose=verbose, **settings)
        self.log = None
        
        # Log data
        self.nFunEval = 0
        self.nSim = 0
        self.nExcept = 0


    def _restrainedFun(self, P):
        """
        Modified base function that incorporates parameter boundries in the optimization process and runs simulation.
        Any configuration outside of the specified domain automatically returns an arbitrary large number (10^10).
        """
        
        self.nFunEval += 1
        
        if (P < self.lowerBound).any() or (P > self.upperBound).any():
            return BIG

        else:
            self.nSim += 1
        
            self.setParameters(P)
            s = self.simulation(verbose=False, **self.parameters, **self.simArgs)

            try:
                s.run(handleCrash=True) # handleCrash is currently specific to DREAMSimulation
            except TransportException:
                self.nExcept += 1
                
                if self.verbose:
                    print('Encountered transport exception.')
                return BIG
            except Exception as err:
                self.nExcept += 1
            
                self.updateLog(P, None, None)
                print(f'Simulation error obtained for parameters:\n {self.parameters}\n')
                raise err

            if self.settings.maximize:
                return -self.settings.obFun(s.output)
            else:
                return self.settings.obFun(s.output)

    def _initBracket(self, P, u):
        """
        Method used to find two initial starting points when attempting to bracket a minimum.
        Additionaly, the upper and lower limits of the 1D line function are returned.
        """

        # Computes the values xi for which P + xi*u lies on the upper and lower boundries of the respective parameters.
        upperCross = (self.upperBound - P) / (u+SMALL)
        lowerCross = (self.lowerBound - P) / (u+SMALL)
        allCross = np.append(upperCross, lowerCross)

        # Finds the positive and negative values for xi that minimize abs(xi), i.e. the points at which the first boundries are crossed.
        xin = np.max(allCross[allCross <= 0.])
        xip = np.min(allCross[allCross > 0.])

        lineBounds = (xin, xip)

        return (-SMALL, xip/10), lineBounds

    def _findBracket(self, fun, b0, bounds, gamma=1, verbose=False):
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

        L = (bx-ax)
        cx = bx + gamma*L
        fc = fun(cx)
        
        i=1
        while fc <= fb:
            i += 1
            gamma *= 2
            
            temp = bx
            bx = cx
            cx = bx + gamma*L
            ax = temp
            fb = fc

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
        elif self.settings.linemin == LINEMIN_GSS:
            linemin = goldenSectionSearch
        else:
            raise AttributeError('The specified 1D minimization method is invalid.')

        P0 = self.getParameters()
        P = np.copy(P0)
        nD = len(P0)

        fp = BIG
        fmin = self._restrainedFun(P0)

        self.log = {'P': np.copy(P), 'fun': np.array([fmin]), 'brackets': [],
                    'nFunEval': [self.nFunEval], 'nSim': [self.nSim], 'nExcept': [self.nExcept],
                    'parameters': self.inputParameters, 'settings': self.settings.asDict}                
        self.writeLog()

        # Main loop that updates the basis to avoid linear dependence.
        i = -1
        while True:

            if self.settings.powellType == POWELL_TYPE_RESET or i == -1:
                basis = np.eye(nD) # Resets basis (other methods not yet implemented).

            # Iterates through nD cycles of the basis vectors until all have been updated.
            for _ in range(nD):
                i += 1

                if 2.*np.abs(fp-fmin) <= self.settings.ftol*(np.abs(fp)+np.abs(fmin)) and i > 0:
                    if self.verbose:
                        print(f'Powell finished after {i} iterations.')
                    self.writeLog()
                    return P, fmin


                fp = fmin

                if i > self.settings.maxIter:
                    if self.verbose:
                        print(f'Maximum number of iterations exceeded (maxIter = {self.settings.maxIter}). Terminating Powell.')
                    self.writeLog()
                    return P, fmin


                f0 = fmin
                # Finds the minimum of the function along each direction in the basis.
                for u in basis:
                    lineFun = lambda x: self._restrainedFun(P + x*u)

                    b0, lineBounds = self._initBracket(P, u)
                    bracket = self._findBracket(lineFun, b0, lineBounds, verbose=self.verbose)

                    xmin, fmin = linemin(lineFun, bracket, tol=self.settings.xtol, maxIter=20, verbose=self.verbose)
                    P += xmin*u

                    self.updateLog(P, fmin, tuple([P + x*u for x in bracket]))

                # Creates a new basis vector from the total distance moved in the previous cycle.
                uN = P - P0

                if self.settings.powellType == POWELL_TYPE_DLD:
                    fE = self._restrainedFun(P0 + 2*uN)

                    diff = self.log['fun'][-nD-1:-1] - self.log['fun'][-nD:]
                    iDf = np.argmax(diff)
                    Df = diff[iDf]

                    if fE < f0 and 2.*(f0-2*fmin+fE) * ((f0-fmin) - Df)**2 < (f0-fE)**2 * Df:
                        if self.verbose:
                            print('Discarding direction of largest decrease.')
                        basis[iDf] = uN

                        # Minimizes along new direction and sets new P0.
                        lineFun = lambda x: self._restrainedFun(P + x*uN)

                        b0, lineBounds = self._initBracket(P, uN)
                        bracket = self._findBracket(lineFun, b0, lineBounds, verbose=self.verbose)

                        xmin, fmin = linemin(lineFun, bracket, tol=self.settings.xtol, maxIter=20, verbose=self.verbose)
                        P += xmin*uN

                        self.updateLog(P, fmin, tuple([P + x*uN for x in bracket]))

                else:
                    basis[:-1] = basis[1:]
                    basis[-1] = uN

                    # Minimizes along new direction and sets new P0.
                    lineFun = lambda x: self._restrainedFun(P + x*uN)

                    b0, lineBounds = self._initBracket(P, uN)
                    bracket = self._findBracket(lineFun, b0, lineBounds, verbose=self.verbose)

                    xmin, fmin = linemin(lineFun, bracket, tol=self.settings.xtol, maxIter=20, verbose=self.verbose)
                    P += xmin*uN
                    
                    self.updateLog(P, fmin, tuple([P + x*uN for x in bracket]))

                P0 = np.copy(P)


        if self.verbose:
            print(f'Powell finished after {i} iterations.')

        self.writeLog()
        return P, fmin


    def updateLog(self, P, fun, bracket, write=True):
        """
        Updates current content of the log.
        """
        self.log['P'] = np.vstack((self.log['P'], P))
        self.log['fun'] = np.append(self.log['fun'], fun)
        self.log['brackets'].append(bracket)
        
        self.log['nFunEval'].append(self.nFunEval)    
        self.log['nSim'].append(self.nSim)
        self.log['nExcept'].append(self.nExcept)

        if write:
            self.writeLog()
        
        
        

    def writeLog(self):
        """
        Saves the current log in as json file.
        """
        
        out = self.settings.out+'.json'
        with open(out, 'w') as fp:
            json.dump(self.log, fp, cls=NumpyEncoder)

    def isFinished(self):
        # return self.output i not None
        pass
