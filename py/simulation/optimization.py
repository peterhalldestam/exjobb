import numpy as np
from linemin import findBracket, Brent, goldenSectionSearch

class Powell:

    def __init__(self, fun, P0, **kwargs):  
        """
        Powell's method for multidimensional minimization.
        
        :param fun:     function handle to a ND objective function that is going to be minimized.
        :param P0:      initial starting point for the method
        :param ftol:    tolerance at which to terminate the minimization process.
        :param maxIter: maximum number of allowed (full) iterations of the method.
        :param verbose: prints information.
        """   
    
        self.baseFun    =   fun
        self.P0         =   P0
        self.log        =   None
        self.lowerBound =   kwargs.get('lb',        tuple(-1e+10*np.ones(len(P0))))
        self.upperBound =   kwargs.get('ub',        tuple(1e+10*np.ones(len(P0))))
        self.ftol       =   kwargs.get('ftol',      1e-2)
        self.maxIter    =   kwargs.get('maxIter',   10)
        self.verbose    =   kwargs.get('verbose',   False)


    def getLog(self):
        return self.log
    
    def setLowerBound(self, lb):
        self.lowerBound = lb
    
    def setUpperBound(self, ub):
        self.upperBound = ub

    def run(self):   
        """
        Runs the main optimization routine.
        Returns the parameter configuration P and function value f at the time of termination.
        """
        P0 = self.P0
        P = np.copy(P0)
        nD = len(P0)

        f0 = np.inf
        fmin = self._restrainedFun(P0)
        
        self.log = {'P': np.copy(P), 'fun': np.array([fmin])}
        
        # Main loop that updates the basis to avoid linear dependence.
        i = 0
        while np.abs(f0-fmin) > self.ftol:
            i += 1
            f0 = fmin
            
            if i > self.maxIter:
                if self.verbose:
                    print(f'Maximum number of iterations exceeded (maxIter = {self.maxIter}). Terminating Powell.')
                return P, fmin
            
            basis = np.eye(nD) # Resets basis (other methods not yet implemented).
            
            # Iterates through nD cycles of the basis vectors until all have been updated.
            for _ in range(nD):
            
                # Finds the minimum of the function along each direction in the basis.
                for u in basis:
                    lineFun = lambda x: self._restrainedFun(P + x*u) 
                    
                    b0, lineBounds = self._initBracket(P, u)
                    bracket = findBracket(lineFun, b0, lineBounds, verbose=self.verbose)          
                    
                    xmin, fmin = Brent(lineFun, bracket, verbose=self.verbose, maxIter=20)
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
                bracket = findBracket(lineFun, b0, lineBounds, verbose=self.verbose)
                
                xmin, fmin = Brent(lineFun, bracket, verbose=self.verbose, maxIter=20)
                P += xmin*uN
                P0 = np.copy(P)
                
                self.log['P'] = np.vstack((self.log['P'], P))
                self.log['fun'] = np.append(self.log['fun'], fmin)
                
        if self.verbose:
            print(f'Powell finished after {i} iterations.')
            
        return P, fmin
        
    def _restrainedFun(self, P):
        """
        Modified base function that incorporates parameter boundries in the optimization process.
        Any configuration outside of the specified domain automatically returns an arbitrary large number (10^10).
        """
        if (P < self.lowerBound).any() or (P > self.upperBound).any():
            return 1e+10
        else:
            return self.baseFun(*P)

    def _initBracket(self, P, u):
        """
        Method used to find two initial starting points when attempting to bracket a minimum.
        Additionaly, the upper and lower limits of the 1D line function are returned. 
        """
        # Arbitrary small number used to avoid dividing by zero and avoid errors if the vector is already located at the minimum.
        eps = 1e-10

        # Computes the values xi for which P + xi*u lies on the upper and lower boundries of the respective parameters.
        upperCross = (self.upperBound - P) / (u+eps)
        lowerCross = (self.lowerBound - P) / (u+eps)
        allCross = np.append(upperCross, lowerCross)
        
        # Finds the positive and negative values for xi that minimize abs(xi), i.e. the points at which the first boundries are crossed.
        xin = np.max(allCross[allCross <= 0.])
        xip = np.min(allCross[allCross > 0.])
        
        lineBounds = (xin, xip)
        lp = xip * np.linalg.norm(u)

        return (-eps, lp/10), lineBounds

