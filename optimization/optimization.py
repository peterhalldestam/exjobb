import numpy as np
from linemin import findBracket, Brent, goldenSectionSearch

#lowerBound = (-1., -4.)
#upperBound = (5., 5.)

class Powell:

    log = None

    def __init__(self, fun, P0, **kwargs):
    
        self.fun        =   fun
        self.P0         =   P0
        self.lowerBound =   kwargs.get('lb',        tuple(-1e+10*np.ones(len(P0))))
        self.upperBound =   kwargs.get('ub',        tuple(1e+10*np.ones(len(P0))))
        self.ftol       =   kwargs.get('ftol',      1e-2)
        self.maxIter    =   kwargs.get('maxIter',   10)
        self.verbose    =   kwargs.get('verbose',   False)

    def getp0(self, P, u):
        eps = 1e-10

        upperCross = (self.upperBound - P) / (u+eps)
        lowerCross = (self.lowerBound - P) / (u+eps)
        allCross = np.append(upperCross, lowerCross)
        
        xin = np.max(allCross[allCross <= 0.])
        xip = np.min(allCross[allCross > 0.])
        
        lineBounds = (xin, xip)
        lp = xip * np.linalg.norm(u)

        return (-eps, lp/10), lineBounds

    def run(self):#fun, P0, ftol, maxIter=10, lb=None, ub=None, verbose=False):
        """
        Powell's method for multidimensional minimization method.
        
        :param fun:     function handle to a ND objective function that is going to be minimized.
        :param P0:      initial starting point for the method
        :param ftol:    tolerance at which to terminate the minimization process.
        :param maxIter: maximum number of allowed (full) iterations of the method.
        :param verbose: prints information.
        """
        """
        if lb:
            lowerBound = lb
        else:
            lowerBound = tuple(-np.inf * np.ones(len(P0))) 
        if ub:
            upperBound = ub
        else:
            print('hej')
            upperBound = tuple(np.inf * np.ones(len(P0)))
        """
        

        def restrainedFun(P):
            if (P < self.lowerBound).any() or (P > self.upperBound).any():
                return 1e+10
            else:
                return fun(*P)

        P0 = self.P0
        Ptrack = np.array([P0])
        P = np.copy(P0)

        fun = self.fun        
        f0 = np.inf
        fmin = fun(*P0)
        
        i = 0
        while np.abs(f0-fmin) > self.ftol:
            i += 1
            if i > self.maxIter:
                if self.verbose:
                    print(f'Maximum number of iterations exceeded (maxIter = {maxIter}). Terminating Powell.')
                return Ptrack
            
            f0 = fmin
            
            basis = np.eye(len(P0))
            for _ in P0:
                for u in basis:
                    lineFun = lambda x: restrainedFun(P + x*u) 
                    
                    p0, lineBounds = self.getp0(P, u)
                    bracket = findBracket(lineFun, p0, lineBounds)          
                    xmin, fmin = Brent(lineFun, bracket)

                    P += xmin*u
                    Ptrack = np.vstack((Ptrack, P))
                     
                uN = P - P0
                basis[:-1] = basis[1:]
                basis[-1] = uN

                lineFun = lambda x: restrainedFun(P + x*uN)
                
                p0, lineBounds = self.getp0(P, uN)
                bracket = findBracket(lineFun, p0, lineBounds)
                xmin, fmin = Brent(lineFun, bracket)

                P += xmin*uN
                P0 = np.copy(P)
                Ptrack = np.vstack((Ptrack, P0))   
                
        if self.verbose:
            print(f'Powell finished after {i} iterations.')
            
        return Ptrack
    
    

