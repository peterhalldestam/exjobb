import numpy as np
from linemin import findBracket, Brent, goldenSectionSearch

bound_lower = (-4., -4.)
bound_upper = (-4., -4.)

def naive(fun, P0, ftol, maxIter=3):

    basis = np.eye(len(P0))
    
    Ptrack = np.array([P0])
    
    P = P0
    
    i = 0
    
    f0 = np.inf
    fmin = fun(*P0)
    while np.abs(f0-fmin) > ftol:
        i += 1
        if i > maxIter:
            break
        
        f0 = fmin
        for u in basis:
            lineFun = lambda x: fun(*(P + x*u)) 
        
            p0 = (-5, 0)
            bracket = findBracket(lineFun, p0)
            
            xmin, fmin = Brent(lineFun, bracket)

            P += xmin*u
            
            Ptrack = np.vstack((Ptrack, P))
        
    return Ptrack


def powell(fun, P0, ftol, maxIter=10, verbose=False):

    Ptrack = np.array([P0])
    P = np.copy(P0)
    
    f0 = np.inf
    fmin = fun(*P0)
    
    i = 0
    while np.abs(f0-fmin) > ftol:
        i += 1
        if i > maxIter:
            if verbose:
                print(f'Maximum number of iterations exceeded (maxIter = {maxIter}). Terminating Powell.')
            return Ptrack
        
        f0 = fmin
        for _ in range(len(P0)):
            basis = np.eye(len(P0))
        
            for u in basis:
                lineFun = lambda x: fun(*(P + x*u)) 
            
                p0 = (-10, 1e-6)
                bracket = findBracket(lineFun, p0)          
                xmin, fmin = Brent(lineFun, bracket)

                P += xmin*u
                Ptrack = np.vstack((Ptrack, P))
                 
            uN = P - P0
            basis[:-1] = basis[1:]
            basis[-1] = uN

            lineFun = lambda x: fun(*(P + x*uN))
            
            p0 = (-10, 1e-6)
            bracket = findBracket(lineFun, p0)
            xmin, fmin = Brent(lineFun, bracket)

            P += xmin*uN
            P0 = np.copy(P)
            Ptrack = np.vstack((Ptrack, P0))   
            
    if verbose:
        print(f'Powell finished after {i} iterations.')
        
    return Ptrack
    
    

