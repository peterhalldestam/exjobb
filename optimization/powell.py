import numpy as np
from linemin import findBracket, Brent, goldenSectionSearch

def naiveOpt(fun, P0, ftol, maxIter=3):

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


def powellOpt(fun, P0, ftol, maxIter=3):

    basis = np.eye(len(P0))
    
    Ptrack = np.array([P0])
    
    P = np.copy(P0)
    
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
        
            p0 = (-10, 0)
            bracket = findBracket(lineFun, p0)
            
            xmin, fmin = goldenSectionSearch(lineFun, bracket)

            P += xmin*u
            
            Ptrack = np.vstack((Ptrack, P))
            
            print(P)
            
        uN = P - P0
        basis[:-1] = basis[1:]
        basis[-1] = uN

        lineFun = lambda x: fun(*(P + x*uN)) 
        
        p0 = (-10, 0)
        bracket = findBracket(lineFun, p0)
        
        xmin, fmin = Brent(lineFun, bracket)

        P += xmin*uN
        P0 = np.copy(P)
        Ptrack = np.vstack((Ptrack, P0))     
        
    return Ptrack
    
    

