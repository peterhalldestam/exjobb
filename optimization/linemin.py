import numpy as np
import scipy.constants as const

def findBracket(fun, p0, bounds, gamma=2):
    """
    Simple algorithm used to initially bracket a local minimum by expontentially increasing step sizes.
    
    :param fun:     function handle to a 1D function where the minimum is going to be bracketed.
    :param p0:      tuple or list object containing two of the points that are going to be included in the bracket.
    :param gamma:   constant factor by which the step size is increased.
    """

    ax = p0[0]; bx = p0[1]
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
    
    while fc <= fb:
        gamma *= 2
        cx = bx + gamma*(bx-ax)
        
        if cx > ux:
            cx = ux
            break
        elif cx < lx:
            cx = lx
            break
        
        fc = fun(cx)
    
    if swap:
        return (cx, bx, ax)
    else:
        return (ax, bx, cx)

def goldenSectionSearch(fun, bracket, tol=1e-2, verbose=False):
    """
    1D minimization function using the Golden Section Search method.

    :param fun:     function handle of the function that is to be minimized.
    :param bracket: tuple or list object containing three abscissas (a,b,c), where a<b<c, fun(b)<fun(a) and fun(b)<fun(c).
    :param tol:     tolerance at which to terminate the minimization process.
    :param verbose: prints information.
    """

    R = const.golden - 1; C = 2 - const.golden
    ax = bracket[0]; bx = bracket[1]; cx = bracket[2]

    x0 = ax
    x3 = cx
    
    if np.abs(cx-bx) > np.abs(bx-ax):
        x1 = bx
        x2 = bx+C*(cx-bx)
    else:
        x2 = bx
        x1 = bx-C*(bx-ax)
        
    f1 = fun(x1)
    f2 = fun(x2)
    
    i = 0
    while np.abs(x3-x0) > tol:
        i += 1
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = R*x1 + C*x3
            f1 = f2
            f2 = fun(x2)       
        else:
            x3 = x2
            x2 = x1
            x1 = R*x2 + C*x0
            f2 = f1
            f1 = fun(x1)
            
    if verbose:      
        print(f'Golden Section Search finished after {i} iterations.')  
    
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

def Brent(fun, bracket, tol=1e-2, maxIter=1000, verbose=False):
    """
    1D minimization function using the Brent's method.

    :param fun:     function handle of the function that is to be minimized.
    :param bracket: tuple or list object containing three abscissas (a,b,c), where a<b<c, fun(b)<fun(a) and fun(b)<fun(c).
    :param tol:     tolerance at which to terminate the minimization process.
    :param matIter: maximum number of allowed iterations
    :param verbose: prints information.
    """

    C = 2 - const.golden; ZEPS = 1e-10 
    
    a = bracket[0]; b = bracket[2]
    v = bracket[1]; w = v; x = v
    fx = fun(x); fv = fx; fw = fx
    e = 0.   # Distance moved on step before last
    
    for i in range(maxIter):
        xm = 0.5*(a+b)
        tol1 = tol*np.abs(x) + ZEPS
        tol2 = 2.*tol1
        
        gold = True
        
        # "Doneness" test
        if(np.abs(x-xm) <= (tol2-0.5*(b-a))):
            if verbose:
                print(f'Brent finished after {i+1} iterations.')
            break
            
        # Trial parabolic fit
        if (np.abs(e) > tol1):
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q - (x-w)*r
            q = 2.*(q-r)
            if(q > 0): 
                p = -p
            q = np.abs(q)
            etemp = e
            e = d
            
            # Determines acceptability of parabolic fit
            if (abs(p) < abs(0.5*q*etemp)) and (p > q*(a-x)) and (p < q*(b-x)):
                d = p/q
                u = x+d
                if (u-a < tol2) or (b-u < tol2):
                    d = tol1 * np.sign(xm-x)
                gold = False
            
        # Golden section step (into the larger of the two segments)
        if (x >= xm) and gold:
            e = a-x
            d = C*e
        elif gold:
            e = b-x 
            d = C*e

        #  Applies step of length d obtained through golden section or parabolic fit
        if(abs(d) >= tol1):
            u = x+d
        else:
            u = x + tol1*np.sign(d)

        fu = fun(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x

            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u

            if (fu <= fw) or (w == x):
                v = w
                fv = fw
                w = u
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
                
    if verbose and i+1 == maxIter:
        print(f'Maximum number of iterations exceeded (maxIter = {maxIter}). Terminating minimization process.')
                
    return x, fx
