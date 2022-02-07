'''
Collection of helper functions and more...
'''

import sys, os
import numpy as np

DREAMPATHS = ('/home/pethalld/DREAM/py', '/home/peterhalldestam/DREAM/py', '/home/hannber/DREAM/py')

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in DREAMPATHS:
        sys.path.append(dp)
    import DREAM


def getCQTime(I, t, tol=.05):
	"""
	If possible, calculates current quench time.
	Returns None otherwise.

    :param I:   1D array of Ohmic current data over time.
    :param t:   corresponding array of timesteps.
    :param tol: tolerance value.
	"""
	assert len(I) == len(t)

	i80 = np.argmin(np.abs(I/I[0] - 0.8))
	i20 = np.argmin(np.abs(I/I[0] - 0.2))

	if np.abs(I[i80]/I[0] - 0.8) > tol:
		msg = f'\nData point at 80% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.'
		warnings.warn(msg)
	elif np.abs(I[i20]/I[0] - 0.2) > tol:
		msg = f'\nData point at 20% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.'
		warnings.warn(msg)
		
	t0_80 = t[i_80]
	t0_20 = t[i_20]
	
	t_80 = fsolve(lambda x: np.interp(x, t, I)/I[0]-0.8, x0 = t0_80)
	t_20 = fsolve(lambda x: np.interp(x, t, I)/I[0]-0.2, x0 = t0_20)
	
	return (t_20 - t_80) / 0.6
