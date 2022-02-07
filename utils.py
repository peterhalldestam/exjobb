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


def getCQTime(I, t, tol=.02):
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
		msg = f'\nUnable to determine time of 80% amplitude within {tol*100}% margin.'
		warnings.warn(msg)
		return None
	elif np.abs(I[i20]/I[0] - 0.2) > tol:
		msg = f'\nUnable to determine time of 20% amplitude within {tol*100}% margin.'
		warnings.warn(msg)
		return None

	return (t[i20] - t[i80]) / 0.6
