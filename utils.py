'''
Collection of helper functions and more...
'''

import sys, os
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

import ITER as tokamak

DREAMPATHS = ('/home/pethalld/DREAM/py', '/home/peterhalldestam/DREAM/py', '/home/hannber/DREAM/py')

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in DREAMPATHS:
        sys.path.append(dp)
    import DREAM

def visualizeCurrents(do, ax=None, show=False):
    """
    Plots the RE, Ohmic and total currents of given DREAM output file.

    :param do:  DREAM output object.
    :param ax:  matplotlib Axes object.
    """
    if ax is None:
        ax = plt.axes()

    try:
        t = do.grid.t * 1e3
        I_ohm = do.eqsys.j_ohm.current()
        I_re = do.eqsys.j_re.current()
        I_tot = do.eqsys.j_tot.current()

        ax.plot(t, 1e-6*do.eqsys.j_ohm.current(), 'r', label='Ohm')
        ax.plot(t, 1e-6*do.eqsys.j_re.current(), 'g', label='RE')
        ax.plot(t, 1e-6*do.eqsys.j_tot.current(), 'b', label='total')

    except AttributeError as err:
        raise Exception('Output does not include needed data.') from err

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (MA)')

    if show:
        plt.show()

    return ax


def getCQTime(I, t, tol=.05):
	"""
	Calculates current quench time through interpolation.

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


def getRRCoefficient(dBB, q=1, R0=tokamak.R0):
	"""
	Calculates the Rechester-Rosenbluth diffusion operator for runaway electrons under the assumption that v_p = c.

	:param dBB:	0-2D array containing the magnetic pertubation spatial/transient profile.
	:param q:	scalar or 1D array representing the tokamak safety factor.
	:param R0:	major radius of the tokamak [m].
	"""

	if not isinstance(dBB, (int, float)):
		assert len(dBB.shape) <= 2

		if not isinstance(q, (int, float)):
			assert len(q) == dBB.shape[-1]

	return np.pi * R0 * q * scp.constants.c * (dBB)**2


def terminate(sim, Tstop):
    """
    Returns true if the temperature reaches Tstop. Used as termination function
    during the thermal quench to determine when to lower/turn off the magnetic
    pertubation dBB (in the end of the TQ?).

    :param sim:     libdreampyface 'Simulation' object.
    :param Tstop:   temperature [eV] at which to terminate the simulation.
    """
    temperature = sim.unknowns.getData('T_cold')
    return temperature['x'][-1,0] < Tstop
