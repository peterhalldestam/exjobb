'''
Collection of helper functions and more...
'''

import sys, os
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt


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
        t       = do.grid.t[1:] * 1e3
        I_ohm   = do.eqsys.j_ohm.current()[1:] * 1e-6
        I_re    = do.eqsys.j_re.current()[1:] * 1e-6
        I_tot   = do.eqsys.j_tot.current()[1:] * 1e-6

        ax.plot(t, I_ohm, 'r', label='Ohm')
        ax.plot(t, I_re, 'g', label='RE')
        ax.plot(t, I_tot, 'b', label='total')

    except AttributeError as err:
        raise Exception('Output does not include needed data.') from err

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (MA)')

    if show:
        plt.show()

    return ax


def getCQTime(do, tol=.05):
	"""
	Calculates current quench time through interpolation.

    :param do:  DREAM output object.
    :param tol: tolerance value.
	"""
    I_ohm = do.eqsys.j_ohm.current()[1:]
	i80 = np.argmin(np.abs(I_ohm/I_ohm[0] - .8))
	i20 = np.argmin(np.abs(I_ohm/I_ohm[0] - .2))

	if np.abs(I_ohm[i80]/I_ohm[1] - 0.8) > tol:
		warnings.warn(f'\nData point at 80% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')
	elif np.abs(I_ohm[i20]/I_ohm[1] - 0.2) > tol:
		warnings.warn(f'\nData point at 20% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')

    # t = do.grid.t[1:]
	t_80 = fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .8, x0=t[i_80])
	t_20 = fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .2, x0=t[i_20])

	return (t_20 - t_80) / 0.6


def getRRCoefficient(dBB, q=1, R0=1., svensson = True):
	"""
	Calculates the Rechester-Rosenbluth diffusion operator for runaway electrons under the assumption that v_p = c.

	:param dBB:	        scalar or 1D array containing the magnetic pertubation spatial profile.
	:param q:	        scalar or 1D array representing the tokamak safety factor.
	:param R0:	        major radius of the tokamak [m].
	:param svensson:	boolean that if true calculates the coefficient on a pi-xi grid.
	"""
	if not isinstance(dBB, (int, float)):
		assert len(dBB.shape) == 1

		if not isinstance(q, (int, float)):
			assert len(q) == dBB.shape[-1]

	c = scp.constants.c

	if svensson:
		p_grid = np.linspace(0, 1.5, 60)
		xi_grid = np.linspace(-1., 1., 45)
		dBB_mesh, xi_mesh, p_mesh = np.meshgrid(dBB, xi_grid, p_grid, indexing='ij')

		D = np.pi * R0 * q * (dBB_mesh)**2 * np.abs(xi_mesh)*p_mesh * c/(np.sqrt(1 + p_mesh**2))

		return D, xi_grid, p_grid

	else:
		D = np.pi * R0 * q * c * (dBB)**2

		return D

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
