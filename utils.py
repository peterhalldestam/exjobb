'''
Collection of helper functions and more...
'''

import sys, os
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
# import wanings123


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
        t = do.grid.t[1:] * 1e3                   # [ms]
        I_ohm = do.eqsys.j_ohm.current() * 1e-6   # [MA]
        I_re = do.eqsys.j_re.current()[1:] * 1e-6
        I_tot = do.eqsys.j_tot.current()[1:] * 1e-6

        ax.plot(t, I_ohm, 'r', label='Ohm')
        ax.plot(t, I_re,  'g', label='RE')
        ax.plot(t, I_tot, 'b', label='total')

    except AttributeError as err:
        raise Exception('Output object does not include needed data.') from err

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
    # I_ohm = do.eqsys.j_ohm.current()
    # i80 = np.argmin(np.abs(I_ohm/I_ohm[0] - .8))
	# i20 = np.argmin(np.abs(I_ohm/I_ohm[0] - .2))

	if np.abs(I_ohm[i80]/I_ohm[1] - .8) > tol:
		warnings.warn(f'\nData point at 80% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')
	elif np.abs(I_ohm[i20]/I_ohm[1] - .2) > tol:
		warnings.warn(f'\nData point at 20% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')

    # t = do.grid.t[1:]
	t_80 = fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .8, x0=t[i_80])
	t_20 = fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .2, x0=t[i_20])

	return (t_20 - t_80) / .6


def getQuadraticMagneticPerturbation(ds, dBB0, dBB1):
    """
    Returns a quadratic profile for the magnetic pertubation, given the
    paramaters dBB0 (defining the integral of dBB) and dBB1. The latter controls
    the profile shape: dBB(r) ~ 1 + dBB1 * r^2.
    """
    try:
        r = np.linspace(0, ds.radialgrid.a, ds.radialgrid.nr)
        assert r.size > 1
    except AttributeError as err:
        raise Exception('Settings object does not include needed data.') from err

    dBB = dBB0 * (1 + dBB1 * r**2)
    return r, dBB


def getDiffusionOperator(dBB, q=1, R0=1., svensson=True):
	"""
	Returns the Rechester-Rosenbluth diffusion operator for REs travelling at the
    speed of light.

	:param dBB:	        Scalar or 1D array containing the magnetic pertubation
                        spatial profile.
	:param q:	        Scalar or 1D array representing the tokamak safety factor.
	:param R0:	        Major radius of the tokamak [m].
	:param svensson:	Boolean that if true calculates the coefficient on a
                        pi-xi grid.
	"""
	if not isinstance(dBB, (int, float)):
		assert len(dBB.shape) == 1

		if not isinstance(q, (int, float)):
			assert len(q) == dBB.shape[-1]

	c = scp.constants.c

	if svensson:
		p = np.linspace(0, 1.5, 60)
		xi = np.linspace(-1., 1., 45)
		dBB, xi, p = np.meshgrid(dBB, xi, p, indexing='ij')
		D = np.pi * R0 * q * dBB**2 * np.abs(xi) * p * c/(np.sqrt(1 + p**2))
		return D, xi, p

	else:
		D = np.pi * R0 * q * c * dBB**2
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
