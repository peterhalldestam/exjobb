'''
Collection of helper functions and more...
'''

import sys, os
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import warnings

DREAMPATHS = ('/home/pethalld/DREAM/py', '/home/peterhalldestam/DREAM/py', '/home/hannber/DREAM/py')

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in DREAMPATHS:
        sys.path.append(dp)
    import DREAM

from DREAM import DREAMOutput

def concatenate(arr1, arr2):
    """
    Returns the concatenation of all entered arrays, with the initial
    timestep removed.

    :param arr1, [arr2,...]: Any number of 1d arrays to concatenate.
    """
    out = []
    out.append(arr1[:])
    out.append(arr2[1:])
    out = np.squeeze(np.array(out))
    return out

def visualizeCurrents(t, I_ohm, I_re, ax=None, show=False):
    """
    Plots the RE, Ohmic and total currents of given DREAM output file.

    :param t:       Simulation time.
    :param I_re:    RE current.
    :param I_ohm:   Ohmic current.
    :param ax:  matplotlib Axes object.
    """
    if ax is None:
        ax = plt.axes()

    # change units
    t *= 1e3        # s to ms
    I_re *= 1e-6    # A to MA
    I_ohm *= 1e-6

    ax.plot(t, I_ohm, 'r', label='Ohm')
    ax.plot(t, I_re,  'b', label='RE')
    ax.plot(t, I_ohm + I_re, 'k', label='total')

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (MA)')

    if show:
        plt.show()

    return ax

def getCQTime(t, I_ohm, tol=5e-2):
    """
    Calculates current quench time through interpolation.

    :param t:       Simulation time.
    :param I_ohm:   Ohmic current.
    :param tol:     Tolerance value.

	"""

    print(I_ohm)

    i80 = np.argmin(np.abs(I_ohm/I_ohm[0] - .8))
    i20 = np.argmin(np.abs(I_ohm/I_ohm[0] - .2))

    print(i80, i20)

    if np.abs(I_ohm[i80]/I_ohm[0] - .8) > tol:
	    warnings.warn(f'\nData point at 80% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')
    elif np.abs(I_ohm[i20]/I_ohm[0] - .2) > tol:
	    warnings.warn(f'\nData point at 20% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')

    t80 = scp.optimize.fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .8, x0=t[i80])
    t20 = scp.optimize.fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .2, x0=t[i20])

    return t20, t80, (t20 - t80) / .6


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
	Returnsout the Rechester-Rosenbluth diffusion operator for REs travelling at the
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
		dBB_mesh, xi_mesh, p_mesh = np.meshgrid(dBB, xi, p, indexing='ij')
		D = np.pi * R0 * q * dBB_mesh**2 * np.abs(xi_mesh) * p_mesh * c/(np.sqrt(1 + p_mesh**2))
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
