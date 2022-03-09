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

def join(dataStr: str, *dos: DREAMOutput, time=False, radius=False) -> np.ndarray:
    """
    Joins data obtained from any number of DREAM output objects in *dos.
    Recognizing axis 0 as the temporal dimension in every data array, this
    function appends the arrays into a single time sequence.

    :param dataStr:     DREAM output data to join, e.g. 'eqsys.T_cold.data'.
    :param dos:         Any number of DREAM output objects ordered in time.
    :param time:        If true, the data arrays are treated as the simulation times.
    :param radius:      If true, return the radial grid from the first output.
    """
    t = 0
    q = np.array([])
    for i, do in enumerate(*dos):
        obj = do
        for attr in dataStr.split('.'):
            if attr.endswith('()'):
                attr = attr[:-2]
                obj = getattr(obj, attr)()
            else:
                obj = getattr(obj, attr)

        obj = np.array(obj)
        if not i:
            q = np.array(obj)

            if radius:
                return q
        else:
            q = np.append(q, t + obj[1:], axis=0)
        if time:
            t += obj[-1]
    return q

def getDensityProfile(do, n, c):
    """
    Returns a density profile with the total (volume integrated) number of
    particles the same as if the density was radially constant n.
    """
    profile = .5 * (1 + np.tanh(c * ((do.grid.r/do.grid.a) - .5)))
    return do.grid.r, n * profile * do.grid.integrate(1) / do.grid.integrate(profile)

def visualizeTemperature(r, T, times=[0,-1], ax=None, show=False):
    """
    Plots the temperature profiles of selected times (times=[0,-1] plots the
    temperature at the first and last timestep).

    :param r:       Minor radius.
    :param T:       Temperature distribution (NT x NR).
    :param times:   Timesteps to plot the radial temperature distribution at.
    :param ax:      matplotlib Axes object.
    :param show:    Show the figure of the temperature profiles.
    """
    if ax is None:
        ax = plt.axes()

    # change units
    T *= 1e-3   # eV to keV

    for ti in times:
        ax.plot(r, T[ti,:], label=ti)

    ax.legend(title='Timestep indices:')
    ax.set_xlabel('minor radius (m)')
    ax.set_ylabel('temperature (keV)')

    if show:
        plt.show()

    return ax

def visualizeTemperatureEvolution(t, T, radii=[0], ax=None, show=False):
    """
    Plots the temperature evolution at given radial nodes (radii=[0] plots the
    temperature at the core over time).

    :param t:       Simulation time.
    :param T:       Temperature distribution (NT x NR).
    :param radii:   Radial nodes to plot temperature evolution.
    :param ax:      matplotlib Axes object.
    :param show:    Show the figure of the temperature evolutions.
    """
    if ax is None:
        ax = plt.axes()

    # change units
    # T *= 1e-3   # eV to keV
    # t *= 1e3    # s to ms

    for ri in radii:
        ax.plot(t, T[:,ri], label=ri)

    ax.legend(title='Radial node indices:')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('temperature (keV)')

    if show:
        plt.show()

    return ax

def visualizeCurrents(t, I_ohm, I_re, I_tot, log=False, ax=None, show=False):
    """
    Plots the RE, Ohmic and total currents.

    :param t:       Simulation time.
    :param I_re:    RE current.
    :param I_ohm:   Ohmic current.
    :param I_tot:   Total current.
    :param ax:      matplotlib Axes object.
    :param show:    Show the figure of the currents.
    """
    if ax is None:
        ax = plt.axes()

    # change units
    t *= 1e3        # s to ms
    # I_re *= 1e-6    # A to MA
    # I_ohm *= 1e-6

    ax.plot(t, I_ohm, 'r', label='Ohmic')
    ax.plot(t, I_re,  'b', label='REs')
    ax.plot(t, I_tot, 'k', label='total')

    ax.legend(title='Currents:')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (MA)')

    if log:
        ax.set_yscale('log')

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
    i80 = np.argmin(np.abs(I_ohm/I_ohm[0] - .8))
    i20 = np.argmin(np.abs(I_ohm/I_ohm[0] - .2))

    if np.abs(I_ohm[i80]/I_ohm[0] - .8) > tol:
	    warnings.warn(f'\nData point at 80% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')
    elif np.abs(I_ohm[i20]/I_ohm[0] - .2) > tol:
	    warnings.warn(f'\nData point at 20% amplitude was not found within a {tol*100}% margin, accuracy of interpolated answer may be affected.')

    t80 = scp.optimize.fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .8, x0=t[i80])[0]
    t20 = scp.optimize.fsolve(lambda x: np.interp(x, t, I_ohm)/I_ohm[0] - .2, x0=t[i20])[0]

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
