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

def checkElectronDensityRatio(do, exc=None, tol=1e-2):
    """
    Checks whether given output contains instances where the assumption
    n_re << n_cold or not. It is invalid when the ratio n_re / n_cold is greater
    that some tolerance tol. If invalid, a warning is shown (or if user provides
    an exception, it will be raised.

    :param do:      DREAM output object.
    :param exc:     Exception to raise.
    :param tol:     Tolerance < n_re / n_cold.
    """
    if exc is not None and not issubclass(exc, Exception):
        raise AttributeError('Expected exc to be an Exception subclass.')

    n_re = do.eqsys.n_re.data
    n_cold = do.eqsys.n_cold.data
    if np.max(n_re / n_cold) > tol:
        msg = f'n_re / n_cold > {tol}'
        if exc is None:
            warnings.warn(msg)
        else:
            raise exc(msg)

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
            q = np.array(obj[1:])

            if radius:
                return np.array(obj)
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

    for ti in times:
        ax.plot(r, T[ti,:] * 1e-3, label=ti)

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

    for ri in radii:
        ax.loglog(t * 1e3, T[:,ri] * 1e-3, label=ri)

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

    ax.plot(t * 1e3, I_ohm * 1e-6, 'r', label='Ohmic')
    ax.plot(t * 1e3, I_re * 1e-6,  'b', label='REs')
    ax.plot(t * 1e3, I_tot * 1e-6, 'k', label='total')

    ax.legend(title='Currents:')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (MA)')

    if log:
        ax.set_yscale('log')

    if show:
        plt.show()

    return ax

def getQuadraticMagneticPerturbation(ds, dBB1, dBB2):
    """
    Returns a quadratic profile for the magnetic pertubation, given the
    paramaters dBB1 and dBB2.
    """
    try:
        r = np.linspace(0, ds.radialgrid.a, ds.radialgrid.nr)
        assert r.size > 1
    except AttributeError as err:
        raise Exception('Settings object does not include needed data.') from err

    dBB = dBB1 * (1 + dBB2 * r**2)
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
