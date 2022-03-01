#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import scipy.constants
import sys
import scipy.io as sio
import subprocess

sys.path.append('/home/hannber/DREAM/py') # Modify these paths as needed on your system
sys.path.append('/home/hannber/exjobb')
from DREAM import DREAMSettings
from DREAM import runiface
from DREAM.Settings import RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.TransportSettings as Transport

import ITER as Tokamak
import utils

ds_init = DREAMSettings()

""" Tokamak geometry """
nr = 40
R0 = Tokamak.R0		# [m]
Tokamak.setMagneticField(ds_init, nr=nr)

""" Disable kinetic grids and set solvers """
ds_init.hottailgrid.setEnabled(False)
ds_init.runawaygrid.setEnabled(False)

ds_init.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
ds_init.solver.setType(Solver.NONLINEAR)

""" Create initial density, temperature and current profiles """
rT0, T_init_profile = Tokamak.getInitialTemperature()
rn0, ne_profile = Tokamak.getInitialDensity()
rj0, j_prof = Tokamak.getCurrentDensity(nr=nr)

fig, ax = plt.subplots(1, 3, figsize=(13, 5))

ax[0].plot(rT0, T_init_profile*1e-3)
ax[0].set_title('Initial temperature profile')
ax[0].set_xlabel('minor radial coordinate [m]')
ax[0].set_ylabel('temperature [keV]')

ax[1].plot(rn0, ne_profile)
ax[1].set_title('Initial density profile')
ax[1].set_xlabel('minor radial coordinate [m]')
ax[1].set_ylabel('density')

ax[2].plot(rj0, j_prof*1e-6)
ax[2].set_title('Target current profile')
ax[2].set_xlabel('minor radial coordinate [m]')
ax[2].set_ylabel('current [MA]')

plt.show()

""" Prescribe plasma parameters """
ds_init.eqsys.E_field.setPrescribedData(0)
ds_init.eqsys.T_cold.setPrescribedData(T_init_profile, radius=rT0)
ds_init.eqsys.n_i.addIon('D', Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=ne_profile*0.9, r=rn0, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
ds_init.eqsys.n_i.addIon(name='Ar', Z=18, Z0=5, iontype=Ions.IONS_DYNAMIC, n=ne_profile*0.1, r=rn0)


""" First initialization simulation: calculates conductivity """
ds_init.timestep.setTmax(1e-6) # Only need one very small step
ds_init.timestep.setNt(1)
ds_init.other.include('fluid/conductivity')

do = runiface(ds_init, 'init/output_conductivity.h5', quiet=False)

""" Second intialization: set target current profile """
I_p = 15e6	# [A]
j_target = I_p * 2*np.pi * j_prof / do.grid.integrate(j_prof)
E = j_target / do.other.fluid.conductivity[-1,:]

ds_init.eqsys.E_field.setPrescribedData(E, radius=do.grid.r)
do = runiface(ds_init, 'init/output_current.h5', quiet=False) # Using the same time step as before

""" Thermal quench simulation: apply self-consistent temperature evolution """
# Change to time configuration for TQ
t_max = 15e-3
nt = 4e3
t = np.linspace(0,t_max,nt)
ds_init.timestep.setTmax(t_max)
ds_init.timestep.setNt(nt)
ds_init.timestep.setNumberOfSaveSteps(100)

# Enable avalanche, hottail and Dreicer generation
ds_init.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID)
ds_init.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

ds_init.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=ne_profile, rT0=rT0, T0=T_init_profile)
ds_init.eqsys.n_re.setHottail(Runaways.HOTTAIL_MODE_ANALYTIC_ALT_PC)

# Enable self consistent evolution of E-field
ds_init.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds_init.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=R0)

# Enable magnetic pertubations that will allow for radial transport
ds_init.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds_init.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

# Run simulation for different uniform perturbations
#dBB_list = np.linspace(1e-3, 5e-3, 5)
r_dBB = np.array([0, 0.1])
dBB = 2e-3 * np.ones(len(r_dBB))

Drr, xi_grid, p_grid = utils.getRRCoefficient(dBB, R0=Tokamak.R0)
Drr = np.tile(Drr, (int(nt),1,1,1))

pstar = 0.5

ds1 = DREAMSettings(ds_init)
ds1.other.include('fluid')
ds1.other.include('scalar')

ds1.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB[0])
ds1.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)

ds1.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
ds1.eqsys.n_re.transport.setSvenssonPstar(pstar)
# Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
ds1.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r_dBB, p=p_grid, xi=xi_grid, interp1d=Transport.INTERP1D_NEAREST)
ds1.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

#do = runiface(ds_main, f'output/output{i}.h5', quiet=False)
do = runiface(ds1, f'output/output1.h5', quiet=False)

""" Current quench simulation: reduce magnetic pertubation strength """
ds2 = DREAMSettings(ds1)

# Change to time configuration for CQ
t_max = 60e-3
nt = 15e3
t = np.linspace(0,t_max,nt)
ds2.timestep.setTmax(t_max)
ds2.timestep.setNt(nt)
ds2.timestep.setNumberOfSaveSteps(150)


dBB = 1e-5 * np.ones(len(r_dBB))#0.1e-3 * np.ones(len(r_dBB))
ds2.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB[0])
ds2.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)

Drr, xi_grid, p_grid = utils.getRRCoefficient(dBB, R0=Tokamak.R0)
Drr = np.tile(Drr, (int(nt),1,1,1))

ds2.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r_dBB, p=p_grid, xi=xi_grid, interp1d=Transport.INTERP1D_NEAREST)

do = runiface(ds2, f'output/output2.h5', quiet=False)


