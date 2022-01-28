#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import scipy.constants
import sys
import scipy.io as sio
import subprocess

sys.path.append('/home/hannber/DREAM/py') # Modify this path as needed on your system
sys.path.append('/home/hannber/exjobb')
from DREAM import DREAMSettings
from DREAM import runiface
from DREAM.Settings import RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.RunawayElectrons as Runaways

import ITER as Tokamak

ds_init = DREAMSettings()

""" Tokamak geometry """
R0 = 6.2		# [m]
a = 1.79		# [m]
r_wall = 1.35*a	# [m]
B0 = 5.3		# [T]
nr = 100

ds_init.radialgrid.setType(RadialGrid.TYPE_CYLINDRICAL)
ds_init.radialgrid.setMinorRadius(a)
ds_init.radialgrid.setWallRadius(r_wall)
ds_init.radialgrid.setNr(nr)
ds_init.radialgrid.setB0(B0)

""" Disable kinetic grids and set solvers """
ds_init.hottailgrid.setEnabled(False)
ds_init.runawaygrid.setEnabled(False)

ds_init.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
ds_init.solver.setType(Solver.NONLINEAR)

""" Create initial density, temperature and current profiles """
r = np.linspace(0, a, nr)

_, T_init_profile = Tokamak.getInitialTemperature(r)
_, ne_profile = Tokamak.getInitialDensity(r)
_, j_prof = Tokamak.getCurrentDensity(r)

fig, ax = plt.subplots(1, 3, figsize=(13, 5))

ax[0].plot(r, T_init_profile*1e-3)
ax[0].set_title('Initial temperature profile')
ax[0].set_xlabel('minor radial coordinate [m]')
ax[0].set_ylabel('temperature [keV]')

ax[1].plot(r, ne_profile)
ax[1].set_title('Initial density profile')
ax[1].set_xlabel('minor radial coordinate [m]')
ax[1].set_ylabel('density')

ax[2].plot(r, j_prof*1e-6)
ax[2].set_title('Target current profile')
ax[2].set_xlabel('minor radial coordinate [m]')
ax[2].set_ylabel('current [MA]')

plt.show()


""" Prescribe plasma parameters """
ds_init.eqsys.E_field.setPrescribedData(0)
ds_init.eqsys.T_cold.setPrescribedData(T_init_profile, radius=r)
ds_init.eqsys.n_i.addIon('D', Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=ne_profile, r=r, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
ds_init.eqsys.n_i.addIon(name='Ar', Z=18, Z0=5, iontype=Ions.IONS_DYNAMIC, n=ne_profile*0.2, r=r)


""" First initialization simulation: calculates conductivity """
ds_init.timestep.setTmax(1e-6) # Only need one very small step
ds_init.timestep.setNt(1)
ds_init.other.include('fluid/conductivity')

do = runiface(ds_init, 'init/output_conductivity.h5', quiet=False)

""" Second intialization: set target current profile """
I_p = 15e6
j_target = I_p * j_prof / do.grid.integrate(j_prof)
E = j_target / do.other.fluid.conductivity[-1,:]

ds_init.eqsys.E_field.setPrescribedData(E, radius=do.grid.r)
do = runiface(ds_init, 'init/output_current.h5', quiet=False) # Using the same time step as before

""" Main simulations: apply self-consistent temperature evolution """
# Change to main time configuration
t_max = 80e-3
nt = 12e3
ds_init.timestep.setTmax(t_max)
ds_init.timestep.setNt(nt)
ds_init.timestep.setNumberOfSaveSteps(200)

# Enable avalanche, hottail and Dreicer generation
ds_init.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID)
ds_init.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

ds_init.eqsys.f_hot.setInitialProfiles(rn0=r, n0=ne_profile, rT0=r, T0=T_init_profile)
ds_init.eqsys.n_re.setHottail(Runaways.HOTTAIL_MODE_ANALYTIC_ALT_PC)

# Enable self consistent evolution of E-field
ds_init.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds_init.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=R0)

# Enable magnetic pertubations that will allow for radial transport
ds_init.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds_init.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

# Run simulation for different uniform perturbations
dBB_list = np.linspace(0.05e-3, 1e-3, 20, dtype=float)
for i, dBB in enumerate(dBB_list):

	ds_main = DREAMSettings(ds_init)
	ds_main.other.include('fluid')
	ds_main.other.include('transport')
	
	ds_main.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB)
	do = runiface(ds_main, f'output/output_dBB_{dBB:.5f}.h5', quiet=False)
	#do = runiface(ds_main, f'output/output_dBB_{i}.h5', quiet=False)








#subprocess.Popen(['rm','output_main.h5'])
#do = runiface(ds_main, 'output_main.h5', quiet=False)


