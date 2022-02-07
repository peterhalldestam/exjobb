#!/usr/bin/env python3

import sys
import numpy as np

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/pethalld/DREAM/py')
    sys.path.append('/home/peterhalldestam/DREAM/py')
    sys.path.append('/home/hannber/DREAM/py')
    import DREAM

from DREAM import DREAMSettings, runiface
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as RunawayElectrons
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

import ITER as Tokamak

NT = 6e3

def getBaseline(n=Tokamak.ne0):
    """
    Generate baseline scenario. This initialization consists of two short
    simulation runs: the first is used to calculate the conductivity, which is
    then used to set a desired current density profile.
    """
    ds = DREAMSettings()

    # set analytic toroidal magnetic field
    Tokamak.setMagneticField(ds, visualize=False)

    # Collision settings (THESE NEED TO BE UNDERSTOOD IN MORE DETAIL...)
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONAL

    # Disable kinetic grids (w13:0e run purely fluid simulations)
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    # Background ion density
    ds.eqsys.n_i.addIon('D', Z=1, iontype=Ions.IONS_DYNAMIC, Z0=1, n=n,
                        opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE) # UNDERSTAND THIS!!!

    # Solver
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    ds.solver.setType(Solver.NONLINEAR)

    # Set temporary time stepper options during initialization
    ds.timestep.setTmax(1e-11)
    ds.timestep.setNt(1)

    # Include all fluid `OtherQuantity`s
    ds.other.include('fluid')

    # get the exponential-decay temperature evolution
    # tT, rT, T0 = Tokamak.g13:0etTemperatureEvolution(tau0=1e-3)

    rT, T0 = Tokamak.getInitialTemperature()
    ############################################################################
    # STEP 1 : Calculate conductivity

    # Prescribe temperature (constant during initialization)
    ds.eqsys.T_cold.setPrescribedData(T0, radius=rT, times=[0])

    # Prescribe dummy electric field
    ds.eqsys.E_field.setPrescribedData(1e-4)

    # Calculate the conductivity
    do = runiface(ds, quiet=False)

    ############################################################################
    # STEP 2 : Obtain initia13:0l current density profile

    # Prescribe the exponential-decay temperature
    #ds.eqsys.T_cold.setPrescribedData(T0, radius=rT, times=tT)

    # Obtain the initial electric field from the conductivity calculation
    rj, j = Tokamak.getCurrentDensity(r=do.grid.r)
    E0 = j / do.other.fluid.conductivity[-1,:] * np.ones((1, rj.size))
    # do.close()

    # Prescribe this initial electric field
    # ds.eqsys.E_field.setPrescribedData(E0, radius=rj, times=[0])

    # Obtain initial current density profile
    do = runiface(ds, 'init_out.h5', quiet=False)

    ############################################################################
    # Final setup of baseline

    # Copy settings
    ds1 = DREAMSettings(ds)


    ds1.fromOutput('init_out.h5', ignore=['n_i'])   # WHY IGNORE n_i??

    # rest time stepper options
    ds1.timestep.setTmax(150e-3)
    ds1.timestep.setNt(NT)
    ds1.timestep.setNumberOfSaveSteps(200)

    # Enable self consistent evolution of E-field
    ds1.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds1.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)

    # Enable self consistent temperature evolution
    ds1.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
    ds1.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

    # Enable magnetic pertubations that will allow for radial transport
    dBB = 1.5e-3 # Impact of this value will greatly depend on occurence of impurities
    ds1.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB)
    ds1.eqsys.f_re.transport.setMagneticPerturbation(dBB=dBB)
    ds1.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
    ds1.eqsys.f_re.transport.setBoundaryCondition(Transport.BC_F_0)
    #
    # Enable avalanche, hottail and Dreicer generation
    ds1.eqsys.n_re.setAvalanche(RunawayElectrons.AVALANCHE_MODE_FLUID)
    ds1.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_NEURAL_NETWORK)

    ds1.eqsys.f_hot.setInitialProfiles(n0=n, rT0=rT, T0=T0)
    ds1.eqsys.n_re.setHottail(RunawayElectrons.HOTTAIL_MODE_ANALYTIC_ALT_PC)

    # # set relative and absolute tolerances
    ds1.solver.tolerance.set(reltol=2e-6)
    ds1.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
    ds1.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?

    # include info about time spent in different parts...
    ds1.output.setTiming(True, True)
    return ds1

def simulate(ds1):

    ds2 = DREAMSettings(ds1)
    ds2.fromOutput('init_out.h5')
    runiface(ds2, 'out.h5')

def main():
    ds = getBaseline()
    simulate(ds)
    return 0

if __name__ == '__main__':
    sys.exit(main())
