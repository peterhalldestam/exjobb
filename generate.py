#!/usr/bin/env python3

import sys
import numpy as np

import utils

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/pethalld/DREAM/py')
    sys.path.append('/home/peterhalldestam/DREAM/py')
    sys.path.append('/home/hannber/DREAM/py')
    import DREAM

from DREAM import DREAMSettings, DREAMException, runiface
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.ElectricField as EField
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as RE
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

import ITER as Tokamak
# import ASDEXU as Tokamak

# TSTOP = 100
NT = 10000

expDecay = True

def getBaseline(**kwargs):
    """
    Generate baseline scenario. This initialization consists of two short
    simulation runs: the first is used to calculate the conductivity, which is
    then used to set a desired current density profile.
    """

    # Adjust parameters if provided
    nD  = kwargs.get('nD',  Tokamak.ne0)    # Deuterium density
    nT  = kwargs.get('nT',  0.)             # Tritium density
    nAr = kwargs.get('nAr', 0.)             # Impurity (Argon) density
    dBB = kwargs.get('dBB', 1.5e-3)         # Magnetic perturbation
    #...

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

    ##### Background ion density #####
    if nD > 0:
        ds.eqsys.n_i.addIon('D', Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=nD,
                            opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
    if nT > 0:
        ds.eqsys.n_i.addIon('T', Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=nT, tritium=True,
                            opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)

    ##### Impurities #####
    if nAr:
        ds.eqsys.n_i.addIon('Ar', Z=18, Z0=5, iontype=Ions.IONS_DYNAMIC, n=nAr)




    # ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)


    # Solver
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    ds.solver.setType(Solver.NONLINEAR)

    # Set temporary time stepper options during initialization
    ds.timestep.setTmax(1e-11)
    ds.timestep.setNt(1)

    # Include all fluid `OtherQuantity`s
    ds.other.include('fluid')

    # get the exponential-decay temperature evolution
    # tT, rT, T0 = Tokamak.getTemperatureEvolution(tau0=1e-3)

    rT, T0 = Tokamak.getInitialTemperature()
    ############################################################################
    # STEP 1 : Calculate conductivity

    # Prescribe temperature (constant during initialization)
    ds.eqsys.T_cold.setPrescribedData(T0, radius=rT, times=[0])

    # Prescribe dummy electric field
    ds.eqsys.E_field.setPrescribedData(1e-4)

    # Calculate the conductivity
    do = runiface(ds, quiet=True)

    ############################################################################
    # STEP 2 : Obtain initial current density profile

    # Prescribe the exponential-decay temperature
    #ds.eqsys.T_cold.setPrescribedData(T0, radius=rT, times=tT)

    # Obtain the initial electric field from the conductivity calculation
    rj, j = Tokamak.getCurrentDensity(r=do.grid.r)
    j /= Tokamak.j0
    j0 = Tokamak.Ip * 2.0*np.pi / do.grid.integrate(j)
    E0 =  j0 * j / do.other.fluid.conductivity[-1,:] * np.ones((1, rj.size))
    # do.close()

    # Prescribe this initial electric field
    ds.eqsys.E_field.setPrescribedData(E0, radius=rj, times=[0])

    # Obtain initial current density profile
    do = runiface(ds, 'init_out.h5', quiet=True)

    ############################################################################
    # Final setup of baseline

    # Copy settings
    ds1 = DREAMSettings(ds)
    ds1.fromOutput('init_out.h5')

    # rest time stepper options
    Tmax = 150e-3
    ds1.timestep.setTmax(Tmax)
    ds1.timestep.setNt(NT)
    ds1.timestep.setNumberOfSaveSteps(200)

    # Enable self consistent evolution of E-field
    ds1.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
    ds1.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)

    # Enable self consistent temperature evolution
    ds1.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
    ds1.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)


    # Enable magnetic pertubations that will allow for radial transport
    Drr = utils.getRRCoefficient(dBB, R0=Tokamak.R0) # Rechester-Rosenbluth diffusion operator
    pstar = 0.5	# Lower momentum boundry for runaway electrons in Svensson Transport [mc]
    t = np.linspace(0, Tmax, NT)

    ds1.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB)
    ds1.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)

    ds1.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
    ds1.eqsys.n_re.transport.setSvenssonPstar(pstar)
    # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
    ds1.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, interp1d=Transport.INTERP1D_NEAREST)
    ds1.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

    # Enable avalanche, hottail and Dreicer generation
    ds1.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID)
    ds1.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)

    # Enable RE generation due to Tritium decay
    if nT > 0:
        print('tritium')
        ds1.eqsys.n_re.setTritium(True)

    # Background free electron density from ions
    nfree, rn0 = ds.eqsys.n_i.getFreeElectronDensity()
    ds1.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T0)

    # Boundary condition on f at p = pMax (assume f(p>pMax) = 0)
    ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)

    # Do not include the Jacobian elements for d f_hot / d n_i, i.e.
    # derivatives with respect to the ion densities (which would take
    # up *significant* space in the matrix)
    ds.eqsys.f_hot.enableIonJacobian(False) # ANY GOOD?

    # Fluid model for hottail generation, MSc thesis Ida Svenningsson (2020)
    ds1.eqsys.n_re.setHottail(RE.HOTTAIL_MODE_ANALYTIC_ALT_PC)

    # # set relative and absolute tolerances
    ds1.solver.tolerance.set(reltol=2e-6)
    ds1.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
    ds1.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?


    ds1.solver.setMaxIterations(maxiter=500)
    # include info about time spent in different parts...
    ds1.output.setTiming(True, True)

    # NOT WORKING!
    # ds1.timestep.setTerminationFunction(lambda s: terminate(s, TSTOP))
    return ds1

def simulate(ds, name='out.h5'):
    do = None
    try:
        do = runiface(ds, name)
    except DREAMException as err:
        print('ERROR : skipping this simulation!')
        print(err)
    return do

def main():
    n0 = Tokamak.ne0
    ds = getBaseline(nD=.1*n0)
    do = simulate(ds)
    return 0

if __name__ == '__main__':
    sys.exit(main())
