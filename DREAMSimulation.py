#!/usr/bin/env python3

import sys
import numpy as np


import utils
import ITER as Tokamak
from simulation import Simulation

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in utils.DREAMPATHS:
        sys.path.append(dp)
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



# TSTOP = 100
TMAX = 1.5e-1
NT = 10000
NR = 20

TQ_EXPDECAY = 1
TQ_PERTURB  = 2

TQ_TIME_DECAY = 1e-3
TQ_INITIAL_DBB = 1.5e-3


class DREAMSimulation(Simulation):
    """
    Description...
    """

    # baseline input parameters
    baseline = {
        # Fuel densities
        'nH':   0.,
        'nD':   Tokamak.ne0,
        'nT':   0.,
        'nHe':  0.,

        # Impurity densities
        'nBe':  0.,
        'nC':   0.,
        'nFe':  0.,
        'nW':   0.,

        # Initial current density j(r) ~ (1-j1*(r/a)^2)^j2, integral(j) = Ip0
        'j1':   0.,
        'j2':   0.,
        'Ip0':  15e6,

        # Initial temperature profile T(r) = T0*(1-T1*(r/a)^2)
        'T0':   20e3,
        'T1':   .99,

        # Magnetic pertubation (post TQ) dBB(r) ~ 1+dBB1*r^2, integral(dBB) = dBB0
        'dBB0': 0.,
        'dBB1': 0.
    }

    def __init__(self, tq=TQ_PERTURB, quiet=False, out='out.h5', **inputs):
        """
        Set input from baseline or from any user provided input parameters.
        """
        super().__init__(DREAMSimulation.baseline, quiet=quiet, **inputs)

        self.quiet = quiet
        self.out = out
        self.doubleIterations = True

        #### Set the disruption sequences in order ####
        self.ds = DREAMSettings()

        # Set solvers
        self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
        # self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        self.ds.solver.setType(Solver.NONLINEAR)
        self.ds.solver.setMaxIterations(maxiter=500)
        self.ds.solver.tolerance.set(reltol=2e-6)
        self.ds.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
        self.ds.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?

        # Set time stepper settings
        self.ds.timestep.setTmax(TMAX)
        self.ds.timestep.setNt(NT)

        # Disable kinetic grids (we run purely fluid simulations)
        self.ds.hottailgrid.setEnabled(False)
        self.ds.runawaygrid.setEnabled(False)

        # Set the magnetic field from specified Tokamak (see imports)
        Tokamak.setMagneticField(self.ds)

        # Set collision settings
        self.ds.collisions.collfreq_mode        = Collisions.COLLFREQ_MODE_FULL
        self.ds.collisions.collfreq_type        = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        self.ds.collisions.bremsstrahlung_mode  = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
        self.ds.collisions.lnlambda             = Collisions.LNLAMBDA_ENERGY_DEPENDENT

        # Add fuel
        if self.input['nH'] > 0:
            self.ds.eqsys.n_i.addIon('H', n=self.input['nH'], Z=1, Z0=1, hydrogen=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input['nD'] > 0:
            self.ds.eqsys.n_i.addIon('D', n=self.input['nD'], Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input['nT'] > 0:
            self.ds.eqsys.n_i.addIon('T', n=self.input['nT'], Z=1, Z0=1, tritium=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input['nHe'] > 0:
            raise NotImplementedError('Helium is not yet implemented...')

        # Add impurities
        if any([self.input[n] > 0 for n in ('nBe', 'nC', 'nFe', 'nW')]):
            raise NotImplementedError('Impurities is not yet implemented...')

        # Set fluid RE generation
        self.ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
        self.ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID)
        self.ds.eqsys.n_re.setHottail(RE.HOTTAIL_MODE_ANALYTIC_ALT_PC)
        # self.ds.eqsys.n_re.setCompton(RE.COMPTON_MODE_NEGLECT)          # <== LOOK INTO THIS
        if self.input['nT'] > 0:
            self.ds.eqsys.n_re.setTritium(True)

        # Set self-consistent electric field (initial condition is determined by the current density)
        self.ds.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
        self.ds.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)

        # Set initial current density
        rj, j = Tokamak.getInitialCurrentDensity(self.input['j1'], self.input['j2'])
        self.ds.eqsys.j_ohm.setInitialProfile(j, radius=rj, Ip0=self.input['Ip0'])

        # Set initial temperature profile
        rT, T = Tokamak.getInitialTemperature(self.input['T0'], self.input['T1'])
        self.ds.eqsys.T_cold.setInitialProfile(T, radius=rT)

        # Background free electron density from ions
        nfree, rn0 = self.ds.eqsys.n_i.getFreeElectronDensity()
        self.ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T)

        # Boundary condition on f at p = pMax (assume f(p>pMax) = 0)
        self.ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)


        # Thermal quench model
        if tq == TQ_PERTURB:

            # Enable self consistent temperature evolution
            self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
            self.ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

            # Enable magnetic pertubations that will allow for radial transport
            # OBS! The current version only supports a flat pertubation profile that is constant in time
            r_dBB = np.array([0, 0.5*Tokamak.a])    # que?
            dBB = TQ_INITIAL_DBB * np.ones(len(r_dBB))

            # Rechester-Rosenbluth diffusion operator
            Drr, xi_grid, p_grid = utils.getRRCoefficient(dBB, R0=Tokamak.R0)
            Drr = np.tile(Drr, (NT,1,1,1))
            t = np.linspace(0, TMAX, NT)


            self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB[0])
            self.ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)


            self.ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
            self.ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs
            # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
            self.ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r_dBB, p=p_grid, xi=xi_grid, interp1d=Transport.INTERP1D_NEAREST)
            self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)




    def run(self, doubleIterations=None):
        """
        Run simulation.
        """
        do = None
        if doubleIterations is not None:
            self.doubleIterations = doubleIterations
        try:
            do = runiface(self.ds, 'test.h5')
        except DREAMException as err:
            if self.doubleIterations:
                global NT
                if NT >= 1e7:
                    print(err)
                    print('ERROR : Skipping this simulation!')
                else:
                    print(err)
                    print(f'WARNING : Number of iterations is doubled from {NT} to {2*NT}!')
                    NT *= 2
                    simulate(ds, name=name, doubleIterations=True)
            else:
                raise err
        return do





def main():
    s = DREAMSimulation(quiet=False)
    do = s.run(doubleIterations=False)
    utils.visualizeCurrents(do, show=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
