#!/usr/bin/env python3

import sys
import pathlib
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
        # sys.path.append(f'{dp}/build/dreampyface/cxx/')
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

import dreampyface

# TSTOP = 100
TMAX = 1.5e-1
NT = 100
NR = 20

TQ_EXPDECAY = 1
TQ_PERTURB  = 2

TQ_TIME_DECAY = 1e-3
TQ_TMAX = 3 * TQ_TIME_DECAY
TQ_INITIAL_dBB0 = 1.5e-3

SETTINGS_DIR    = 'settings/'
OUTPUT_DIR      = 'outputs/'

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
        'T0':   2e4,
        'T1':   .99,

        # Magnetic pertubation (post TQ) dBB(r) ~ 1+dBB1*r^2, integral(dBB) = dBB0
        'dBB0': 0.,
        'dBB1': 0.
    }

    def __init__(self, tq=TQ_PERTURB, id='', quiet=False, **inputs):
        """
        Set input from baseline or from any user provided input parameters.
        """
        super().__init__(DREAMSimulation.baseline, id=id, quiet=quiet, **inputs)

        self.outputFile         = f'{OUTPUT_DIR}{id}output.h5'
        self.settingsFile       = f'{SETTINGS_DIR}{id}settings.h5'
        self.doubleIterations   = True

        #### Set the disruption sequences in order ####
        self.ds = DREAMSettings()

        # Set timestep to be fixed throughout the entire simulation
        self.ds.timestep.setDt(TMAX / NT)

        # Set solvers
        self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
        # self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        self.ds.solver.setType(Solver.NONLINEAR)
        self.ds.solver.setMaxIterations(maxiter=500)
        self.ds.solver.tolerance.set(reltol=2e-6)
        self.ds.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
        self.ds.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?

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


    def run(self, tq=None, doubleIterations=True):
        """
        Run simulation
        """
        assert self.output is None, \
            f'Output object already exists!'

        if doubleIterations is not None:
            self.doubleIterations = doubleIterations

        # Thermal quench model
        if self.tq == TQ_PERTURB:    # (Not working due to problems with DREAM branch origin/theater)

            # Set edge vanishing TQ magnetic pertubation
            r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, TQ_INITIAL_dBB0, -1/Tokamak.a**2)
            _setSvenssonTransport(dBB, r)

            # ds1.timestep.setTerminationFunction(lambda s: terminate(s, TSTOP))
            # self.run(dreampyface=True)

            ds = DREAMSettings(self.ds)
            #...

        elif self.tq == TQ_EXPDECAY:

            # Set prescribed temperature evolution
            self.ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)

            # Set exponential-decay temperature
            t, r, T = Tokamak.getTemperatureEvolution(self.input['T0'], self.input['T1'], tmax=TMAX, nt=NT)
            ds.eqsys.T_cold.setPrescribedData(T, radius=r, times=t)

            # Set time stepper settings
            self.ds.timestep.setTmax(TQ_TMAX)

            do = self._run()

            ds = DREAMSettings(self.ds)



    def _setSvenssonTransport(self, dBB, radius):
        """
        Configures the Svensson transport settings.
        """
        assert dBB.shape == radius.shape

        # Enable self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
        self.ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

        # Enable magnetic pertubations that will allow for radial transport
        self.ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)

        self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB, r=r)

        # Rechester-Rosenbluth diffusion operator
        Drr, xi, p = utils.getDiffusionOperator(dBB, R0=Tokamak.R0)
        Drr = np.tile(Drr, (NT,1,1,1))
        t = np.linspace(0, TMAX, NT)

        self.ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
        self.ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs
        # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
        self.ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r_dBB, p=p, xi=xi, interp1d=Transport.INTERP1D_NEAREST)
        self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)



    def _run(self, doubleIterations=None, dreampyface=False, ntmax=1e7):
        """
        Run simulation and rerun simulation with a doubled number of timesteps
        if it crashes.
        """
        do = None
        if doubleIterations is not None:
            self.doubleIterations = doubleIterations
        try:
            if dreampyface:
                s = dreampyface.setup_simulation(self.ds)
                do = s.run()
            else:
                do = runiface(self.ds)
        except DREAMException as err:
            if self.doubleIterations:
                nt = self.ds.timestep.nt
                if nt >= ntmax:
                    print(err)
                    print('ERROR : Skipping this simulation!')
                else:
                    print(err)
                    print(f'WARNING : Number of iterations is doubled from {nt} to {2*nt}!')
                    self.ds.timestep.setNt(2*nt)
                    run(doubleIterations=True, dreampyface=dreampyface)
            else:
                raise err
        return do

    def _getFileName(self, str: io, str: dir):
        """
        Returns appropriate name of file and makes sure its directory exists.
        """
        filename = f'{dir}{self.id}_{io}.h5'

        # Create directory if needed
        p = pathlib.Path(filename).parent.resolve()
        if not p.exists():
            p.mkdir(parents=True)

        return p.resolve()

def main():
    s = DREAMSimulation(quiet=False)
    do = s.run(doubleIterations=False)
    utils.visualizeCurrents(do, show=True)
    print(utils.getCQTime(do))
    return 0

if __name__ == '__main__':
    sys.exit(main())
