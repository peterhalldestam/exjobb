#!/usr/bin/env python3

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

import utils
import ITER as Tokamak
from simulation import Simulation, Parameter

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in utils.DREAMPATHS:
        sys.path.append(dp)
        # sys.path.append(f'{dp}/build/dreampyface/cxx/')
    import DREAM

from DREAM import DREAMSettings, DREAMOutput, DREAMException, runiface
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
NT = 100
NR = 20

# thermal quench settings
EXP_DECAY = True
TQ_TIME_DECAY = 1e-3
TQ_TMAX = 1e-3
TQ_INITIAL_dBB0 = 1.5e-3

SETTINGS_DIR    = 'settings/'
OUTPUT_DIR      = 'outputs/'

# Input parameter limits
MAX_FUEL_DENSITY = 1e20
MAX_INJECTED_DENSITY = 1e20

class DREAMSimulation(Simulation):
    """
    Description...
    """

    ############## DATA CLASSES ##############

    @dataclass
    class Input(Simulation.Input):
        """
        Input parameters for the DREAM simulation. Defined below is the default
        baseline values of each parameter, as well as domain intervals.
        """
        # Fuel densities
        nH:     Parameter = Parameter(min=0., max=MAX_FUEL_DENSITY,     val=0.)
        nD:     Parameter = Parameter(min=0., max=MAX_FUEL_DENSITY,     val=Tokamak.ne0)
        nT:     Parameter = Parameter(min=0., max=MAX_FUEL_DENSITY,     val=0.)
        nHe:    Parameter = Parameter(min=0., max=MAX_FUEL_DENSITY,     val=0.)

        # Injected ion densities
        nD2:    Parameter = Parameter(min=0., max=MAX_INJECTED_DENSITY, val=0.)
        nNe:    Parameter = Parameter(min=0., max=MAX_INJECTED_DENSITY, val=0.)
        #...

        # Inital current density profile
        j1:     Parameter = Parameter(min=0., max=1.,                   val=1.)
        j2:     Parameter = Parameter(min=0., max=4,                    val=.41)
        Ip0:    Parameter = Parameter(min=0., max=2e7,                  val=Tokamak.Ip)

        # Initial temperature profile
        T1:     Parameter = Parameter(min=0., max=2e6,                  val=Tokamak.T_initial)
        T2:     Parameter = Parameter(min=0., max=1.,                   val=0.)

        # Post TQ magnetic perturbation profile
        dBB1:   Parameter = Parameter(min=0., max=1e-3,                 val=1e-4)
        dBB2:   Parameter = Parameter(min=0., max=1e2,                  val=0.)


    @dataclass
    class Output(Simulation.Output):
        """
        Output variables from the DREAM simulation. The constructor expects one
        DREAMOutput object as argument.
        """
        do:     DREAMOutput

        # Output quantities from DREAM output object
        r:      np.ndarray = field(init=False)  # radial grid
        t:      np.ndarray = field(init=False)  # simulation time
        I_re:   np.ndarray = field(init=False)  # RE current
        I_ohm:  np.ndarray = field(init=False)  # Ohmic current
        I_tot:  np.ndarray = field(init=False)  # total current
        T_cold: np.ndarray = field(init=False)  # cold electron temperature

        # Derived timing quantities
        tCQ:    float = field(init=False)   # current quench time, tCQ(t20, t80)
        t20:    float = field(init=False)   # I_ohm(t20) / I_ohm(0) = 20%
        t80:    float = field(init=False)   # I_ohm(t80) / I_ohm(0) = 80%

        def __post_init__(self):
            """
            Checks that all list sizes are equal and sets the current quench
            time, t20 and t80 (CQ reference points).
            """
            self.r      = self.do.grid.r 
            self.t      = self.do.grid.t
            self.I_re   = self.do.eqsys.j_re.current()
            self.I_ohm  = self.do.eqsys.j_ohm.current()
            self.I_tot  = self.do.eqsys.j_tot.current()
            self.T_cold = self.do.eqsys.T_cold.data
            self.t20, self.t80, self.tCQ = utils.getCQTime(self.t, self.I_ohm)

            assert len(self.t) == NT
            assert len(self.r) == NR
            assert all(I.shape == self.t.shape for I in [self.I_re, self.I_ohm])
            assert self.T_cold.shape == (NT, NR)


        def getMaxRECurrent(self):
            return self.I_re.max()

        def visualizeCurrents(self, ax=None, show=False):
            utils.visualizeCurrents(self.t, self.I_ohm, self.I_re, self.I_tot, ax=ax, show=show)
            return ax

        def visualizeTemperature(self, times=[0,-1], ax=None, show=False):
            # utils.visualizeTemperature(sel.)
            self.do.eqsys.T_cold.plot(ax=ax, show=show, t=times, log=False)
            plt.show()
            return ax

    ############## DISRUPTION SIMULATION SETUP ##############

    def __init__(self, id='baseline', verbose=True, **inputs):
        """
        Set input from baseline or from any user provided input parameters.
        """
        super().__init__(id=id, verbose=verbose, **inputs)

        self.outputFile         = f'{OUTPUT_DIR}{id}output.h5'
        self.settingsFile       = f'{SETTINGS_DIR}{id}settings.h5'
        self.doubleIterations   = False

        self.ds1 = DREAMSettings()
        self.ds2 = None
        self.do1 = None
        self.do2 = None

        # Set timestep to be fixed throughout the entire simulation
        self.ds1.timestep.setDt(TMAX / NT)

        # Set solvers
        self.ds1.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
        # self.ds1.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        self.ds1.solver.setType(Solver.NONLINEAR)
        self.ds1.solver.setMaxIterations(maxiter=500)
        self.ds1.solver.tolerance.set(reltol=2e-6)
        self.ds1.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
        self.ds1.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?

        # Disable kinetic grids (we run purely fluid simulations)
        self.ds1.hottailgrid.setEnabled(False)
        self.ds1.runawaygrid.setEnabled(False)
        # Set the magnetic field from specified Tokamak (see imports)print
        Tokamak.setMagneticField(self.ds1, nr=NR)

        # Set collision settings
        self.ds1.collisions.collfreq_mode        = Collisions.COLLFREQ_MODE_FULL
        self.ds1.collisions.collfreq_type        = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        self.ds1.collisions.bremsstrahlung_mode  = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
        self.ds1.collisions.lnlambda             = Collisions.LNLAMBDA_ENERGY_DEPENDENT

        # Add fuel
        if self.input.nH.val > 0:
            self.ds1.eqsys.n_i.addIon('H', n=self.input.nH.val, Z=1, Z0=1, hydrogen=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nD.val > 0:
            self.ds1.eqsys.n_i.addIon('D', n=self.input.nD.val, Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nT.val > 0:
            self.ds1.eqsys.n_i.addIon('T', n=self.input.nT.val, Z=1, Z0=1, tritium=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nHe.val > 0:
            raise NotImplementedError('Helium is not yet implemented...')

        # Add injected ions
        if self.input.nNe.val > 0:
            raise NotImplementedError('injected ions are not yet implemented...')
        if self.input.nD2.val > 0:
            raise NotImplementedError('injected ions are not yet implemented...')

        # Set fluid RE generation
        self.ds1.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
        self.ds1.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID)
        self.ds1.eqsys.n_re.setHottail(RE.HOTTAIL_MODE_ANALYTIC_ALT_PC)
        # self.ds1.eqsys.n_re.setCompton(RE.COMPTON_MODE_NEGLECT)          # <== LOOK INTO THIS
        if self.input.nT.val > 0:
            self.ds1.eqsys.n_re.setTritium(True)

        # Set self-consistent electric field (initial condition is determined by the current density)
        self.ds1.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
        self.ds1.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)

        # Set initial temperature profile
        rT, T = Tokamak.getInitialTemperature(self.input.T1.val, self.input.T2.val)
        self.ds1.eqsys.T_cold.setInitialProfile(T, radius=rT)

        # Set initial current density
        rj, j = Tokamak.getInitialCurrentDensity(self.input.j1.val, self.input.j2.val)
        self.ds1.eqsys.j_ohm.setInitialProfile(j, radius=rj, Ip0=self.input.Ip0.val)

        # Background free electron density from ions
        nfree, rn0 = self.ds1.eqsys.n_i.getFreeElectronDensity()
        self.ds1.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T)

        # Boundary condition on f at p = pMax (assume f(p>pMax) = 0)
        self.ds1.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)


    def run(self, doubleIterations=True):
        """
        Runs the simulation and produce a single output.
        """
        assert self.output is None, 'Output object already exists!'

        if EXP_DECAY:
            self._runExpDecayTQ()
        else:
            self._runPerturbTQ()

        assert isinstance(self.do1, DREAMOutput)
        assert isinstance(self.do2, DREAMOutput)

        # Join the two outputs
        t     = np.append(self.do1.grid.t, self.do1.grid.t[-1] + self.do2.grid.t[1:])
        I_re  = np.append(self.do1.eqsys.j_re.current(), self.do2.eqsys.j_re.current()[1:])
        I_ohm = np.append(self.do1.eqsys.j_ohm.current(), self.do2.eqsys.j_ohm.current()[1:])
        I_tot = np.append(self.do1.eqsys.j_tot.current(), self.do2.eqsys.j_tot.current()[1:])

        self.output = self.Output(self.do2)
        return 0

    def _runExpDecayTQ(self):
        """
        Run an exponential decay thermal quench (prescribed temperature evolution)the
        """
        # Set prescribed temperature evolution
        self.ds1.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)

        # Set exponential-decay temperature
        t, r, T = Tokamak.getTemperatureEvolution(self.input.T1.val, self.input.T2.val, tmax=TMAX, nt=NT)
        self.ds1.eqsys.T_cold.setPrescribedData(T, radius=r, times=t)

        # Set TQ time stepper settings
        self.ds1.timestep.setTmax(TQ_TMAX)

        # run TQ part of simulation
        out = self._getFileName('TQ_output', OUTPUT_DIR)
        self.ds1.output.setFilename(out)
        self.ds1.save(self._getFileName('TQ_settings', SETTINGS_DIR))
        self.do1 = self._run(self.ds1, out)

        ##### Post-TQ (self consistent temperature evolution) #####

        self.ds2 = DREAMSettings(self.ds1)
        self.ds2.fromOutput(out)

        # Change to self consistent temperature and set external magnetic pertubation
        r, dBB = utils.getQuadraticMagneticPerturbation(self.ds2, self.input.dBB1.val, self.input.dBB2.val)
        self._setSvenssonTransport(self.ds2, dBB, r)

        # Set post-TQ time stepper settings
        self.ds2.timestep.setTmax(TMAX - TQ_TMAX)

        # run post-TQ part of simulation
        out = self._getFileName('pTQ_output', OUTPUT_DIR)
        self.ds2.output.setFilename(out)
        self.ds2.save(self._getFileName('pTQ_setting', SETTINGS_DIR))
        self.do2 = self._run(self.ds2)


    def _runPerturbTQ(self):
        raise NotImplementedError('TQ_PERTURB is not yet implemented...')

        # Set edge vanishing TQ magnetic pertubation
        r, dBB = utils.getQuadraticMagneticPerturbation(self.ds1, TQ_INITIAL_dBB0, -1/Tokamak.a**2)
        self._setSvenssonTransport(self.ds1, dBB, r)

        # ds1.timestep.setTerminationFunction(lambda s: terminate(s, TSTOP))
        # self.run(dreampyface=True)

        self.ds = DREAMSettings(self.ds1)
        #...


    def _setSvenssonTransport(self, ds, dBB, r):
        """
        Configures the Svensson transport settings.
        """
        assert dBB.shape == r.shape

        # Enable self consistent temperature evolution
        ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
        ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

        # Enable magnetic pertubations that will allow for radial transport
        t = np.linspace(0, TMAX, NT)

        ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
        ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=np.tile(dBB, (NT, 1)), r=r, t=t)

        # Rechester-Rosenbluth diffusion operator
        Drr, xi, p = utils.getDiffusionOperator(dBB, R0=Tokamak.R0)
        Drr = np.tile(Drr, (NT,1,1,1))

        ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
        ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs

        # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
        print(Drr.shape)
        ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r, p=p, xi=xi, interp1d=Transport.INTERP1D_NEAREST)
        ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)



    def _run(self, ds, out='out.h5', doubleIterations=None, dreampyface=False, ntmax=1e7):
        """
        Run simulation and rerun simulation with a doubled number of timesteps
        if it crashes.
        """
        do = None
        if doubleIterations is not None:
            self.doubleIterations = doubleIterations
        try:
            if dreampyface:
                s = dreampyface.setup_simulation(ds)
                do = s.run()
            else:
                do = runiface(ds, out, quiet=not self.verbose)
        except DREAMException as err:
            if self.doubleIterations:
                dt = ds.timestep.dt
                if dt >= ntmax / NT:
                    print(err)
                    print('ERROR : Skipping this simulation!')
                else:
                    print(err)
                    print(f'WARNING : Timestep is reduced from {dt} to {.5*dt}!')
                    ds.timestep.setDt(2*dt)
                    self._run(ds, doubleIterations=True, dreampyface=dreampyface)
            else:
                raise err
        return do

    def _getFileName(self, io, dir):
        """
        Returns appropriate name of file and makes sure its directory exists.
        """
        filename = f'{dir}{self.id}_{io}.h5'

        # Create directory if needed
        fp = pathlib.Path(filename)
        dir = fp.parent.resolve()
        if not dir.exists():
            dir.mkdir(parents=True)

        return str(fp.absolute())





def main():

    # Set to exponential decay in TQ if dreampyface doesn't exist
    global EXP_DECAY
    if not EXP_DECAY:
        try:
            import dreampyface
        except ModuleNotFoundError as err:
            EXP_DECAY = True
            print('ERROR: Python module dreampyface not found. Switchin to exp-decay...')

    s = DREAMSimulation()
    s.run(doubleIterations=False)

    # s.output.visualizeCurrents(show=True)
    s.output.visualizeTemperature(show=True)

    return 0

if __name__ == '__main__':
    sys.exit(main())
