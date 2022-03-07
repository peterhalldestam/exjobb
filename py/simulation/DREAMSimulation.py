#!/usr/bin/env python3

import sys, os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

sys.path.append(os.path.abspath('..'))
import utils
import tokamaks.ITER as Tokamak
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


# Number of radial nodes
NR = 5

# Number of time iterations in each step
NT_IONIZ    = 1000
NT_TQ       = 2000
NT_CQ       = 4000

# Amount of time (s) in each step
TMAX_TOT    = 1.5e-1
TMAX_IONIZ  = 1e-6
TMAX_TQ     = Tokamak.t0 * 4



# thermal quench settings
EXP_DECAY = True
TQ_TIME_DECAY = Tokamak.t0
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

    @dataclass(init=False)
    class Input(Simulation.Input):
        """
        Input parameters for the DREAM simulation. Defined below is the default
        baseline values of each parameter, as well as domain intervals.
        """

        def __init__(self):
            """
            Constructor.

            NAME:                       MIN:    MAX:              DEFAULT VALUE:
            """
            # Fuel densities
            self.nH         = Parameter(min=0., max=MAX_FUEL_DENSITY,       val=0.)
            self.nD         = Parameter(min=0., max=MAX_FUEL_DENSITY,       val=Tokamak.ne0)
            self.nT         = Parameter(min=0., max=MAX_FUEL_DENSITY,       val=0.)
            self.nHe        = Parameter(min=0., max=MAX_FUEL_DENSITY,       val=0.)

            # Injected ion densities
            self.nD2        = Parameter(min=0., max=MAX_INJECTED_DENSITY,   val=7*Tokamak.ne0)
            self.alphaD2    = Parameter(min=-10,max=10,                     val=0.)
            self.nNe        = Parameter(min=0., max=MAX_INJECTED_DENSITY,   val=.08*Tokamak.ne0)
            self.alphaNe    = Parameter(min=-10,max=10,                     val=0.)
            #...

            # Inital current density profile
            self.j1         = Parameter(min=0., max=1.,                     val=1.)
            self.j2         = Parameter(min=0., max=4,                      val=.41)
            self.Ip0        = Parameter(min=0., max=2e7,                    val=Tokamak.Ip)

            # Initial temperature profile
            self.T1         = Parameter(min=0., max=2e6,                    val=Tokamak.T_initial)
            self.T2         = Parameter(min=0., max=1.,                     val=.99)

            # Post TQ magnetic perturbation profile
            self.dBB1       = Parameter(min=0., max=1e-3,                   val=1e-4)
            self.dBB2       = Parameter(min=0., max=1e2,                    val=0.)


    @dataclass(init=False)
    class Output(Simulation.Output):
        """
        Output variables from the DREAM simulation. The constructor expects one
        or more DREAMOutput objects as arguments.
        """
        dos: list[DREAMOutput]

        # Output quantities from DREAM output object
        r:      np.ndarray  # radial grid
        t:      np.ndarray  # simulation time
        I_re:   np.ndarray  # RE current
        I_ohm:  np.ndarray  # Ohmic current
        I_tot:  np.ndarray  # total current
        T_cold: np.ndarray  # cold electron temperature

        # Derived timing quantities
        tCQ: float  # current quench time, tCQ(t20, t80)
        t20: float  # I_ohm(t20) / I_ohm(0) = 20%
        t80: float  # I_ohm(t80) / I_ohm(0) = 80%

        def __init__(self, *dos):
            """
            Constructor. Joins data from the provided DREAM output objects.
            """
            self.t      = utils.join('grid.t', dos, time=True)
            self.r      = utils.join('grid.r', dos, radius=True)
            self.I_re   = utils.join('eqsys.j_re.current()', dos)
            self.I_ohm  = utils.join('eqsys.j_ohm.current()', dos)
            self.I_tot  = utils.join('eqsys.j_tot.current()', dos)
            self.T_cold = utils.join('eqsys.T_cold.data', dos)
            self.t20, self.t80, self.tCQ = utils.getCQTime(self.t, self.I_ohm)

            # # (for debug, doesn't work if DREAM crashes just once in _run())
            # nt = 1 + NT_IONIZ + NT_TQ + NT_CQ
            # assert len(self.t) == nt, print(nt, self.t.shape)
            # assert len(self.r) == NR, print(NR, self.r.shape, self.r)
            # assert all(I.shape == self.t.shape for I in [self.I_re, self.I_ohm, self.I_tot])
            # assert self.T_cold.shape == (nt, NR), print(self.T_cold.shape, (nt, NR))

        def getMaxRECurrent(self):
            return self.I_re.max()

        def visualizeCurrents(self, log=False, ax=None, show=False):
            return utils.visualizeCurrents(self.t, self.I_ohm, self.I_re, self.I_tot, log=log, ax=ax, show=show)

        def visualizeTemperature(self, times=[0,-1], ax=None, show=False):
            return utils.visualizeTemperature(self.r, self.T_cold, times=times, ax=ax, show=show)

        def visualizeTemperatureEvolution(self, radii=[0], ax=None, show=False):
            return utils.visualizeTemperatureEvolution(self.t, self.T_cold, radii=radii, ax=ax, show=show)

    ############## DISRUPTION SIMULATION SETUP ##############

    def __init__(self, id='baseline', verbose=True, **inputs):
        """
        Set input from baseline or from any user provided input parameters.
        """
        super().__init__(id=id, verbose=verbose, **inputs)

        self.ds = None      # This will be updated for each subsequent simulation.
        self.do = None      # We need access to do.grid.integrate()

        self.handleCrash = True
        self.maxReruns   = 3

        ##### Generate the initialization simulation #####

        self.ds = DREAMSettings()

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
        Tokamak.setMagneticField(self.ds, nr=NR)

        # Set collision settings
        self.ds.collisions.collfreq_mode        = Collisions.COLLFREQ_MODE_FULL
        self.ds.collisions.collfreq_type        = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        self.ds.collisions.bremsstrahlung_mode  = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
        self.ds.collisions.lnlambda             = Collisions.LNLAMBDA_ENERGY_DEPENDENT

        # Add fuel
        if self.input.nH.val > 0:
            self.ds.eqsys.n_i.addIon('H', n=self.input.nH.val, Z=1, Z0=1, hydrogen=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nD.val > 0:
            self.ds.eqsys.n_i.addIon('D', n=self.input.nD.val, Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nT.val > 0:
            self.ds.eqsys.n_i.addIon('T', n=self.input.nT.val, Z=1, Z0=1, tritium=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nHe.val > 0:
            raise NotImplementedError('Helium is not yet implemented...')

        # Set fluid RE generation
        self.ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
        self.ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID)
        self.ds.eqsys.n_re.setHottail(RE.HOTTAIL_MODE_ANALYTIC_ALT_PC)
        # ds.eqsys.n_re.setCompton(RE.COMPTON_MODE_NEGLECT)          # <== LOOK INTO THIS
        if self.input.nT.val > 0:
            self.ds.eqsys.n_re.setTritium(True)

        # Set prescribed electric field
        self.ds.eqsys.E_field.setPrescribedData(1e-4)

        # Set initial temperature profile
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)
        rT, T = Tokamak.getInitialTemperature(self.input.T1.val, self.input.T2.val)
        self.ds.eqsys.T_cold.setPrescribedData(T, radius=rT)

        # Background free electron density from ions
        nfree, rn0 = self.ds.eqsys.n_i.getFreeElectronDensity()
        self.ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T)

        # Boundary condition on f at p = pMax (assume f(p>pMax) = 0)
        self.ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)

        # We need to access methods from within a DREAM output object
        self.ds.timestep.setTmax(1e-11)
        self.ds.timestep.setNt(1)
        self.do = self._run(verbose=False)

        # Set self-consistent electric field (initial condition is determined by the current density)
        self.ds.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
        self.ds.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)

        # Set initial current density
        rj, j = Tokamak.getInitialCurrentDensity(self.input.j1.val, self.input.j2.val, NR)
        self.ds.eqsys.j_ohm.setInitialProfile(j, radius=rj, Ip0=self.input.Ip0.val)



    def run(self, handleCrash=None):
        """
        Runs the simulation and return a single simulation Output object from
        several DREAM output objects, which are joined during during initialization.

        :param handleCrash:     If True, any crashed simulation are rerun in
                                higher resolution in time, until some max number
                                of iterations is reached.
        """
        assert self.output is None, 'Output object already exists!'

        if handleCrash is not None:
            self.handleCrash = handleCrash

        if EXP_DECAY:
            do1, do2, do3 = self._runExpDecayTQ()
        else:
            self._runPerturbTQ()

        self.output = self.Output(do1, do2, do3)
        return 0

    def _runInjectionIonization(self):


        # Add injected ions
        if self.input.nD2.val > 0:
            r, nD2 = utils.getDensityProfile(self.do, self.input.nD2.val, self.input.alphaD2.val)
            self.ds.eqsys.n_i.addIon('D2', Z=1, iontype=Ions.IONS_DYNAMIC, Z0=0, n=nD2, r=r,
            opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)

        if self.input.nNe.val > 0:
            r, nNe = utils.getDensityProfile(self.do, self.input.nNe.val, self.input.alphaNe.val)
            self.ds.eqsys.n_i.addIon('Ne', Z=10, iontype=Ions.IONS_DYNAMIC, Z0=0, n=nNe, r=r)


        self.ds.solver.setTolerance(reltol=1e-2)
        self.ds.solver.setMaxIterations(maxiter=500)
        self.ds.timestep.setTmax(TMAX_IONIZ)
        self.ds.timestep.setNt(NT_IONIZ)

        out = self._getFileName('ioniz_output', OUTPUT_DIR)
        self.ds.output.setFilename(out)
        do = self._run(out=out)

        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out)
        return do



    def _runExpDecayTQ(self):
        """
        Run an exponential decay thermal quench (prescribed temperature evolution)the
        """
        # Set prescribed temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)

        # Set exponential-decay temperature
        t, r, T = Tokamak.getTemperatureEvolution(self.input.T1.val, self.input.T2.val, tmax=TMAX_TQ)#, nt=NT_TQ)
        self.ds.eqsys.T_cold.setPrescribedData(T, radius=r, times=t)

        do1 = self._runInjectionIonization()

        # Set TQ time stepper settings
        self.ds.timestep.setNt(NT_TQ)
        self.ds.timestep.setTmax(TMAX_TQ - TMAX_IONIZ)

        # run TQ part of simulation
        out = self._getFileName('TQ_output', OUTPUT_DIR)
        self.ds.output.setFilename(out)
        do2 = self._run(out=out)
        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out)

        ##### Post-TQ (self consistent temperature evolution) #####

        # Set CQ/post-TQ time stepper settings
        self.ds.timestep.setNt(NT_CQ)
        self.ds.timestep.setTmax(TMAX_TOT - TMAX_TQ - TMAX_IONIZ)

        # # Change to self consistent temperature and set external magnetic pertubation
        r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, self.input.dBB1.val, self.input.dBB2.val)
        self._setSvenssonTransport(dBB, r)

        # run final part of simulation
        out = self._getFileName('CQ_output', OUTPUT_DIR)
        self.ds.output.setFilename(out)
        do3 = self._run(out=out)

        return do1, do2, do3

    def _runPerturbTQ(self):
        raise NotImplementedError('TQ_PERTURB is not yet implemented...')

        # Set edge vanishing TQ magnetic pertubation
        r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, TQ_INITIAL_dBB0, -1/Tokamak.a**2)
        self._setSvenssonTransport(dBB, r)

        # ds1.timestep.setTerminationFunction(lambda s: terminate(s, TSTOP))
        # self.run(dreampyface=True)

        # self.ds = DREAMSettings(self.ds1)
        #...


    def _setSvenssonTransport(self, dBB, r):
        """
        Configures the Svensson transport settings.
        """
        assert dBB.shape == r.shape

        # Enable self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
        self.ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)

        tmax = self.ds.timestep.tmax
        nt = self.ds.timestep.nt

        # Enable magnetic pertubations that will allow for radial transport
        t = np.linspace(0, tmax, nt)

        self.ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
        self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=np.tile(dBB, (nt, 1)), r=r, t=t)

        # Rechester-Rosenbluth diffusion operator
        Drr, xi, p = utils.getDiffusionOperator(dBB, R0=Tokamak.R0)
        Drr = np.tile(Drr, (nt,1,1,1))

        self.ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
        self.ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs

        # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
        self.ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r, p=p, xi=xi, interp1d=Transport.INTERP1D_NEAREST)
        self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)



    def _run(self, out=None, verbose=None, ntmax=None):
        """
        Run simulation and handle crashes.
        """
        do = None
        if verbose is None:
            quiet = (not self.verbose)
        else:
            quiet = (not verbose)

        global EXP_DECAY
        if EXP_DECAY:
            try:
                do = runiface(self.ds, out, quiet=quiet)
            except DREAMException as err:
                if self.handleCrash:
                    tmax = self.ds.timestep.tmax
                    nt = self.ds.timestep.nt
                    if ntmax is None:
                        ntmax = 2**maxReruns * max(NT_IONIZ, NT_TQ, NT_CQ)
                    if nt >= ntmax:
                        print(err)
                        print('ERROR : Skipping this simulation!')
                    else:
                        print(err)
                        print(f'WARNING : Number of iterations is increased from {nt} to {2*nt}!')
                        self.ds.timestep.setNt(2*nt)
                        do = self._run(out=out, verbose=verbose, ntmax=ntmax)
                else:
                    raise err
        else:
            try:
                import dreampyface
                s = dreampyface.setup_simulation(self.ds)
                do = s.run()

            # Set to exponential decay in TQ if dreampyface doesn't exist
            except ModuleNotFoundError as err:
                EXP_DECAY = True
                print('ERROR: Python module dreampyface not found. Switchin to exp-decay...')
                do = self._run(out=out, verbose=verbose, ntmax=ntmax)

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
    s.run(handleCrash=False)

    # analyze output data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    s.output.visualizeCurrents(ax=ax1)
    s.output.visualizeTemperature(ax=ax2)
    s.output.visualizeTemperatureEvolution(ax=ax3)
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
