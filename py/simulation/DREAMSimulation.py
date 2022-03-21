#!/usr/bin/env python3

import sys, os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

sys.path.append(os.path.abspath('..'))
import utils
import tokamaks.ITER as Tokamak
from simulation import Simulation
from simulationException import SimulationException

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    for dp in utils.DREAMPATHS:
        sys.path.append(dp)
    import DREAM

from DREAM import DREAMSettings, DREAMOutput, DREAMException, runiface
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.OhmicCurrent as JOhm
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.ElectricField as EField
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as RE
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

CHECK_OUTPUT = True     # Check if n_re / n_cold > 1e-2 post simulation
REMOVE_FILES = False     # Removes output files post simulation
MAX_RERUNS = 4          # Maximum number of reruns before raising SimulationException

# Number of radial nodes
NR = 5

# Maximum simulation time
TMAX_TOT    =   1.5e-1

# Enable transport of cold electrons and REs
TRANSPORT_COLD  = True
TRANSPORT_RE    = True

# Thermal quench modes
TQ_MODE_EXPDECAY    = 1
TQ_MODE_TRANSPORT   = 2

# (TQ) Exponential decay settings
TMAX_IONIZ  = 1e-6
TMAX_TQ     = Tokamak.t0 * 8
NT_IONIZ    = 1000
NT_TQ       = 2000
NT_CQ       = 6000

# (TQ) Transport settings
TQ_DECAY_TIME = Tokamak.t0
TQ_STOP_FRACTION = 1 / 2000  # 20 kev -> 10 eV
TQ_FINAL_TEMPERATURE = Tokamak.T_final
TQ_INITIAL_dBB0 = 3.5e-3

SETTINGS_DIR    = 'settings/'
OUTPUT_DIR      = 'outputs/'

# Input parameter limits
MAX_FUEL_DENSITY = 1e20
MAX_INJECTED_DENSITY = 1e20

# Custom exceptions
class TransportException(SimulationException):
    pass

class MaximumIterationsException(SimulationException):
    pass

class DREAMCrashedException(SimulationException):
    pass


class DREAMSimulation(Simulation):
    """
    Description...
    """

    ############## DATA CLASSES ##############

    @dataclass
    class Input(Simulation.Input):
        """
        Input parameters for the DREAM simulation. Defined below is the default
        baseline values of each parameter.
        """
        # Fuel densities
        nH:     float = 0.
        nD:     float = .5 * Tokamak.ne0
        nT:     float = .5 * Tokamak.ne0
        nHe:    float = 0.

        # Massive gas injection density profiles
        nD2:    float = 7 * Tokamak.ne0
        nNe:    float = .08 * Tokamak.ne0
        aD2:    float = 0.
        aNe:    float = 0.
        #...

        # Initial current density profile
        j1:     float = 1.
        j2:     float = .41
        Ip0:    float = Tokamak.Ip

        # Initial temperature profile
        T1:     float = Tokamak.T_initial
        T2:     float = .99

        # Post TQ magnetic perturbation profile
        dBB1:   float = 4e-4
        dBB2:   float = 0.

    @dataclass(init=False)
    class Output(Simulation.Output):
        """
        Output variables from the DREAM simulation. The constructor expects one
        or more DREAMOutput objects as arguments.
        """
        dos: DREAMOutput

        # Output quantities from DREAM output object
        r:      np.ndarray  # radial grid
        t:      np.ndarray  # simulation time
        I_re:   np.ndarray  # RE current
        I_ohm:  np.ndarray  # Ohmic current
        I_tot:  np.ndarray  # total current
        T_cold: np.ndarray  # cold electron temperature

        def __init__(self, *dos, close=True):
            """
            Constructor. Joins data from the provided DREAM output objects.
            """
            self.t      = utils.join('grid.t', dos, time=True)
            self.r      = utils.join('grid.r', dos, radius=True)
            self.I_re   = utils.join('eqsys.j_re.current()', dos)
            self.I_ohm  = utils.join('eqsys.j_ohm.current()', dos)
            self.I_tot  = utils.join('eqsys.j_tot.current()', dos)
            self.T_cold = utils.join('eqsys.T_cold.data', dos)

            if close:
                for do in dos:
                    do.close()

            if REMOVE_FILES:
                paths = [OUTPUT_DIR + path for path in os.listdir(OUTPUT_DIR)]
                for fp in paths:
                    os.remove(fp)

            assert len(self.r) == NR
            assert all(I.shape == self.t.shape for I in [self.I_re, self.I_ohm, self.I_tot])

        def _getTime(self, arr, x):
            """
            Returns the first time an element in arr is a fraction x of its
            maximum value.
            """
            assert len(self.t) == len(arr)
            assert 0 < x < 1
            val0 = arr.max()
            for t, val in zip(self.t, arr):
                if val <= x * val0:
                    return t

        def getCQTime(self):
            """
            Tries to calculate the current quench time and returns it. If unable
            it will return infinity.
            """
            t80 = self._getTime(self.I_ohm, .8)
            t20 = self._getTime(self.I_ohm, .2)
            if t80 is not None and t20 is not None:
                return (t20 - t80) / .6
            else:
                return np.inf

        def getMaxRECurrent(self):
            """
            Returns the maximum runaway electron current.
            """
            return self.I_re.max()

        def visualizeCurrents(self, log=False, ax=None, show=False):
            """
            Plot the Ohmic, RE and total currents.
            """
            return utils.visualizeCurrents(self.t, self.I_ohm, self.I_re, self.I_tot, log=log, ax=ax, show=show)

        def visualizeTemperature(self, times=[0,-1], ax=None, show=False):
            """
            Plot the temperature profile at selected timesteps.
            """
            return utils.visualizeTemperature(self.r, self.T_cold, times=times, ax=ax, show=show)

        def visualizeTemperatureEvolution(self, radii=[0], ax=None, show=False):
            """
            Plot the temperature evolution at selected radii.
            """
            return utils.visualizeTemperatureEvolution(self.t, self.T_cold, radii=radii, ax=ax, show=show)

    ############## DISRUPTION SIMULATION SETUP ##############

    def __init__(self, mode=TQ_MODE_EXPDECAY, id='out', verbose=True, **inputs):
        """
        Constructor. Core simulation settings. No input parameters are used here.
        """
        super().__init__(id=id, verbose=verbose, **inputs)

        self.mode = mode

        self.ds = None      # This will be updated for each subsequent simulation.
        self.do = None      # We need access to do.grid.integrate()

        self.handleCrash = True

        self.tStop = None   # Used for debuggin in TQ_MODE_TRANSPORT (perhaps we don't need this...)

        ##### Generate the initialization simulation #####
        self.ds = DREAMSettings()
        self.ds.other.include('fluid')

        # Set solver settings
        self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
        self.ds.solver.setType(Solver.NONLINEAR)
        self.ds.solver.setMaxIterations(maxiter=500)
        self.ds.solver.tolerance.set(reltol=2e-6)
        self.ds.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
        self.ds.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?

        # Disable kinetic grids (we run purely fluid simulations)
        self.ds.hottailgrid.setEnabled(False)
        self.ds.runawaygrid.setEnabled(False)

        # Set collision settings
        self.ds.collisions.collfreq_mode        = Collisions.COLLFREQ_MODE_FULL
        self.ds.collisions.collfreq_type        = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        self.ds.collisions.bremsstrahlung_mode  = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
        self.ds.collisions.lnlambda             = Collisions.LNLAMBDA_ENERGY_DEPENDENT
        self.ds.collisions.pstar_mode           = Collisions.PSTAR_MODE_COLLISIONAL

        # Set fluid RE generation
        self.ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
        self.ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID)
        self.ds.eqsys.n_re.setHottail(RE.HOTTAIL_MODE_ANALYTIC_ALT_PC)
        self.ds.eqsys.n_re.setCompton(RE.COMPTON_RATE_ITER_DMS)
        # REs due to tritium decay is enabled later on if nT > 0

        # Use Sauter formula for conductivity
        self.ds.eqsys.j_ohm.setConductivityMode(JOhm.CONDUCTIVITY_MODE_SAUTER_COLLISIONAL)

        # Set the magnetic field from specified Tokamak (see imports)
        Tokamak.setMagneticField(self.ds, nr=NR)


    def run(self, handleCrash=None):
        """
        Runs the simulation and creates single simulation Output object from
        several DREAM output objects. By default it runs an exponential decay
        for the thermal quench and then switches to a self consistent evolution
        of temperature. If this DREAMSimulation is initialized with the mode
        TQ_MODE_TRANSPORT, it will instead run a thermal quench induced by a perturbed
        magnetic field.

        :param handleCrash:     If True, any crashed simulation are rerun in
                                higher resolution in time, until a maximum
                                number of MAX_RERUNS is performed. Then the
                                simulation output is left as None.
        """
        assert self.output is None, 'Output object already exists!'

        if handleCrash is not None:
            self.handleCrash = handleCrash

        self._setInitialProfiles()

        ## PRESCRIBED EXPONENTIAL DECAYING TEMPERATURE
        if self.mode == TQ_MODE_EXPDECAY:

            # Set exponential decaying temperature
            t, r, T = Tokamak.getTemperatureEvolution(self.input.T1, self.input.T2, tau0=TQ_DECAY_TIME, T_final=TQ_FINAL_TEMPERATURE, tmax=TMAX_TQ)#, nt=NT_TQ)
            self.ds.eqsys.T_cold.setPrescribedData(T, radius=r, times=t)

            # Massive material injection
            do1 = self._runMMI('1', NT_IONIZ, TMAX_IONIZ)

            # run TQ part of simulation
            do2 = self._getDREAMOutput('2', NT_TQ, TMAX_TQ - TMAX_IONIZ)

            # Change to self consistent temperature evolution
            self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

            # Set transport settings
            r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, self.input.dBB1, self.input.dBB2)
            self._setTransport(dBB, r, NT_CQ, TMAX_TOT - TMAX_TQ - TMAX_IONIZ)

            # Run CQ and runaway plateau part of simulation
            do3 = self._getDREAMOutput('3', NT_CQ, TMAX_TOT - TMAX_TQ - TMAX_IONIZ)

            # Set output from DREAM output
            self.output = self.Output(do1, do2, do3)


        ## TRANSPORT INDUCED THERMAL QUENCH
        elif self.mode == TQ_MODE_TRANSPORT:

            # Set inital temperature
            rT, T = Tokamak.getInitialTemperature(self.input.T1, self.input.T2)
            self.ds.eqsys.T_cold.setInitialProfile(T, radius=rT)

            # Set to self consistent temperature evolution
            self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

            # Set the initial magnetic pertubation
            r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, TQ_INITIAL_dBB0, 0)
            self._setTransport(dBB, r, NT_IONIZ, TMAX_IONIZ)

            # Massive material injection
            do1 = self._runMMI('1', NT_IONIZ, TMAX_IONIZ)

            # Test run TQ and obtain time when the temperature reaches a certain value
            do2 = self._getDREAMOutput('2', NT_TQ, TMAX_TQ - TMAX_IONIZ)
            tmpOut = self.Output(do1, do2, close=False)
            tmax = tmpOut._getTime(tmpOut.T_cold[:,0], TQ_STOP_FRACTION)
            do2.close()

            if tmax is None:
                msg = f'Final core temperature {tmpOut.T_cold[-1,0]} did not reach {tmpOut.T_cold[0,0] * TQ_STOP_FRACTION}'
                raise TransportException(msg)
            else:
                self.tStop = tmax

            # Restart TQ simulation and stop at tmax
            out_ioniz = self._getFileName('1', OUTPUT_DIR)
            self.ds = DREAMSettings(self.ds)
            self.ds.fromOutput(out_ioniz)
            do3 = self._getDREAMOutput('3', NT_TQ, self.tStop - TMAX_IONIZ)

            # Set the final magnetic pertubation
            r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, TQ_INITIAL_dBB0*1e-1, 0)
            self._setTransport(dBB, r,  NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

            # Run CQ and runaway plateau part of simulation
            do4 = self._getDREAMOutput('4', NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

            # Set output from DREAM output
            self.output = self.Output(do1, do3, do4)

        else:
            raise AttributeError(f'Unexpected mode value mode={self.mode}.')


        return 0


    ###### SIMULATION HELPER FUNCTIONS #######

    def _getDREAMOutput(self, name, nt, tmax):
        """
        Runs DREAM simulation with given time resolution options and returns the
        resulting DREAM output object.
        """
        self.ds.timestep.setNt(nt)
        self.ds.timestep.setTmax(tmax)
        out = self._getFileName(name, OUTPUT_DIR)
        self.ds.output.setFilename(out)
        do = self._run(out=out)
        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out)
        return do


    def _setInitialProfiles(self):
        """
        Set initial profiles from input parameters.
        """
        # Add fuel
        if self.input.nH:
            self.ds.eqsys.n_i.addIon('H', n=self.input.nH, Z=1, Z0=1, hydrogen=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nD:
            self.ds.eqsys.n_i.addIon('D', n=self.input.nD, Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_TRANSPARENT)
        if self.input.nT:
            self.ds.eqsys.n_i.addIon('T', n=self.input.nT, Z=1, Z0=1, tritium=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_TRANSPARENT)
            self.ds.eqsys.n_re.setTritium(True)
        if self.input.nHe:
            raise NotImplementedError('Helium is not yet implemented...')

        # Set prescribed electric field
        self.ds.eqsys.E_field.setPrescribedData(1e-4)

        # Set initial temperature profile
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)
        rT, T = Tokamak.getInitialTemperature(self.input.T1, self.input.T2)
        self.ds.eqsys.T_cold.setPrescribedData(T, radius=rT)

        # Background free electron density from ions
        nfree, rn0 = self.ds.eqsys.n_i.getFreeElectronDensity()
        self.ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T)

        # We need to access methods from within a DREAM output object
        self.ds.timestep.setTmax(1e-11)
        self.ds.timestep.setNt(1)
        self.do = self._run(verbose=False)

        # From now on, we store but a small subset of all timesteps to reduce memory use
        self.ds.timestep.setNumberOfSaveSteps(200)

        # Set self-consistent electric field (initial condition is determined by the current density)
        self.ds.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
        self.ds.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0, R0=Tokamak.R0)
        self.ds.solver.tolerance.set('psi_wall', abstol=1e-6)

        # Set initial current density
        rj, j = Tokamak.getInitialCurrentDensity(self.input.j1, self.input.j2, NR)
        self.ds.eqsys.j_ohm.setInitialProfile(j, radius=rj, Ip0=self.input.Ip0)


    def _setTransport(self, dBB, r, nt, tmax):
        """
        Configures the transport settings.
        """
        assert dBB.shape == r.shape

        t = np.linspace(0, tmax, nt)

        if TRANSPORT_COLD: # Enable radial transport of cold electrons
            self.ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
            self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB[0])#=np.tile(dBB, (nt, 1)), r=r, t=t)

        if TRANSPORT_RE: # Enable radial transport of REs
            # Drr, xi, p = utils.getDiffusionOperator(dBB, R0=Tokamak.R0)
            # Drr = np.tile(Drr, (nt,1,1,1))
            Drr = utils.getDiffusionOperator(dBB, R0=Tokamak.R0, svensson=False)
            self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)
            self.ds.eqsys.n_re.transport.prescribeDiffusion(Drr[0])
            # self.ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
            # self.ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs
            #
            # # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
            # self.ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r, p=p, xi=xi, interp1d=Transport.INTERP1D_NEAREST)
            # self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)


    def _runMMI(self, name, nt, tmax):
        """
        Injects neutral gas and run a short ionization simulation to allow them
        to settle.
        """
        # Add injected materials
        if self.input.nD2:
            r, n = utils.getDensityProfile(self.do, self.input.nD2, self.input.aD2)
            self.ds.eqsys.n_i.addIon('D2', Z=1, iontype=Ions.IONS_DYNAMIC, Z0=0, n=n, r=r,
            opacity_mode=Ions.ION_OPACITY_MODE_TRANSPARENT)

        if self.input.nNe:
            r, n = utils.getDensityProfile(self.do, self.input.nNe, self.input.aNe)
            self.ds.eqsys.n_i.addIon('Ne', Z=10, iontype=Ions.IONS_DYNAMIC, Z0=0, n=n, r=r)

        self.ds.solver.tolerance.set(reltol=1e-2)

        return self._getDREAMOutput(name, nt, tmax)


    def _run(self, out=None, verbose=None, ntmax=None, getTmax=False):
        """
        Run single simulation from DREAMSettings object and handles crashes.
        """

        do = None
        if verbose is None:
            quiet = (not self.verbose)
        else:
            quiet = (not verbose)

        try:
            do = runiface(self.ds, out, quiet=quiet)
            utils.checkElectronDensityRatio(do, exc=SimulationException)
            return do

        except DREAMException as err:
            if self.handleCrash:
                tmax = self.ds.timestep.tmax
                nt = self.ds.timestep.nt
                if ntmax is None:
                    ntmax = 2**MAX_RERUNS * max(NT_IONIZ, NT_TQ, NT_CQ)
                if nt >= ntmax:
                    raise MaximumIterationsException('ERROR: MAXIMUM NUMBER OF RERUNS REACHED!') from err
                else:
                    print(err)
                    print(f'WARNING : Number of iterations is increased from {nt} to {2*nt}!')
                    self.ds.timestep.setNt(2*nt)
                    return self._run(out=out, verbose=verbose, ntmax=ntmax)
            else:
                raise DREAMCrashedException('ERROR: DREAM CRASHED!') from err


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


    s = DREAMSimulation(mode=TQ_MODE_TRANSPORT)
    s.configureInput(nNe=1.5e19, nD2=7e20)

    s.run(handleCrash=False)

    if s.mode == TQ_MODE_TRANSPORT:
        print(f'tTQ ~ {s.tStop*1e3} ms')

    print(f'tCQ = {s.output.getCQTime()*1e3} ms')
    s.output.visualizeTemperatureEvolution(radii=[0,-1], show=True)
    s.output.visualizeCurrents(show=True)

    return 0

if __name__ == '__main__':
    sys.exit(main())
