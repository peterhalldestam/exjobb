import sys, os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict

# sys.path.append(os.path.abspath('..'))
import utils
import tokamaks.ITER as Tokamak # This sets the used tokamak geometry
import sim.simulation as sim
from sim.simulationException import SimulationException

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
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

CHECK_OUTPUT = True     # Check if n_re / n_cold > 1e-2 post simulation
REMOVE_FILES = False     # Removes output files post simulation

# Number of radial nodes
NR = 20

# Maximum no. iterations
NT_MAX = 30_000

# Where to store DREAMOutput files during simulation
OUTPUT_DIR = 'outputs/'



# Custom exceptions

class MaximumIterationsException(SimulationException):
    pass

class DREAMCrashedException(SimulationException):
    pass

class DREAMSimulation(sim.Simulation):
    """
    Description...
    """

    ############## DATA CLASSES ##############

    @dataclass
    class Input(sim.Simulation.Input):
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
        cD2:    float = 0.
        cNe:    float = 0.
        #...

        # Initial current density profile
        j1:     float = 1.
        j2:     float = .41
        Ip0:    float = Tokamak.Ip

        # Initial temperature profile
        T1:     float = Tokamak.T_initial
        T2:     float = .99

        # Post TQ magnetic perturbation profile
        dBB0:   float = 0.#4e-4
        dBB1:   float = 0.

        # Inverse wall time
        tau0:   float = 0.

        @property
        def asDict(self):
            """ Returns input data as dictionary. """
            return {key: val for key, val in asdict(self).items() if self.__dataclass_fields__[key].repr}

        @property
        def initialTemperature(self):
            """ Initial temperature profile. """
            r, T = Tokamak.getInitialTemperature(self.T1, self.T2, NR)
            return r, T

        @property
        def initialCurrentDensity(self):
            """ Initial curent density profile. """
            r, j, I = Tokamak.getInitialCurrentDensity(self.j1, self.j2, self.Ip0, NR)
            return r, j, I

        def getInitialDensity(self, do: DREAMOutput, n: float, c: float):
            """ Initial density MMI profiles. """
            r, n = utils.getDensityProfile(do, n, c)
            return r, n

        def getMagneticPerturbation(self, do: DREAMOutput, dBB0: float, dBB1: float):
            """ Magnetic perturbation profiles. """
            r, dBB = utils.getMagneticPerturbation(do, dBB0, dBB1)
            return r, dBB

        def getTransportTime(self, do: DREAMOutput, dBB0: float, dBB1: float):
            """ Characteristic transport time. """
            pass


    @dataclass(init=False)
    class Output(sim.Simulation.Output):
        """
        Output variables from the DREAM simulation. The constructor expects one
        or more DREAMOutput objects as arguments.
        """
        dos: DREAMOutput

        # Output quantities from DREAM output object
        r:          np.ndarray  # radial grid
        t:          np.ndarray  # simulation time
        I_re:       np.ndarray  # RE current
        I_ohm:      np.ndarray  # Ohmic current
        I_tot:      np.ndarray  # total current
        T_cold:     np.ndarray  # cold electron temperature

        def __init__(self, *dos, close=True):
            """
            Constructor. Joins data from the provided DREAM output objects.
            """
            self.t          = utils.join('grid.t', dos, time=True)
            self.r          = utils.join('grid.r', dos, radius=True)
            self.I_re       = utils.join('eqsys.j_re.current()', dos)
            self.I_ohm      = utils.join('eqsys.j_ohm.current()', dos)
            self.I_tot      = utils.join('eqsys.j_tot.current()', dos)
            self.T_cold     = utils.join('eqsys.T_cold.data', dos)
            self.j_ohm      = utils.join('eqsys.j_ohm.data', dos)

            assert len(self.r) == NR
            assert all(I.shape == self.t.shape for I in [self.I_re, self.I_ohm, self.I_tot])

            for do in dos:
                if close:
                    if REMOVE_FILES:
                        os.remove(do.filename)
                    do.close()


        @property
        def finalOhmicCurrent(self):
            """ Ohmic current in the final time step. """
            return self.I_ohm[-1]

        @property
        def currentQuenchTime(self):
            """ Calculate CQ time, unable  it will return infinity. """
            t80, _ = self.getTime(self.I_ohm, .8)
            t20, _ = self.getTime(self.I_ohm, .2)
            if t80 is not None and t20 is not None:
                return (t20 - t80) / .6
            else:
                return 1e10

        @property
        def maxRECurrent(self):
            """ Maximum runaway electron current. """
            return self.I_re.max()

        def getTime(self, arr, x):
            """  Get first time t when arr(t) / max(arr) = x is satisfied """
            assert len(arr) == len(self.t)
            assert 0 < x < 1
            maxVal = max(arr)
            for t, val in zip(self.t, arr):
                if val <= x * maxVal:
                    return t, val
            return None, None


        def visualizeCurrents(self, log=False, ax=None, show=False):
            """ Plot Ohmic, RE and total current. """
            return utils.visualizeCurrents(self.t, self.I_ohm, self.I_re, self.I_tot, log=log, ax=ax, show=show)

        def visualizeTemperature(self, times=[0,-1], ax=None, show=False):
            """ Plot temperature profile at selected timesteps. """
            return utils.visualizeTemperature(self.r, self.T_cold, times=times, ax=ax, show=show)

        def visualizeTemperatureEvolution(self, r=[0], ax=None, show=False):
            """ Plot temperature evolution at selected radial nodes. """
            return utils.visualizeTemperatureEvolution(self.t, self.T_cold, r=r, ax=ax, show=show)

        def visualizeCurrentDensity(self, ax=None, show=False):
            """ Plot Ohmic current density over time. """
            return utils.visualizeCurrentDensity(self.t, self.r, self.j_ohm,  ax=ax, show=show)

        # def visualizeOutputVariables(self, ax=None, show=False):
        #     """ Plot currents and indicate CQ time, max RE and final Ohmic current. """
        #     return
        #
        #
        #     ax = self.visualizeCurrents(log=False, ax=ax, show=False)
        #
        #
        #     # add CQ time
        #     t80, ohm80 = self.getTime(self.I_ohm, .8)
        #     t20, ohm20 = self.getTime(self.I_ohm, .2)
        #     t20 *= 1e3
        #     t80 *= 1e3
        #     ohm20 *= 1e-6
        #     ohm80 *= 1e-6
        #     assert t80 is not None and t20 is not None, 'no full CQ!'
        #     ax.plot([t80, t80, t20], [ohm80, ohm20, ohm20], 'k')
        #
        #     # add maximal RE current
        #     t_re = self.t[np.argmax(self.I_re)] * 1e3
        #     max_I_re = self.maxRECurrent * 1e-6
        #     ax.plot([self.t[0], t_re, t_re], [max_I_re, max_I_re, 0], 'k')
        #
        #     # add final Ohmic current
        #     final_I_ohm = self.finalOhmicCurrent * 1e-6
        #     ax.plot([self.t[0] * 1e3, self.t[-1] * 1e3], [final_I_ohm, final_I_ohm], 'k')function
        #
        #     plt.show()


    ############## DISRUPTION SIMULATION SETUP ##############

    def __init__(self, transport_cold=True, transport_re=True, svensson=False, id='out', verbose=True, **inputs):
        """ Constructor. Core simulation settings. """
        super().__init__(id=id, verbose=verbose, **inputs)

        assert not (svensson and not transport_re), 'Svensson transport requires RE transport to be active'

        # Transport options
        self.transport_cold = transport_cold
        self.transport_re   = transport_re
        self.svensson       = svensson

        self.outputDir = OUTPUT_DIR # relative dir path
        self._handleCrash = True
        self._removeFilesIfCrash = True

        self.dos = []
        self.ds = None      # This will be updated for each subsequent simulation.
        self.do = None      # We need access to do.grid.integrate()

        ##### Generate the initialization simulation #####

        self.ds = DREAMSettings()

        self.ds.other.include(['fluid'])

        # Set solver settings
        self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        self.ds.solver.setType(Solver.NONLINEAR)
        self.ds.solver.setMaxIterations(maxiter=500)

        self.ds.solver.tolerance.set(unknown='n_re', reltol=2e-6, abstol=1e5)
        self.ds.solver.tolerance.set(unknown='j_re', reltol=2e-6, abstol=1e-5) # j ~ e*c*n_e ~ n_e*1e-10 ?
        self.ds.solver.tolerance.set(reltol=1e-2)

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
        self.ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)
        self.ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID)
        self.ds.eqsys.n_re.setHottail(Runaways.HOTTAIL_MODE_ANALYTIC_ALT_PC)
        self.ds.eqsys.n_re.setCompton(Runaways.COMPTON_RATE_ITER_DMS)
        # REs due to tritium decay is enabled later on if nT > 0

        # Use Sauter formula for conductivity
        self.ds.eqsys.j_ohm.setConductivityMode(JOhm.CONDUCTIVITY_MODE_SAUTER_COLLISIONAL)

        # Set the magnetic field from specified Tokamak (see imports)
        Tokamak.setMagneticField(self.ds, nr=NR)


    def run(self, handleCrash=None):
        """ Run the simulation. """
        assert self.output is None, 'Output object already exists!'
        if handleCrash is not None:
            self._handleCrash = handleCrash



    ###### SIMULATION HELPER FUNCTIONS #######

    def setInitialProfiles(self):
        """
        Set initial profiles from input parameters.
        """
        # Add fuel
        if self.input.nH:
            self.ds.eqsys.n_i.addIon('H', n=self.input.nH, Z=1, Z0=1, hydrogen=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nD:
            self.ds.eqsys.n_i.addIon('D', n=self.input.nD, Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        if self.input.nT:
            self.ds.eqsys.n_i.addIon('T', n=self.input.nT, Z=1, Z0=1, tritium=True, iontype=Ions.IONS_DYNAMIC, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
            self.ds.eqsys.n_re.setTritium(True)
        if self.input.nHe:
            raise NotImplementedError('Helium is not yet implemented...')

        # Set initial temperature profile
        rT, T = self.input.initialTemperature
        self.ds.eqsys.T_cold.setPrescribedData(T, radius=rT)
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)

        # Background free electron density from ions
        nfree, rn0 = self.ds.eqsys.n_i.getFreeElectronDensity()
        self.ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, n0=nfree, rT0=rT, T0=T)

        # Set dummy electric field
        self.ds.eqsys.E_field.setPrescribedData(1e-4)

        # We need to access methods from within a DREAM output object
        self.ds.timestep.setTmax(1e-11)
        self.ds.timestep.setNt(1)
        self.do = self._run(quiet=True)

        # From now on, we store but a small subset of all timesteps to reduce memory use
        self.ds.timestep.setNumberOfSaveSteps(200)

        # Set self-consistent evolution of the electric field
        self.ds.eqsys.E_field.setType(EField.TYPE_SELFCONSISTENT)
        self.ds.eqsys.E_field.setBoundaryCondition(EField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=self.input.tau0, R0=Tokamak.R0)
        self.ds.solver.tolerance.set('psi_wall', abstol=1e-6)

        # Set initial current density
        rj, j, Ip = self.input.initialCurrentDensity
        self.ds.eqsys.j_ohm.setInitialProfile(j, radius=rj, Ip0=Ip)


    def setTransport(self, dBB0, dBB1, nt, tmax):
        """
        Configures DREAM transport settings for given magnetic perturbation
        profile, number of timesteps and maximum simulation time.
        """
        r, dBB = self.input.getMagneticPerturbation(self.do, dBB0, dBB1)
        R0dBB = np.sqrt(Tokamak.R0) * dBB

        t = np.linspace(0, tmax, nt)

        if self.transport_cold:
            self.ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
            self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=np.tile(R0dBB, (nt, 1)), r=r, t=t)

        if self.transport_re:
            if self.svensson:
                Drr, xi, p = utils.getDiffusionOperator(dBB, R0=Tokamak.R0)
                Drr = np.tile(Drr, (nt,1,1,1))

                self.ds.eqsys.n_re.transport.setSvenssonInterp1dParam(Transport.SVENSSON_INTERP1D_PARAM_TIME)
                self.ds.eqsys.n_re.transport.setSvenssonPstar(0.5) # Lower momentum boundry for REs

                # Used nearest neighbour interpolation thinking it would make simulations more efficient since the coefficient for the most part won't be varying with time.
                self.ds.eqsys.n_re.transport.setSvenssonDiffusion(drr=Drr, t=t, r=r, p=p, xi=xi, interp1d=Transport.INTERP1D_NEAREST)
                self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

            else:
                Drr = utils.getDiffusionOperator(dBB, R0=Tokamak.R0, svensson=False)
                Drr = np.tile(Drr, (nt, 1))
                self.ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)
                self.ds.eqsys.n_re.transport.prescribeDiffusion(Drr, t=t, r=r)


    def setMMI(self):
        """ Injects neutral gas from input parameters. """
        if self.input.nD2:
            r, n = self.input.getInitialDensity(self.do, self.input.nD2, self.input.cD2)
            self.ds.eqsys.n_i.addIon('D2', Z=1, iontype=Ions.IONS_DYNAMIC, Z0=0, n=n, r=r,
            opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)

        if self.input.nNe:
            r, n = self.input.getInitialDensity(self.do, self.input.nNe, self.input.cNe)
            self.ds.eqsys.n_i.addIon('Ne', Z=10, iontype=Ions.IONS_DYNAMIC, Z0=0, n=n, r=r)


    def getFilePath(self, text, dir):
        """ Returns path of file and makes sure its directory exist. """
        filename = f'{dir}{self.id}_{text}.h5'
        path = pathlib.Path(filename)
        dir = path.parent.resolve()
        if not dir.exists():
            dir.mkdir(parents=True)
        fp = str(path.absolute())
        return fp


    def runDREAM(self, name, nt, tmax):
        """
        Runs DREAM simulation with given time resolution options and returns the
        resulting DREAM output object.
        """
        self.ds.timestep.setNt(nt)
        self.ds.timestep.setTmax(tmax)
        out = self.getFilePath(name, self.outputDir)
        self.ds.output.setFilename(out)
        do = self._run(out=out)
        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out)
        self.dos.append(do)
        return do


    def _run(self, out=None, quiet=None):
        """  Run simulation and handle any crashes. """
        do = None
        if quiet is None:
            quiet = not self.verbose

        try:
            do = runiface(self.ds, out, quiet=quiet)
            utils.checkElectronDensityRatio(do, exc=SimulationException)
            return do

        except DREAMException as err:
            if self._handleCrash:
                tmax = self.ds.timestep.tmax
                nt = self.ds.timestep.nt
                if nt >= NT_MAX:
                    if self._removeFilesIfCrash:
                        for do in self.dos:
                            os.remove(do.filename)

                    raise MaximumIterationsException('ERROR: MAXIMUM NUMBER OF TIMESTEPS REACHED!') from err
                else:
                    print(err)
                    print(f'WARNING : Number of iterations is increased from {nt} to {2*nt}!')
                    self.ds.timestep.setNt(2*nt)
                    return self._run(out=out)
            else:
                raise DREAMCrashedException('ERROR: DREAM CRASHED!') from err
