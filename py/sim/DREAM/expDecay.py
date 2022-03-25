

from DREAMSimulation import DREAMSimulation


class ExponentialDecaySimulation(DREAMSimulation):

    @dataclass(init=False)
    class Output(DREAMSimulation.Output):
        """
        Additional output variables other than what is defined in the superclass.
        These are
        """
        P_trans:    np.ndarray  # rate of energyloss through plasma edge [J s^-1 m^-1]
        P_rad:      np.ndarray  # radiated power density [J s^-1 m^-1]

        def __init__(self, *dos, close=True):
            self.P_trans    = utils.join('other.scalar.energyloss_T_cold.data', dos, other=True)
            self.P_rad      = utils.join('other.fluid.Tcold_radiation.integral()', dos, other=True)
            super().__init__(dos, close=close)

        @property
        def transportedFraction(self):
            """
            Fraction of energy loss caused by transport through the plasma edge.
            """
            dt = self.t[1:] - self.t[:-1]
            Q_trans = np.sum(self.P_trans[:,0] * dt)
            Q_rad = np.sum(self.P_rad * dt)
            Q_tot = Q_trans + Q_rad
            return Q_trans/Q_tot

    #### DISRUPTION SIMULATION SETUP ######

    def  __init__(self, id='out_expDecay', verbose=True, **inputs):
        """
        Constructor.
        """
        super().__init__(id=id, verbose=verbose, **inputs)
        self.ds.other.include(['scalar.energyloss_T_cold', 'fluid.Tcold_radiation'])

    def run(self, handleCrash=None):
        """
        Run simulation.
        """
        assert self.output is None, 'Output object already exists!'

        self._setInitialProfiles()
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
        r, dBB = utils.getQuadraticMagneticPerturbation(self.ds, self.input.dBB1, self.input.dBB2)
        self._setTransport(dBB, r,  NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Run CQ and runaway plateau part of simulation
        do4 = self._getDREAMOutput('4', NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Set output from DREAM output
        self.output = self.Output(do1, do3, do4)
