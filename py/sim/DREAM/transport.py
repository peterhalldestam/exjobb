import sys, os
import numpy as np
from dataclasses import dataclass

import utils
from sim.DREAM.DREAMSimulation import DREAMSimulation

import DREAM.Settings.Equations.ColdElectronTemperature as Temperature


TQ_STOP_FRACTION = 1 / 2000  # 20 kev -> 10 eV
TQ_INITIAL_dBB0 = 3.5e-3

TMAX_IONIZ  = 1e-6
TMAX_TQ     = 8e-3
NT_IONIZ    = 1000
NT_TQ       = 2000
NT_CQ       = 6000


class TransportSimulation(DREAMSimulation):
    """
    Disruption simulation with the thermal quench driven by radial transport.
    """


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

    def  __init__(self, transport_cold=True, transport_re=True, svensson=False, id='out_expDecay', verbose=True, **inputs):
        """
        Constructor.
        """
        super().__init__(transport_cold=transport_cold, transport_re=transport_re, svensson=svensson, id=id, verbose=verbose, **inputs)
        self.ds.other.include(['scalar'])


    def run(self, handleCrash=None):
        """
        Run simulation.
        """
        assert self.output is None, 'Output object already exists!'

        self._setInitialProfiles()
        # Set inital temperature
        rT, T = self._getInitialTemperature()
        self.ds.eqsys.T_cold.setInitialProfile(T, radius=rT)

        # Set to self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

        # Set the initial magnetic pertubation
        self._setTransport(TQ_INITIAL_dBB0, 0, NT_IONIZ, TMAX_IONIZ)

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
        self._setTransport(self.input.dBB1, self.input.dBB2,  NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Run CQ and runaway plateau part of simulation
        do4 = self._getDREAMOutput('4', NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Set output from DREAM output
        self.output = self.Output(do1, do3, do4)


def main():

    s = TransportSimulation()
    s.configureInput(nNe=1.5e19, nD2=7e20)

    s.run(handleCrash=False)


    print(f'tCQ = {s.output.getCQTime()*1e3} ms')
    s.output.visualizeTemperatureEvolution(radii=[0,-1], show=True)
    s.output.visualizeCurrents(show=True)
    s.output.getTransportedFraction()

    return 0

if __name__ == '__main__':
    sys.exit(main())
