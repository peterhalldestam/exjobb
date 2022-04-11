#!/usr/bin/env python3
import sys, os
import numpy as np
from dataclasses import dataclass

import utils
import sim.DREAM.DREAMSimulation as sim

from DREAM import DREAMSettings
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature

OUTPUT_ID = 'out_transport'
OUTPUT_DIR = 'outputs/'


TQ_STOP_FRACTION    = 1 / 2000  # 20 kev -> 10 eV
TQ_INITIAL_dBB0     = 4e-3

TMAX_TOT    = 1e-2
TMAX_IONIZ  = 1e-6
TMAX_TQ     = 8e-3

NT_IONIZ    = 1000
NT_TQ       = 2000
NT_CQ       = 2000


class TransportSimulation(sim.DREAMSimulation):
    """
    Disruption simulation with the thermal quench driven by radial transport.
    """

    @dataclass
    class Input(sim.DREAMSimulation.Input):
        """ Include DREAMSimulation input and TQ settings. """
        TQ_stop_fraction:   float = TQ_STOP_FRACTION
        TQ_initial_dBB0:    float = TQ_INITIAL_dBB0

    @dataclass(init=False)
    class Output(sim.DREAMSimulation.Output):
        """
        Additional output variables other than what is defined in the superclass.
        These are
        """
        P_trans:    np.ndarray  # rate of energyloss through plasma edge [J s^-1 m^-1]
        W_cold:     float       # initial thermal energy [J m^-1]

        def __init__(self, *dos, close=True):
            """ Constructor. """
            self.P_trans    = utils.join('other.scalar.energyloss_T_cold.data', dos, other=True)
            self.W_cold      = utils.join('other.eqsys.W_cold.integral()', dos, other=True)[0]
            super().__init__(*dos, close=close)

        @property
        def averageTemperature(self):
            """ Spatially averaged cold electron temperature. """
            return np.mean(self.T_cold, axis=1)

        @property
        def transportedFraction(self):
            """ Fraction of energy loss caused by transport through the plasma edge. """
            dt = self.t[1:] - self.t[:-1]
            Q_trans = np.sum(self.P_trans[:,0] * dt)
            return Q_trans/self.W_cold


    #### DISRUPTION SIMULATION SETUP ######

    def  __init__(self, transport_cold=True, transport_re=True, svensson=True, id=OUTPUT_ID, verbose=True, **inputs):
        """ Constructor. """
        super().__init__(transport_cold=transport_cold, transport_re=transport_re, svensson=svensson, id=id, verbose=verbose, **inputs)
        self.ds.other.include(['scalar'])


    def run(self, handleCrash=None):
        """ Run simulation. """
        super().run(handleCrash=handleCrash)

        self.setInitialProfiles()

        # Set to self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

        # Set inital temperature
        rT, T = self.input.initialTemperature
        self.ds.eqsys.T_cold.setInitialProfile(T, radius=rT)

        # Set the initial magnetic pertubation
        self.setTransport(self.input.TQ_initial_dBB0, 0, NT_IONIZ, TMAX_IONIZ)

        # Massive material injection
        self.setMMI()

        # Let the ions settle
        do1 = self.runDREAM('1', NT_IONIZ, TMAX_IONIZ)

        # Test run TQ and obtain time when the temperature reaches a certain value
        do2 = self.runDREAM('2', NT_TQ, TMAX_TQ - TMAX_IONIZ)
        tmpOut = self.Output(do1, do2, close=False)
        tmax = tmpOut.getTime(tmpOut.averageTemperature, self.input.TQ_stop_fraction)
        do2.close()

        if tmax is None:
            msg = f'Final core temperature {tmpOut.T_cold[-1,0]} did not reach {tmpOut.T_cold[0,0] * TQ_STOP_FRACTION}'
            raise TransportException(msg)
        else:
            self.tStop = tmax

        # Restart TQ simulation and stop at tmax
        out_ioniz = self.getFilePath('1', OUTPUT_DIR)
        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out_ioniz)
        do3 = self.runDREAM('3', NT_TQ, self.tStop - TMAX_IONIZ)

        # Set the final magnetic pertubation
        self.setTransport(self.input.dBB0, self.input.dBB1,  NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Run CQ and runaway plateau part of simulation
        do4 = self.runDREAM('4', NT_CQ, TMAX_TOT - self.tStop - TMAX_IONIZ)

        # Set output from DREAM output
        self.output = self.Output(do1, do3, do4)


def main():

    s = TransportSimulation()

    s.configureInput()

    s.run(handleCrash=False)

    print(f'tCQ = {s.output.currentQuenchTime*1e3} ms')
    print(f'transported fraction = {s.output.transportedFraction}')

    s.output.visualizeTemperatureEvolution(r=[0,-1], show=True)
    s.output.visualizeCurrents(show=True)

    return 0

if __name__ == '__main__':
    sys.exit(main())
