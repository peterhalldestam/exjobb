#!/usr/bin/env python3
import sys, os
import numpy as np
from dataclasses import dataclass

import utils
import sim.DREAM.DREAMSimulation as sim
from sim.simulationException import SimulationException

from DREAM import DREAMSettings
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature

OUTPUT_ID = 'out_transport'
OUTPUT_DIR = 'outputs/'


TQ_STOP_FRACTION    = 1 / 1000  # 20 kev -> 20 eV
TQ_INITIAL_dBB0     = 4e-3


TMAX_TOT    = 1.5e-1
TMAX_IONIZ  = 2e-6
TMAX_TQ     = 10e-3


NT_IONIZ    = 6000 # previously 3000
NT_TQ       = 8000
NT_CQ       = 14000


class TransportException(SimulationException):
    pass

class TransportSimulation(sim.DREAMSimulation):
    """
    Disruption simulation with the thermal quench driven by radial transport.
    """

    @dataclass
    class Input(sim.DREAMSimulation.Input):
        """ Include DREAMSimulation input and TQ settings. """
        TQ_stop_fraction:   float = TQ_STOP_FRACTION
        TQ_initial_dBB0:    float = TQ_INITIAL_dBB0

        nt_ioniz:   int = NT_IONIZ
        nt_TQ:      int = NT_TQ
        nt_CQ:      int = NT_CQ

        tmax_tot:   float = TMAX_TOT
        tmax_ioniz: float = TMAX_IONIZ
        tmax_TQ:    float = TMAX_TQ

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
            self.W_cold      = utils.join('eqsys.W_cold.integral()', dos, other=True)[0]
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

    def  __init__(self, transport_cold=True, transport_re=True, svensson=False, id=OUTPUT_ID, verbose=True, **inputs):
        """ Constructor. """
        super().__init__(transport_cold=transport_cold, transport_re=transport_re, svensson=svensson, id=id, verbose=verbose, **inputs)
        self.ds.other.include(['scalar'])

        # Resolution parameters
        self.nt_ioniz   = self.input.nt_ioniz
        self.nt_TQ      = self.input.nt_TQ
        self.nt_CQ      = self.input.nt_CQ
        self.tmax_ioniz = self.input.tmax_ioniz
        self.tmax_TQ    = self.input.tmax_TQ - self.input.tmax_ioniz
        self.tmax_CQ    = self.input.tmax_tot - self.input.tmax_TQ - self.input.tmax_ioniz


    def run(self, handleCrash=None):
        """ Run simulation. """
        super().run(handleCrash=handleCrash)

        self.setInitialProfiles()

        dBB  = self.input.TQ_initial_dBB0
        #T0   = self.do.eqsys.T_cold.data[0,0] *1.6e-19
        n0   = self.do.eqsys.n_cold.data[0,0]
        R0   = self.do.grid.R0[0]
        a    = self.do.grid.a[0]
        m_e  = 9.1e-31
        q    = 1

        W_cold = self.do.eqsys.W_cold.data[0,0]
        v_th = np.sqrt(2*W_cold/(m_e*n0))
        
        D = np.pi*R0*q*v_th*dBB**2
        tau = a**2/D
        #self.tmax_TQ = self.tmax_ioniz + np.min([8*tau, TMAX_TQ])
        self.nt_TQ = int((self.tmax_TQ - self.tmax_ioniz)*500e3)

        # Set to self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

        # Set inital temperature
        rT, T = self.input.initialTemperature
        self.ds.eqsys.T_cold.setInitialProfile(T, radius=rT)

        # Set the initial magnetic pertubation
        self.setTransport(self.input.TQ_initial_dBB0, 0, self.nt_ioniz, self.tmax_ioniz)



        # Massive material injection
        self.setMMI()

        # Let the ions settle
        do1 = self.runDREAM('1', self.nt_ioniz, self.tmax_ioniz)

        # Test run TQ and obtain time when the temperature reaches a certain value
        do2 = self.runDREAM('2', self.nt_TQ, self.tmax_TQ - self.tmax_ioniz)
        tmpOut = self.Output(do1, do2, close=False)
        tmax = tmpOut.getTime(tmpOut.averageTemperature, self.input.TQ_stop_fraction)


        if tmax is None:
            os.remove(do1.filename)
            do1.close()
            os.remove(do2.filename)
            do2.close()
            
            msg = f'Final core temperature {tmpOut.T_cold[-1,0]} did not reach {tmpOut.T_cold[0,0] * TQ_STOP_FRACTION}'
            raise TransportException(msg)
        else:
            self.tStop = tmax

        # Restart TQ simulation and stop at tmax
        out_ioniz = self.getFilePath('1', OUTPUT_DIR)
        self.ds = DREAMSettings(self.ds)
        self.ds.fromOutput(out_ioniz)
        do3 = self.runDREAM('3', int((self.tStop - self.tmax_ioniz)*500e3), self.tStop - self.tmax_ioniz)

        # Set the final magnetic pertubation
        self.setTransport(self.input.dBB0, self.input.dBB1,  self.nt_CQ, self.tmax_CQ - self.tStop - self.tmax_ioniz)

        # Run CQ and runaway plateau part of simulation
        do4 = self.runDREAM('4', self.nt_CQ, self.tmax_CQ - self.tStop - self.tmax_ioniz)

        # Remove old output file
        os.remove(do2.filename)
        do2.close()
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
