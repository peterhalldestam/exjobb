#!/usr/bin/env python3

import sys, os
import numpy as np
from dataclasses import dataclass

import sim.DREAM.DREAMSimulation as sim

import DREAM.Settings.Equations.ColdElectronTemperature as Temperature

TQ_DECAY_TIME           = 1e-3
TQ_FINAL_TEMPERATURE    = 10  # 20 kev -> 10 eV

TMAX_TOT    = 1.5e-1
TMAX_IONIZ  = 1e-6
TMAX_TQ     = 8e-3

NT_IONIZ    = 2000#1000
NT_TQ       = 6000#4000
NT_CQ       = 10000#4000


class ExponentialDecaySimulation(sim.DREAMSimulation):
    """
    Disruption simulation with initially a prescribed exponential decaying
    temperature to model the thermal quench.
    """
    @dataclass
    class Input(sim.DREAMSimulation.Input):
        """ Include DREAMSimulation input and TQ settings. """
        TQ_decay_time:          float = TQ_DECAY_TIME
        TQ_final_temperature:   float = TQ_FINAL_TEMPERATURE

        nt_ioniz:   int = NT_IONIZ
        nt_TQ:      int = NT_TQ
        nt_CQ:      int = NT_CQ

        tmax_tot:   float = TMAX_TOT
        tmax_ioniz: float = TMAX_IONIZ
        tmax_TQ:    float = TMAX_TQ

        @property
        def temperatureTQ(self):
            """ Exponential decaying temperature during thermal quench. """
            nt = 100
            r, T0 = self.initialTemperature
            T1 = self.TQ_final_temperature
            t = np.linspace(0, TMAX_TQ, nt).reshape((nt,1))
            T = T1 + (T0 - T1) * np.exp(-t / self.TQ_decay_time)
            return t.reshape((nt,)), r, T


    #### DISRUPTION SIMULATION SETUP ######

    def  __init__(self, id='out_expDecay', verbose=True, **inputs):
        """ Constructor. """
        super().__init__(id=id, verbose=verbose, **inputs)

        # Resolution parameters
        self.nt_ioniz   = self.input.nt_ioniz
        self.nt_TQ      = self.input.nt_TQ
        self.nt_CQ      = self.input.nt_ioniz
        self.tmax_ioniz = self.input.tmax_ioniz
        self.tmax_TQ    = self.input.tmax_TQ - self.input.tmax_ioniz
        self.tmax_CQ   = self.input.tmax_tot - self.input.tmax_TQ - self.input.tmax_ioniz


    def run(self, handleCrash=None):
        """ Run simulation. """
        super().run(handleCrash=handleCrash)

        self.setInitialProfiles()

        # Set exponential decaying temperature
        tT, rT, T = self.input.temperatureTQ
        self.ds.eqsys.T_cold.setPrescribedData(T, radius=rT, times=tT)

        # Massive material injection
        do1 = self.setMMI()

        # Let the ions settle
        do1 = self.runDREAM('1', self.nt_ioniz, self.tmax_ioniz)

        # run rest of TQ part of simulation
        do2 = self.runDREAM('2', self.nt_TQ, self.tmax_TQ)

        # Change to self consistent temperature evolution
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)

        # Set transport settings
        self.setTransport(self.input.dBB0, self.input.dBB1, self.nt_CQ, self.tmax_CQ)

        # Run CQ and runaway plateau part of simulation
        do3 = self.runDREAM('3', self.nt_CQ, self.tmax_CQ)

        # Set output from DREAM output
        self.output = self.Output(do1, do2, do3)


def main():

    s = ExponentialDecaySimulation()

    # s.configureInput()
    s.run(handleCrash=False)

    print(f'tCQ = {s.output.currentQuenchTime*1e3} ms')

    s.output.visualizeTemperatureEvolution(r=[0,-1], show=True)
    s.output.visualizeCurrents(show=True)

    return 0

if __name__ == '__main__':
    sys.exit(main())
