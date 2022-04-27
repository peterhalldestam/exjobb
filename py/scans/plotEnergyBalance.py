import os, sys
import numpy as np
import matplotlib.pyplot as plt

import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways

from sim.DREAM.DREAMSimulation import DREAMSimulation
from tokamaks import ITER as Tokamak

OHMIC_CURRENT_DENSITY = .8e6

DEUTERIUM_DENSITIES = [3e19]
NEON_DENSITIES      = [1e19]#, 2e19, 1e20]

NR = 151

NT_INIT             = 2
NT_RESTART_IONIZ    = 500
NT_RESTART_EQ       = 1000
NT_RESTART_RAD      = 2

TMAX_INIT           = 1e-11
TMAX_RESTART_IONIZ  = 2e-6
TMAX_RESTART_EQ     = 3e-2
TMAX_RESTART_RAD    = 1e-11

MAX_TEMPERATURE = 7e-1
MIN_TEMPERATURE = 2e3

def main():

    timegrid    = np.array([0])
    radialgrid  = np.linspace(0, Tokamak.a, NR)
    temperature = np.logspace(np.log10(MIN_TEMPERATURE), np.log10(MAX_TEMPERATURE), NR) * np.ones((1, NR))

    for nD2 in DEUTERIUM_DENSITIES:
        for nNe in NEON_DENSITIES:

            sim = DREAMSimulation()

            sim.ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
            sim.ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_NEGLECT)
            sim.ds.eqsys.n_re.setHottail(Runaways.HOTTAIL_MODE_DISABLED)
            sim.ds.eqsys.n_re.setCompton(Runaways.COMPTON_MODE_NEGLECT)

            # add initial deuterium
            sim.ds.eqsys.n_i.addIon(name='D', Z=1, n=sim.input.nD, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)

            # add initial tritium
            sim.ds.eqsys.n_i.addIon('T', Z=1, n=sim.input.nT, tritium=True, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)


            # add the injected deuterium
            sim.ds.eqsys.n_i.addIon('D2', Z=1, n=nD2, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)

            # add the injected neon
            sim.ds.eqsys.n_i.addIon('Ne', Z=10, n=nNe, iontype=Ions.IONS_DYNAMIC_NEUTRAL)


            # set temperatures to probe (with no heat transport, these are conveniently defined on the radial grid)
            sim.ds.eqsys.T_cold.setPrescribedData(temperature=temperature, times=timegrid, radius=radialgrid)

            # set arbitrary electric field
            efield = 1e-3 * np.ones((1, 2))
            sim.ds.eqsys.E_field.setPrescribedData(efield=efield, times=timegrid, radius=[0, Tokamak.a])
            sim.ds.eqsys.E_field.setBoundaryCondition()


            sim.ds.timestep.setTmax(TMAX_INIT)
            sim.ds.timestep.setNt(NT_INIT)
            do1 = sim._run()

if __name__ == '__main__':
    sys.exit(main())
