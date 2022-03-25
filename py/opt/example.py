"""
This script showcases how a tokamak distribution simulation is run using an
instance of the DREAMSimulation class, from which relevant output data is
obtained. Here we vary the injected neon density, assumed to be homogeneously
distributed in space. / Peter
"""
import sys, os, pathlib
import numpy as np


sys.path.append(os.path.abspath('..'))
from simulation.DREAMSimulation import DREAMSimulation, OUTPUT_DIR
from DREAM import DREAMOutput

NSIM = 3
NEON_DENSITIES = [x * 1e18 for x in range(1, NSIM+1)]

def main():

    outputs = []

    # make sure the output directory exists
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # get output filenames if there's any
    paths = [f'{OUTPUT_DIR}{fp}' for fp in os.listdir(OUTPUT_DIR)]

    # without previous ouput data, run the simulations
    if not paths:
        for i, n in enumerate(NEON_DENSITIES):

            print(f'Running simulation #{i+1}/{NSIM}')

            ###### IMPORTANT PART ######
            # Create simulation with an ID and set the injected neon density
            s = DREAMSimulation(id=f'out{i}', verbose=False, nNe=n)    # produces 3 files per simulation

            # Run simulation
            s.run(handleCrash=True)

            # Access relevant output data
            outputs.append(s.output)
            ############################


    # with previous output data, load the files
    else:
        assert (len(paths) == 3 * NSIM), f'remove all files in {OUTPUT_DIR} and rerun'
        for i, n in enumerate(NEON_DENSITIES):
            dos = [DREAMOutput(f'{OUTPUT_DIR}out{i}_{j}.h5') for j in (1, 2, 3)]
            output = DREAMSimulation.Output(*dos)
            outputs.append(output)

    for out, n in zip(outputs, NEON_DENSITIES):
        print(f'nNe = {n}\t tCQ = {out.tCQ}\t max(I_re) = {out.getMaxRECurrent()}')
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
