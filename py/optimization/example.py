"""
This script showcases how a tokamak distribution simulation is run using an
instance of the DREAMSimulation class, from which relevant output data is
obtained. Here we vary the injected neon density, assumed to be homogeneously
distributed in space. / Peter
"""
import sys, os, pathlib
import numpy as np

sys.path.append(os.path.abspath('..'))
from simulation.DREAMSimulation import DREAMSimulation
from simulation.simulation import Parameter

NSIM = 3
NEON_DENSITIES = [x * 1e20 for x in range(1, NSIM)]
OUTPUT_DIR = 'example_outputs/'

def main():

    outputs = []

    # make sure the output directory exists
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # get output filenames if there's any
    paths = [f'{OUTPUT_DIR}{fp}' for fp in os.listdir(OUTPUT_DIR)]

    # without previous ouput data, run the simulations
    if not paths:
        for i, n in enumerate(NEON_DENSITIES):

            # this way of entering parameters is temporary....
            tmp = Parameter(0., np.inf, n)

            ###### IMPORTANT PART ######
            # Create simulation with an ID and set the injected neon density
            s = DREAMSimulation(id=f'{OUTPUT_DIR}out{i}', nNe=tmp)    # produces 3 files per simulation

            # Run simulation
            s.run(handleCrash=False)

            # Access relevant output data
            outputs.append(s.output)
            ############################


    # with previous output data, load the files
    else:
        assert (len(paths) == 3 * NSIM), f'remove all files in {OUTPUT_DIR} and rerun'
        for i, n in enumerate(NEON_DENSITIES):
            dos = [f'{OUTPUT_DIR}out{i}_{j}.h5' for j in (1, 2, 3)]
            output = DREAMSimulation.Output(*dos)
            outputs.append(output)

    for output, n in zip(outputs, NEON_DENSITIES):
        print(f'nNe = {n}\t tCQ = {out.tCQ}\t max(I_re) = {out.getCQTime()}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
