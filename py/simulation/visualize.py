#!/usr/bin/env python3
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
import utils
from DREAM import DREAMOutput
from DREAMSimulation import DREAMSimulation

OUTPUT_DIR = 'outputs/'

def main():

    paths = [OUTPUT_DIR + fp for fp in os.listdir(OUTPUT_DIR)]

    if not paths:
        s = DREAMSimulation()
        s.run(handleCrash=False)
        out = s.output
    else:
        dos = []
        for fp in sorted(paths):
            if fp.endswith('.h5'):
                print(fp)
                dos.append(DREAMOutput(fp))
        out = DREAMSimulation.Output(*dos)

    out.visualizeCurrents()
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
