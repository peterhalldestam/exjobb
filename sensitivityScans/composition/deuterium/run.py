#!/usr/bin/env python3

import sys, os
import numpy as np

try:
    import DREAM
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/pethalld/DREAM/py')
    sys.path.append('/home/peterhalldestam/DREAM/py')
    sys.path.append('/home/hannber/DREAM/py')
    import DREAM

from DREAM import runiface

sys.path.append('../../../')
from generate import getBaseline

DEUTERIUM_DENSITIES = (1e19, 1e20, 1e21)

def main():

    for n in DEUTERIUM_DENSITIES:

        # run simulation
        ds = getBaseline(n=n)
        ds.fromOutput('init_out.h5')
        runiface(ds, f'outputs/out{n:3.3}.h5')
        os.remove('init_out.h5')

if __name__ == '__main__':
    sys.exit(main())
