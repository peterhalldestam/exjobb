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


n0 = 1e19
cases = ((1., 0.), (.7, .3), (.3, .7), (0., 1.))

def main():

    for i, case in enumerate(cases):

        print(case)

        # run simulation
        ds = getBaseline(nD=case[0]*n0, nT=case[1]*n0)
        ds.fromOutput('init_out.h5')
        runiface(ds, f'outputs/out{i}.h5')
        os.remove('init_out.h5')

if __name__ == '__main__':
    sys.exit(main())
