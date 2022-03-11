#!/usr/bin/env python3

import sys
import numpy as np
import json
from DREAMSimulation import DREAMSimulation

if len(sys.argv) == 2:
    with open(sys.argv[1], 'r') as fp:
        log = json.load(fp)

    nD2, nNe = log['P'][-1]

    s = DREAMSimulation(nD2=nD2, nNe=nNe, id='optim0')
    s.run(handleCrash=True)
    
    print('\nMaximal runaway current'.ljust(26) + ' = ' + f'{s.output.getMaxRECurrent()*1e-6:.3} MA'.rjust(10))
    print('Current quench time'.ljust(25) + ' = ' + f'{s.output.getCQTime()*1e3:.4} ms'.rjust(10))
    s.output.visualizeCurrents(show=True)
