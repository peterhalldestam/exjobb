#!/usr/bin/env python3

import sys, os, glob
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import json

LOG_DIRECTORY = 'logsNew'

files = sorted(glob.glob(LOG_DIRECTORY+'/log_dBB*'), key=lambda x: float(x[len(LOG_DIRECTORY)+8:-5]))


P_list, fun_list, dBB_list = [], [], []

for file in files:
    with open(file, 'r') as fp:
        log = json.load(fp)

    P_list.append(log['P'][-1])
    fun_list.append(log['fun'][-1])
    dBB_list.append(float(file[len(LOG_DIRECTORY)+8:-5]))
    #plt.plot(log['fun'])
    #plt.title(dBB_list[-1])
    #plt.show()

    #print(file[12:-5])

P_list = np.array(P_list)

fig, ax = plt.subplots()
ax.scatter(P_list[:,0], P_list[:,1], c=np.log10(fun_list), cmap='plasma')

for i, P in enumerate(P_list):
#    ax.annotate(i+1, P, xytext=P+(0.7e20, 0.7e17), label=f'dBB = {dBB_list[i]*100}%')
    ax.annotate(f'{dBB_list[i]*100}%', P, xytext=P+(0.4e20, 0.4e17))


plt.show()
