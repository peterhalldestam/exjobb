#!/usr/bin/env python3
import sys, os
import logging
import numpy as np
import matplotlib.pyplot as plt

import opt.objective as obj

PLOT_OBJECTIVE = True   # if false, I_re is plotted
SHOW_POINTS = False
LOG_PATH = 'data/large_opaque_neon_deuterium.log'
LEVELS= [0., 15., 30., 45., 60., 75., 90., 105., 120., 135., 150., 165., 180.,
         195., 210., 225., 240., 255., 270., 285., 300., 315., 330., 345.]


def sigmoid(x, x0=0, k=1):
    """
    Hyperbolic tangent sigmoid function.
    """
    return 1/2 + 1/2*np.tanh(k*(x-x0))

def baseObjective(I_re, I_ohm, tCQ):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM.
    """

    obj1 = I_re / obj.CRITICAL_RE_CURRENT
    obj2 = I_ohm / obj.CRITICAL_OHMIC_CURRENT
    obj3 = sigmoid(-tCQ, -obj.CQ_TIME_MIN, obj.SLOPE_LEFT)
    obj4 = sigmoid(tCQ, obj.CQ_TIME_MAX, obj.SLOPE_RIGHT)

    return obj1 + obj2 + obj.CQ_WEIGHT * (obj3 + obj4)

def main():

    nD, nNe, tCQ, I_re, I_ohm = [], [], [], [], []
    nD_, nNe_, tCQ_ = [], [], []
    target = []

    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f'The log file {LOG_PATH} does not exist!')

    # Set up data log
    with open(LOG_PATH) as fp:
        fp.readline()
        while line := fp.readline():
            data = line.rstrip().replace(',', ' ').split()[-5:]

            nNe.append(float(data[0]))
            nD.append(float(data[1]))
            tCQ.append(np.inf if data[2]=='inf' else float(data[2]))
            I_re.append(float(data[3]))
            I_ohm.append(float(data[4]))
            target.append(baseObjective(I_re[-1], I_ohm[-1], tCQ[-1]))

            if not data[2] == 'inf':
                nNe_.append(float(data[0]))
                nD_.append(float(data[1]))
                tCQ_.append(float(data[2]))


    if len(nD) == 0:
        raise EOFError(f'No data in {LOG_PATH}')

    #print(np.array(nD).reshape((20,20)))
    #print(nNe)
    #print(np.array(nNe).reshape((20,20)))
    #print(np.array(I_re).reshape((20,20)))

    fig, ax = plt.subplots()

    if PLOT_OBJECTIVE:

        cntr = ax.tricontourf(nD, nNe, target, levels=LEVELS, cmap="RdBu_r")
        fig.colorbar(cntr, ax=ax)

    else:

        # Current plot
        ax.tricontour(nD, nNe, I_re, levels=14, linewidths=0.5, colors='k')
        cntr2 = ax.tricontourf(nD, nNe, I_re, levels=14, cmap="RdBu_r")
    #    cntr2 = ax2.contourf(np.array(nD).reshape((20,20)), np.array(nNe).reshape((20,20)), np.log10(np.array(I_re).reshape((20,20))), levels=14, cmap="RdBu_r")
    #    ax2.contour(np.array(nD).reshape((20,20)), np.array(nNe).reshape((20,20)), np.log10(np.array(I_re).reshape((20,20))), levels=14, linewidths=0.5, colors='k')
        fig.colorbar(cntr2, ax=ax)

        # Current quench time plot
        ax.tricontour(nD_, nNe_, tCQ_, levels=[50e-3, 150e-3], linewidths=2, linestyles=['dashed', 'dotted'])


    if SHOW_POINTS:
        for t, n1, n2 in zip(tCQ, nD, nNe):
            if np.isinf(t):
                ax.plot(n1, n2, 'ro', ms=3)
            else:
                ax.plot(n1, n2, 'ko', ms=2, alpha=.5)



    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
                # sys.exit()

    return 0

if __name__ == '__main__':
    sys.exit(main())
