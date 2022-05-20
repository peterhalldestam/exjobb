#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import sim.DREAM.expDecay as sim

FIRST = [1.4e20, 4.5e18]
LAST = [1.1652767024506405e22, 6.99511456882493e16] # Powell's optimum

def runSimulation(nD, nNe):

    s = sim.ExponentialDecaySimulation()
    s.configureInput(nD2=nD, nNe=nNe)
    s.run(handleCrash=False)
    return s.output.t * 1e3, s.output.I_re * 1e-6, s.output.I_ohm * 1e-6

def main():


    utils.setFigureFonts()
    fig, ax = plt.subplots(figsize=utils.FIGSIZE_1X1, sharey=True)


    show = False
    for (nD, nNe), fmt in zip((FIRST, LAST), ('--', '-')):

        t, I_re, I_ohm = runSimulation(nD, nNe)

        ax.plot(t, I_re, fmt, c='b', lw=2, label=r'$\rm RE$' if show else None)
        ax.plot(t, I_ohm, fmt, c='r', lw=2, label=r'$\rm Ohmic$' if show else None)
        ax.plot(t, I_re + I_ohm, fmt, lw=2, c='k', label=r'$\rm total$' if show else None)

        show = True

    ax.set_xlim([t[0], t[-1]])

    ax.set_xlabel(r'$t\,({\rm ms})$')
    ax.set_ylabel(r'$I\,({\rm MA})$')

    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])

    plt.legend(title=r'$\rm Currents$')
    plt.tight_layout()
    plt.savefig('current_evolution.eps', forma='eps')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
