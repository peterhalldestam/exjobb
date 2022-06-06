#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.constants import physical_constants
from dataclasses import dataclass

import utils

BINDING_ENERGY_DATA = 'mass.mas03'

def get_mass(neutrons, protons):

    neutron_mass = physical_constants['neutron mass energy equivalent in MeV']
    proton_mass = physical_constants['proton mass energy equivalent in MeV']
    return neutrons * neutron_mass + protons * proton_mass

def main():


    # Read the experimental data into a Pandas DataFrame.
    df = pd.read_fwf(BINDING_ENERGY_DATA, usecols=(2,3,4,9,11),
                  widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
                  skiprows=39, header=None,
                  index_col=False)
    df.columns = ('N', 'Z', 'A', 'massExcess', 'avEbind')

    # Extrapolated values are indicated by '#' in place of the decimal place, so
    # the avEbind column won't be numeric. Coerce to float and drop these entries.
    df['avEbind'] = pd.to_numeric(df['avEbind'], errors='coerce')
    df['massExcess'] = pd.to_numeric(df['massExcess'], errors='coerce')
    df = df.dropna()

    # Also convert from keV to MeV.
    df['avEbind'] /= 1000

    print(get_mass(1,0))

    fig, ax = plt.subplots(figsize=utils.FIGSIZE_1X1)



    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xticks([50, 100, 150, 200, 250])
    ax.set_yticks([2, 4, 6, 8])

    ax.set_xlabel(r'$A=Z+N$')
    ax.set_ylabel(r'$\Delta E/A\,({\rm MeV})$')



    # ax.set_xlim(-50,299)

    ax.plot((1), (0), ls="", marker=">", ms=6, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=6, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    x = 'A'
    y = 'avEbind'
    size = 10

    ax.scatter(df[x], df[y], c='k', s=1)
    print(df['avEbind'])

    deuteron = df.loc[(df['A']==2) & (df['Z']==1)]
    ax.scatter(deuteron[x], deuteron[y], c='r', s=size)

    triton = df.loc[(df['A']==3) & (df['Z']==1)]
    ax.scatter(triton[x], triton[y], c='b', s=size)

    alpha = df.loc[(df['A']==4) & (df['Z']==2)]
    ax.scatter(alpha[x], alpha[y], c='c', s=size)

    neutron = df.loc[(df['A']==1) & (df['N']==1)]


    uranium = df.loc[(df['A']==235) & (df['Z']==92)]
    ax.scatter(uranium[x], uranium[y], c='c', s=size)

    barium = df.loc[(df['A']==141) & (df['Z']==56)]
    ax.scatter(barium[x], barium[y], c='c', s=size)

    krypton = df.loc[(df['A']==92) & (df['Z']==36)]
    ax.scatter(krypton[x], krypton[y], c='c', s=size)

    print(deuteron, triton, alpha, uranium, barium, krypton)

    plt.tight_layout()
    plt.show()
    #
    # # ax.plot([62, 62], [8, 9])
    #
    # ax.text(53, 9.1, r'$\rm{Fe}$')
    #
    # # ax.legend()
    # # We don't expect the SEMF to work very well for light nuclei with small
    # # average binding energies, so display only data relevant to avEbind > 7 MeV.
    # # ax.set_ylim(-1, 10 )
    #
    #
    # plt.tight_layout()
    # plt.show()




if __name__ == '__main__':
    utils.setFigureFonts()
    sys.exit(main())
