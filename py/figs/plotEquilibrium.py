#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

import utils

NPSI = 8   # number of flux surfaces to show

# resolution
NR = 1000
NTHETA = 100

# ITER parameters
MINOR_RADIUS    = 2.
WALL_RADIUS     = 2.15
MAJOR_RADIUS    = 6

def getPolynom(a0, a1, a2, a3, a4, r):
    return np.polyval(np.flip([a0, a1, a2, a3, a4]), r)

def getShapeITER(r):
    """ Same used in Istvan's paper. """
    kappa = getPolynom(1.5, 0, 0, 0, .02, r)
    delta = getPolynom(0, 0, .035, 0, .017, r)
    Delta = getPolynom(0, 0, -.00658678, -.00634124, 0, r)
    return kappa, delta, Delta

def getPosition(r, theta):
    """ Return (R, z)-coordinates for given flux surface label r. """
    kappa, delta, Delta = getShapeITER(r)
    R = Delta + r * np.cos(theta + delta * np.sin(theta)) + MAJOR_RADIUS
    z = r * kappa * np.sin(theta)
    return R, z


def main():

    utils.setFigureFonts()
    #
    # fig, ax = plt.subplots()
    # r = np.linspace(0, WALL_RADIUS)
    # # kappa, delta, Delta = getShapeITER(r)
    # psi = getPolynom(0, 0, .794102, -0.117139, 0, r)
    # ax.plot(r, psi)
    # ax.plot(r, r)
    # # # ax.plot(r, kappa/1.5, label='kappa')
    # # # ax.plot(r, delta, label='delta')
    # # # ax.plot(r, Delta/MINOR_RADIUS, label='Delta')
    # # ax.legend()
    # sys.exit(plt.show())



    fig, ax = plt.subplots(figsize=utils.FIGSIZE_2X2)
    ax.set_aspect('equal')

    # create list of points
    cycle = np.linspace(0, 2 * np.pi, NTHETA)
    r = np.linspace(0, MINOR_RADIUS, NR).repeat(NTHETA)
    theta = np.ravel([cycle for _ in range(NR)])

    # parametrization
    R, z = getPosition(r, theta)

    # contour plot of poloidal magnetic flux
    psi = getPolynom(0, 0, .794102, -0.117139, 0, r)
    cntr = ax.tricontour(R, z, psi, levels=NPSI, colors='grey', linewidths=1)#, cmap=cc.cm.diverging_bwr_40_95_c42)

    # indicate plasma edge
    R_edge, z_edge = getPosition(MINOR_RADIUS, cycle)
    ax.plot(R_edge, z_edge, 'r', lw=2)

    # indicate tokamak wall
    R_wall, z_wall = getPosition(WALL_RADIUS, cycle)
    ax.plot(R_wall, z_wall, 'k--', lw=2)


    # R axis
    # ax.arrow(MAJOR_RADIUS - 1.5 * WALL_RADIUS, 0, 2.8 * WALL_RADIUS, 0, color='k', head_width=.1, zorder=2)
    # ax.text(MAJOR_RADIUS + WALL_RADIUS + 1, -.15, r'$R$')



    # major radius
    # ax.plot([MAJOR_RADIUS, MAJOR_RADIUS], [-.06, .06], 'k')
    # ax.text(MAJOR_RADIUS - .2, -.45, r'$R_0$')

    # z axis
    # ax.arrow(MAJOR_RADIUS - 1.5 * WALL_RADIUS, 1, 0, WALL_RADIUS, color='k', head_width=.1, zorder=2)
    # ax.text(-.1, WALL_RADIUS + .25, r'$z$')

    # indicate random flux surface
    # R_ref, z_ref = getPosition(MINOR_RADIUS * .5, cycle)
    # ax.plot(R_ref, z_ref, 'k', lw=2)


    r_ref = MINOR_RADIUS * 0.785
    R_ref, z_ref = getPosition(r_ref, cycle)
    ax.plot(R_ref, z_ref, 'k', lw=2)

    # arrow for r
    x, y = getPosition([0, r_ref], [0, 0])
    ax.arrow(x[0], y[0], x[1]-x[0]-.15, 0, color='k', head_width=.1, zorder=2)
    ax.text(MAJOR_RADIUS + .4 * r_ref, -.35, r'$r$')

    # line for theta
    theta_ref =  np.pi/3
    x, y = getPosition([0, r_ref], [0, theta_ref])
    ax.plot(x, y, 'k', zorder=2)

    # mark point flux surface
    ax.scatter(x[-1], y[-1], s=20, c='k', zorder=2)
    # ax.text(x[-1] - .5, y[-1] - .3, r'$\psi$')

    # mark phi
    x, y = MAJOR_RADIUS + 2, MINOR_RADIUS + 1
    ax.scatter(x, y, c='k', s=60)
    ax.scatter(x, y, c='w', s=40)
    ax.scatter(x, y, c='k', s=10)
    ax.text(x-.1, y+.25, r'$\varphi$')


    delta_ref = getPolynom(0, 0, .035, 0, .017, r_ref)
    theta_arc = np.linspace(0, theta_ref, 20)
    x_arc = .15 * MINOR_RADIUS * np.cos(theta_arc + delta_ref * np.sin(theta_arc)) + MAJOR_RADIUS - .03
    y_arc = .15 * MINOR_RADIUS * np.sin(theta_arc)
    ax.plot(x_arc, y_arc, 'k')
    ax.text(MAJOR_RADIUS + .25, .25, r'$\theta$')


    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
