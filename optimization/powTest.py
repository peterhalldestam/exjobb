#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optimization import Powell
from matplotlib.patches import Rectangle

theta = np.pi * 2/7
fun = lambda x,y: 1*(x*np.cos(theta) - y*np.sin(theta) - 2)**2 + 2*(x*np.sin(theta) + y*np.cos(theta) - 2)**2

x = np.linspace(-2, 6)
y = np.linspace(-5, 6)


P0 = np.array([0., -3.9])
lowerBound = (-1., -4.)
upperBound = (5., 5.)


powOpt = Powell(fun, P0, lb=lowerBound, ub=upperBound, verbose=True)
Ptrack = powOpt.run()

numbers = np.arange(1, len(Ptrack)+1)

X, Y = np.meshgrid(x, y)
Z = fun(X, Y)

fig, ax = plt.subplots()

ax.contour(X, Y, Z, levels = 15)
#ax.plot(Ptrack[:,0], Ptrack[:,1], 'o')

#X, Y = np.meshgrid(Ptrack[:,0], Ptrack[:,1])

X = Ptrack[:-1, 0]
Y = Ptrack[:-1, 1]

U = Ptrack[1:, 0] - Ptrack[:-1, 0]
V = Ptrack[1:, 1] - Ptrack[:-1, 1]

#U = X[1:] - X[:-1]
#V = Y[1:] - Y[:-1]

#ax.quiver(X,Y,U,V)
for i in range(len(X)):
    ax.arrow(X[i], Y[i], U[i], V[i], width=0.05, length_includes_head=True)

#for i in range(len(numbers)):
#    ax.text(Ptrack[i,0], Ptrack[i,1]+0.1, str(numbers[i]))

ax.add_patch( Rectangle((-1., -4.),
                        6, 9,
                        fc ='none', 
                        ec ='g',
                        lw = 2) )

"""                         
ax.plot(crossMTrack[1:, 0], crossMTrack[1:, 1], 'bx')
ax.plot(crossPTrack[1:, 0], crossPTrack[1:, 1], 'rx')
"""
#print(crossTrack[:,0])

plt.show()
