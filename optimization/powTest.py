#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optimization import powell, naive

theta = np.pi * 2/7
fun = lambda x,y: 1*(x*np.cos(theta) - y*np.sin(theta) - 2)**2 + 2*(x*np.sin(theta) + y*np.cos(theta) - 2)**2

x = np.linspace(-1, 5)
y = np.linspace(-5, 5)


P0 = np.array([0., -3.9])
Ptrack = powell(fun, P0, ftol=5e-1, verbose=True)

numbers = np.arange(1, len(Ptrack)+1)

X, Y = np.meshgrid(x, y)
Z = fun(X, Y)

plt.contour(X, Y, Z)
plt.plot(Ptrack[:,0], Ptrack[:,1], 'o')

for i in range(len(numbers)):
    plt.text(Ptrack[i,0], Ptrack[i,1]+0.1, str(numbers[i]))

plt.show()
