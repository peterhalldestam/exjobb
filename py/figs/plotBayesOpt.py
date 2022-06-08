from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import utils

def target(x):
    return -np.sin(x)
    # return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=utils.FIGSIZE_1X1)
    steps = len(optimizer.space)
    # fig.suptitle(
    #     'Gaussian Process and Utility Function After {} Steps'.format(steps),
    #     fontdict={'size':30}
    # )

    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    # axis = plt.subplot(gs[0])
    # acq = plt.subplot(gs[1])

    axis = plt.subplot()

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    if len(y_obs)==0:
        mu = np.zeros(x.shape)
        sigma = np.ones(x.shape)
    else:
        mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    axis.fill(np.concatenate([x, x[::-1]]),
        -np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        fc='c', ec='None', label=r'${95\,\%\:\rm confidence}$', c='pink')
    axis.plot(x, -y, linewidth=3, label=r'$f(x)=\sin{x}$')
    axis.plot(x_obs.flatten(), -y_obs, 'D', markersize=8, label=r'${\rm Dataset\:}\mathcal{D}_n$', color='r')
    axis.plot(x, -mu, '--', color='k', label=r'${\rm Prediction\:}\mu_n(x)$')


    axis.set_xlim((-2, 10))
    axis.set_ylim((-3, 3))
    axis.set_ylabel(r'$y$', fontdict={'size':20})
    axis.set_xlabel(r'$x$', fontdict={'size':20})

    # utility_function = UtilityFunction(kind="ei", kappa=5, xi=0)
    # utility = utility_function.utility(x, optimizer._gp, 0)
    # acq.plot(x, utility, label='Utility Function', color='purple')
    # acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
    #          label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    # acq.set_xlim((-2, 10))
    # acq.set_ylim((0, np.max(utility) + 0.5))
    # acq.set_ylabel('Utility', fontdict={'size':20})
    # acq.set_xlabel('x', fontdict={'size':20})
    plt.tight_layout()
    axis.legend(loc='upper right')#, bbox_to_anchor=(1.1, 1), borderaxespad=0.)



optimizer = BayesianOptimization(
    target,
    {'x': (-2, 10)},
    random_state=420
)
# optimizer.set_gp_params(normalize_y=True)
optimizer.maximize(init_points=1, n_iter=0, kappa=5, acq='ei')




normalize_y=True,


x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)


utils.setFigureFonts()

plt.rc('legend', fontsize=15)

plot_gp(optimizer, x, y)

plt.show()
