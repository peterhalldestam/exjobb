import json
import numpy as np
import matplotlib.pyplot as plt



from bayes_opt import BayesianOptimization



from bayes_opt.util import load_logs

from test import blackBoxFunction

opt = BayesianOptimization(
    f=blackBoxFunction,
    pbounds={'nD': (1e19, 1.5e22), 'nNe': (1e15, 1e19)}
)




# def posterior(optimizer, x_obs, y_obs, grid):
#
#
#
#     optimizer._gp.fit(x_obs, y_obs)
#
#     mu, sigma = optimizer._gp.predict(grid, return_std=True)
#     return mu, sigma
#
# def plot_gp(optimizer, x, y, ax=None):
#
#     if ax is None:
#         ax = plt.axes()
#
#     fig = plt.figure(figsize=(16, 10))
#     steps = len(optimizer.space)
#     fig.suptitle(
#         'Gaussian Process after {} Steps'.format(steps),
#         fontdict={'size':30}
#     )
#
#
#     X = np.array([[res["params"]["nD", res["params"]["nNe"]] for res in optimizer.res])
#
#     y = np.array([res["target"] for res in optimizer.res]) * (-1)
#
#     # Input space
#     x1 = np.linspace(X[:,0].min(), X[:,0].max()) #p
#     x2 = np.linspace(X[:,1].min(), X[:,1].max()) #q
#     x = (np.array([x1, x2])).T
#
#     mu, sigma = posterior(optimizer, x, y, xGRID)
#     ax.plot(x, y, linewidth=3, label='Target')
#     ax.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
#     ax.plot(x, mu, '--', color='k', label='Prediction')
#
#     # axis.fill(np.concatenate([x, x[::-1]]),
#     #           np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
#     #     alpha=.6, fc='c', ec='None', label='95% confidence interval')
#
#
#
#

load_logs(opt, logs='50iterations.json')
X = np.array([[res["params"]["nD"], res["params"]["nNe"]] for res in opt.res])
y = np.array([res["target"] for res in opt.res]) * (-1)

print(X[np.argmin(y)])
