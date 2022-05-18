#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

plt.rcParams['text.usetex'] = True

SMALL_SIZE = 19
MEDIUM_SIZE = 22
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('xtick', direction = 'out')    # fontsize of the tick labels

from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes.opt1 import blackBoxFunction
import sklearn.gaussian_process as gp

import emcee
import corner

np.random.seed(3)

FILE_PATH = 'data/new_log_4D_dBB50e-4.json'
SAVE_PATH = 'MCMCSamples/4D_mcmc_20.h5'

FIG_NAME = None #'cornerPlot.pdf'

NBURNIN = 1000
NSTEPS = 20_000

def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        print(f'Running MCMC sampling, saving to \'{SAVE_PATH}\'')
        samples, autoCorr = run()
    else:
        print(f'Loading MCMC samples from \'{SAVE_PATH}\'')
        reader = emcee.backends.HDFBackend(SAVE_PATH)
        samples = reader.get_chain(discard=NBURNIN, flat=True)
        autoCorr = np.mean(reader.get_autocorr_time())
        
    print("Mean autocorrelation time: {0:.3f} steps".format(autoCorr))
        
    labels = [r'$\bar{n}_{\rm D}$', r'$ \bar{n}_{\rm Ne}$', r'$c_{\rm D}$', r'$c_{\rm Ne}$']
    fig = corner.corner(samples, plot_density=False, bins=50, hist_bin_factor=2, 
                  smooth=1., smooth1d=1.5, fill_contours=True, labels=labels, fig=None,
                  contourf_kwargs={'colors': None, 'cmap': cc.cm.diverging_bwr_40_95_c42_r, 'levels': 20})

    axes = fig.axes
    for i, ax in enumerate(axes):
    
        if i == 4:
            yticks = [16, 18, 20]
            ax.set_yticks(yticks)
            ylabels = [r'$10^{16}$', r'$10^{18}$', r'$10^{20}$']
            ax.set_yticklabels(ylabels, rotation = 0)
        elif i%4==0 and i > 0:
            ylabels = [rf'${int(tick)}$' for tick in ax.get_yticks()]
            ax.set_yticklabels(ylabels, rotation = 0)
            
        if i == 12:
            xticks = [18, 20, 22]
            ax.set_xticks(xticks)
            xlabels = [r'$10^{18}$', r'$10^{20}$', r'$10^{22}$']
            
        elif i == 13:
            xticks = [16, 18, 20]
            ax.set_xticks(xticks)
            xlabels = [r'$10^{16}$', r'$10^{18}$', r'$10^{20}$']
        elif i > 13:
            xlabels = [rf'${int(tick)}$' for tick in ax.get_xticks()]  
            
        if i >= 12:
            ax.set_xticklabels(xlabels, rotation = 0)
            x0, y0 = ax.xaxis.label.get_position()
            ax.xaxis.set_label_coords(x0, y0+0.07)
            
    fig.set_figheight(10)
    fig.set_figwidth(10)

    if FIG_NAME:
        plt.savefig(FIG_NAME)
    plt.show()

def run():
    bounds = {'log_nD': (17, 22), 'log_nNe': (15, 21), 'cD2': (-12, 12), 'cNe': (-12, 12)}

    opt = BayesianOptimization(
        f=blackBoxFunction,
        pbounds=bounds,
        random_state=1
    )

    load_logs(opt, logs=FILE_PATH)

    input = np.array([[res['params']['log_nD'], res['params']['log_nNe'], res['params']['cD2'], res['params']['cNe']] for res in opt.res])
    output = np.log(-np.array([res["target"] for res in opt.res]))

    reg = gp.GaussianProcessRegressor(
        kernel=gp.kernels.Matern(length_scale=[1., 1., 1., 1.], nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=40,
        random_state=3,
    )
    reg.fit(input, output)

    def log_posterior(params):
        for param, bound in zip(params, bounds.values()):
            if not bound[0] <= param <= bound[1]:
                return -np.inf
                
        mu = reg.predict(np.array([params]))

        return -mu[0]*1.8

    ndim, nwalkers = 4, 4*8

    backend = emcee.backends.HDFBackend(SAVE_PATH)
    backend.reset(nwalkers, ndim)

    p0 = np.array([20, 19.5, 0., 0.]) + np.random.randn(nwalkers, ndim)*0.3
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend)
    state = sampler.run_mcmc(p0, NBURNIN)
    sampler.reset()

    sampler.run_mcmc(state, NSTEPS)
    samples = sampler.get_chain(flat=True)  
    
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    
    return samples, np.mean(sampler.get_autocorr_time())

if __name__ == '__main__':
    sys.exit(main())
