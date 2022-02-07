#!/usr/bin/env python3

import sys, os, glob, warnings
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

sys.path.append('/home/hannber/DREAM/py')
from DREAM.DREAMOutput import DREAMOutput

def current_quench_time(I, t):
	""" 
	If possible, calculates current quench time.
	Returns None otherwise.
	"""
	t0_80 = t[np.argmin(np.abs(I/I[0] - 0.8))]
	t0_20 = t[np.argmin(np.abs(I/I[0] - 0.2))]
	
	t_80 = fsolve(lambda x: np.interp(x, t, I)/I[0]-0.8, x0 = t0_80)
	t_20 = fsolve(lambda x: np.interp(x, t, I)/I[0]-0.2, x0 = t0_20)
	
	return (t_20 - t_80) / 0.6


def analyze(do, ax=None):
	""" 
	Extracts quantities current quench time and maximal runaway current.
	Plots current evolution if pyplot axis object is given
	"""
	
	t = do.grid.t
	dt = t[1:] - t[:-1]
	I_ohm = do.eqsys.j_ohm.current()
	I_re = do.eqsys.j_re.current()
	I_tot = do.eqsys.I_p
	R0 = do.settings.radialgrid.R0
	
	I_re_max = np.amax(I_re)
	t_CQ = current_quench_time(I_ohm, t)
	Q_trans = np.sum(do.other.scalar.energyloss_T_cold[:]*dt)*R0

	if ax:
		ax.plot(t*1e3, I_ohm[:]*1e-6, label = '$I_{\Omega}$')
		ax.plot(t*1e3, I_re[:]*1e-6, label = '$I_{RE}$')
		ax.plot(t*1e3, I_tot[:]*1e-6, label = '$I_{tot}$')
		
	return I_re_max, t_CQ, Q_trans


if __name__ == '__main__':

	outputs = sorted(glob.glob('output/*'), key = lambda x: float(x[20:23])*100 + float(x[29:32]))

	I_list = np.array([])
	t_CQ_list = np.array([])
	Q_trans_list = np.array([])
	alpha_list = np.array([])
	beta_list = np.array([])
	do_list = []

	for output in outputs:

		do = DREAMOutput(output)

		I, t_CQ, Q_trans = analyze(do)
		
		alpha_list = np.append(alpha_list, float(output[20:23]))
		beta_list = np.append(beta_list, float(output[29:32]))
		I_list = np.append(I_list, I)
		t_CQ_list = np.append(t_CQ_list, t_CQ)
		Q_trans_list = np.append(Q_trans_list, Q_trans)
		do_list.append(do)
		

	fig, ax = plt.subplots(3, 3, figsize=(11, 10), sharey='row', sharex='col')
	for i in range(3):
		ax[i, 0].set_ylabel('current [MA]')
		ax[-1, i].set_xlabel('time [ms]')
		for j in range(3):
			analyze(DREAMOutput(outputs[i*3+j]), ax[i,j])
			ax[i, j].text(50, 11, r'$\tilde{I}_{RE}$'f'$={I_list[i*3+j]*1e-6:.3}$ MA', fontsize = 12)
			ax[i, j].text(50, 10, '$t_{CQ}$'f'$={t_CQ_list[i*3+j]*1e3:.4}$ ms', fontsize = 12)
			ax[i, j].text(50, 9, '$Q_{trans}$'f'$={Q_trans_list[i*3+j]*1e-9:.4}$ GJ', fontsize = 12)
			ax[i, j].set_title(fr'$\alpha = {alpha_list[i*3+j]}$  $\beta = {beta_list[i*3+j]}$')
			
	fig.legend(['$I_{\Omega}$', '$I_{RE}$', '$I_{tot}$'], fontsize = 15)

	fig.suptitle(r'$j \propto (1 - \alpha (\frac{r}{a})^2)^\beta$     '+r'$I_p = $'+'15 MA', fontsize=15)
	plt.show()
	

