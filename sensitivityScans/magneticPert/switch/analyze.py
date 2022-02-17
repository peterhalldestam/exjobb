#!/usr/bin/env python3

import sys, os, glob, warnings
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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
	Q_trans = np.sum(do.other.scalar.energyloss_T_cold[:,0]*dt)*R0

	if ax:
		#ax.plot(t*1e3, I_ohm[:]*1e-6, label = '$I_{\Omega}$')
		ax.semilogy(t*1e3, I_ohm[:]*1e-6, label = '$I_{\Omega}$')
		ax.plot(t*1e3, I_re[:]*1e-6, label = '$I_{RE}$')
		ax.plot(t*1e3, I_tot[:]*1e-6, label = '$I_{tot}$')

	return I_re_max, t_CQ, Q_trans


if __name__ == '__main__':

	#outputs = sorted(glob.glob('output/*'))
	outputs = sorted(glob.glob('output/*'))
	#do1 = DREAMOutput(outputs[0])
	#do2 = DREAMOutput(outputs[1])

	I_list = np.array([])
	t_CQ_list = np.array([])
	Q_trans_list = np.array([])
	dBB_list = np.array([])
	do_list = []

	for output in outputs:

		do = DREAMOutput(output)

		I, t_CQ, Q_trans = analyze(do)
		dBB = do.settings.eqsys.T_cold.transport.dBB[0]
		
		dBB_list = np.append(dBB_list, dBB)
		I_list = np.append(I_list, I)
		Q_trans_list = np.append(Q_trans_list, Q_trans)
		t_CQ_list = np.append(t_CQ_list, t_CQ)
		do_list.append(do)

	fig, ax = plt.subplots(1, len(outputs), figsize=(17,6), sharey='row')
	
	for i in range(len(outputs)):
		do = do_list[i]
		analyze(do, ax[i])
		#ax[i].set_title(f'dBB = {dBB_list[i]}')
		#ax[i].set_title(f'pstar = {outputs[i][20:-3]}')
		#ax[i].text(50, 14, '$\hat{I}_{RE}$'f'$={I_list[i]*1e-6:.3}$ MA', fontsize = 12)
		#ax[i].text(50, 13, '$t_{CQ}$'f'$={t_CQ_list[i]*1e3:.4}$ ms', fontsize = 12)
		ax[i].text(10, 1e-14, '$Q_{trans}$'f'$={Q_trans_list[i]*1e-6:.4}$ MJ', fontsize = 12)
		ax[i].set_xlabel('time [ms]')
			
		
		
	ax[0].set_ylabel('current [MA]')
	fig.legend(['$I_{\Omega}$', '$I_{RE}$', '$I_{tot}$'], fontsize = 15)
	
	plt.show()
	

