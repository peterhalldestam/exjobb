#!/usr/bin/env python3

import sys, os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/hannber/DREAM/py')
from DREAM.DREAMOutput import DREAMOutput

def current_quench_time(I, t):
	""" 
	If possible, calculates current quench time.
	Returns None otherwise.
	"""	
	
	tol = 0.02
	
	ind_80 = np.argmin(np.abs(I/I[0] - 0.8))
	ind_20 = np.argmin(np.abs(I/I[0] - 0.2))

	if np.abs(I[ind_80]/I[0] - 0.8) > tol:
		msg = f'\nUnable to determine time of 80% amplitude within {tol*100}% margin.'
		warnings.warn(msg)
		return None
	elif np.abs(I[ind_20]/I[0] - 0.2) > tol:
		msg = f'\nUnable to determine time of 20% amplitude within {tol*100}% margin.'
		warnings.warn(msg)
		return None

	return (t[ind_20] - t[ind_80]) / 0.6

def analyze(do, ax=None):
	""" 
	Extracts quantities current quench time and maximal runaway current.
	Plots current evolution if pyplot axis object is given
	"""
	
	t = do.grid.t
	I_ohm = do.eqsys.j_ohm.current()
	I_re = do.eqsys.j_re.current()
	I_tot = do.eqsys.I_p
	
	I_re_max = np.amax(I_re)
	t_CQ = current_quench_time(I_ohm, t)

	if ax:
		ax.plot(t*1e3, I_ohm[:]*1e-6, label = '$I_{\Omega}$')
		ax.plot(t*1e3, I_re[:]*1e-6, label = '$I_{RE}$')
		ax.plot(t*1e3, I_tot[:]*1e-6, label = '$I_{tot}$')
		
		ax.set_xlabel('time [ms]')
		ax.set_ylabel('current [MA]')
		ax.legend(fontsize = 12)

	return I_re_max, t_CQ


if __name__ == '__main__':

	outputs = sorted(glob.glob('output/*'))

	I_list = np.array([])
	t_CQ_list = np.array([])
	dBB_I_list = np.array([])
	dBB_CQ_list = np.array([])

	for output in outputs:

		do = DREAMOutput(output)

		I, t_CQ = analyze(do)
		dBB = float(output[18:-3])
		
		I_list = np.append(I_list, I)
		dBB_I_list = np.append(dBB_I_list, dBB)
		if t_CQ:
			t_CQ_list = np.append(t_CQ_list, t_CQ)
			dBB_CQ_list = np.append(dBB_CQ_list, dBB)
		else:
			msg = f'\nNeglecting dBB = {dBB} from current quench times.'
			warnings.warn(msg)


	fig, ax = plt.subplots(1, 2)

	ax[0].plot(dBB_I_list, I_list*1e-6)
	ax[0].set_xlabel('magnetic perturbation')
	ax[0].set_ylabel('maximal runaway current [MA]')
	ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

	ax[1].plot(dBB_CQ_list, t_CQ_list*1e3)
	ax[1].set_xlabel('magnetic perturbation')
	ax[1].set_ylabel('current quench time [ms]')
	ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
"""	
	fig_test, ax_test = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			analyze(DREAMOutput(outputs[i*5+j]), ax_test[i,j])
"""	
	plt.show()
	

