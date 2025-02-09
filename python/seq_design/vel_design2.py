import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp
from pypulseq.convert import convert

import scipy as sp
from scipy.interpolate import CubicSpline, PchipInterpolator
import scipy.special as sps

from sequtil import SafetyLimits, LTIGradientKernels

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def first_moment(T, DT, slew_rate):
	return DT*slew_rate*(2*np.square(DT) + 3*T*DT + np.square(T))

# max_grad [mT/m], slew_rate [T/m/s]
def calc_T_DT(venc, max_grad, slew_rate, gamma, tot_T=None):
	max_grad *= 1e-3
	
	DM = np.pi/(gamma*venc)
	#DM = first_moment(0.005, max_grad/slew_rate, slew_rate)

	if tot_T is None:
		# T=0 gradients =>
		DT = (DM / (2 * slew_rate)) ** (1.0 / 3.0)

		if DT * slew_rate > max_grad:
			DT = max_grad / slew_rate

			root = 0.25 * np.square(DT) + DM/(slew_rate*DT)
			T = -(3/2)*DT + np.sqrt(root)

			DM_back = first_moment(T, DT, slew_rate)

			return T, DT
		else:
			return 0.0, DT
	else:
		# For solutions see the SymPy script at vel_maths.ipynb

		L = float(tot_T)

		DT = (1/4)*(L**2*slew_rate - math.sqrt(L*slew_rate*(-32*DM + L**3*slew_rate)))/(L*slew_rate)

		T = (1/2)*math.sqrt(-32*DM/(L*slew_rate) + L**2)

		if slew_rate * DT > max_grad:
			raise RuntimeError("Can't calculate T,DT from total time given this slew \
					  max gradient is exceeded, increase total time, slew_rate or max_grad")
		
		if abs(L - (2*T + 4*DT)) > 1e-9:
			raise RuntimeError("Large difference between total time and calculated T,DT")

		return T, DT



class VelocityEncodingFactory:
	def __init__(self, ltik: LTIGradientKernels, sl: SafetyLimits, print_calc=False):
		self.system = ltik.system
		self.ltik = ltik
		self.sl = sl

		ko = self.ltik.kernel_oversampling
		self.longest_kernel = max([math.ceil(kernel.shape[0]/ko) for kernel in ltik.kernels.values()])
	
	def get_gradients(self, velocity_vector):
		
		max_grad = self.system.max_grad * self.sl.grad_ratio
		max_slew = self.system.max_slew * self.sl.slew_ratio
		
		# We need to find which gradient will take the longset time, and how long that time is
		grad_lengths = [calc_T_DT(vel, max_grad, max_slew, self.system.gamma) for vel in velocity_vector]   

		max_grad_length = max([gradlen[0] + 2*gradlen[1] for gradlen in grad_lengths])
		
		L = 2 * max_grad_length

		channels = ['x', 'y', 'z']
		for i in range(len(velocity_vector)):
			
			T, DT = calc_T_DT(velocity_vector[i], max_grad, max_slew, self.system.gamma, L)
			print('T: ', T, ' DT: ', DT)

			Gmax = max_slew*DT

			grad_wave = pp.points_to_waveform(
				np.array([0.0, 	0.0, 								-Gmax, 		-Gmax, 		0.0, 		Gmax, 		Gmax, 		0.0, 		0.0]),	
				self.system.grad_raster_time,
				np.array([0.0, 	self.system.grad_raster_time, 		DT, 		T+DT, 		T+2*DT, 	T+3*DT, 	2*T+3*DT, 	2*T+4*DT, 	2*T+4*DT + self.system.grad_raster_time])
			)
			t = np.linspace(0, 2*T+4*DT + self.system.grad_raster_time, grad_wave.shape[0])

			plt.figure()
			plt.plot(t, grad_wave, 'r-')
			plt.title('Velocity Gradient waveforms')
			plt.show()

			grad_wave = np.concatenate([grad_wave, np.zeros((self.longest_kernel,))])

			plt.figure()
			plt.plot(t, grad_wave, 'r-')
			plt.title('Velocity Gradient waveforms')
			plt.show()

		

			



if __name__ == "__main__":

	#T, DT = calc_T_DT(0.001, 80, 200, 42.58e6)
	system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)

	ltik = LTIGradientKernels(system).init_from_test()

	vef = VelocityEncodingFactory(ltik, SafetyLimits())

	back = vef.get_gradients([0.01, 0.01, 0.01])