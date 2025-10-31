import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp
from pypulseq.convert import convert

import scipy as sp
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
import scipy.special as sps

from sequtil import SafetyLimits, LTIGradientKernels

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def first_moment(T, DT, slew_rate):
	return DT*slew_rate*(2*np.square(DT) + 3*T*DT + np.square(T))

# max_grad [mT/m], slew_rate [T/m/s]
def calc_T_DT(venc, max_grad, slew_rate, gamma, tot_T=None):
	max_grad_T = max_grad * 1e-3
	
	DM = np.pi/(gamma*venc)
	#DM = first_moment(0.005, max_grad/slew_rate, slew_rate)

	if tot_T is None:
		# T=0 gradients =>
		DT = (DM / (2 * slew_rate)) ** (1.0 / 3.0)

		if DT * slew_rate > max_grad_T:
			DT = max_grad_T / slew_rate

			root = 0.25 * np.square(DT) + DM/(slew_rate*DT)
			T = -(3/2)*DT + np.sqrt(root)
			#DM_back = first_moment(T, DT, slew_rate)
			return T, DT
		else:
			return 0.0, DT
	else:
		# For solutions see the SymPy script at vel_maths.ipynb

		L = float(tot_T)

		#DT = (1/4)*(L**2*slew_rate + math.sqrt(L*slew_rate*(-32*DM + L**3*slew_rate)))/(L*slew_rate)
		#T = (L - 4*DT) / 2

		T = (1/2)*math.sqrt(-32*DM/(L*slew_rate) + L**2)
		DT = (L - 2*T) / 4

		if slew_rate * DT > max_grad:
			raise RuntimeError("Can't calculate T,DT from total time given this slew \
					  max gradient is exceeded, increase total time, slew_rate or max_grad")
		
		if abs(L - (2*T + 4*DT)) / abs(L) > 1e-9:
			raise RuntimeError("Large difference between total time and calculated T,DT")

		return T, DT



class VelocityEncodingFactory:
	def __init__(self, ltik: LTIGradientKernels, sl: SafetyLimits, print_calc=False):
		self.system = ltik.system
		self.ltik = ltik
		self.sl = sl

		self.print_calc = print_calc

		ko = self.ltik.kernel_oversampling
	
	def ltik_corrected_upsamp_grad(self, grad_wave, channel):
		ko = self.ltik.kernel_oversampling
		GRT = self.system.grad_raster_time

		tup = np.linspace(0, grad_wave.shape[0]*GRT, grad_wave.shape[0]*ko)
		upsamp_grad_wave = interp1d(
			np.linspace(0, grad_wave.shape[0]*GRT, grad_wave.shape[0]), 
			grad_wave, 
			kind='linear'
		)(tup)
		grad_wave = np.convolve(grad_wave, self.ltik.kernels[channel], mode='full')

		return upsamp_grad_wave, tup

	def calculate_moments_and_venc(self, grad_wave, t):
		dt = t[1]-t[0]
		grad_wave = 1e-3*convert(grad_wave, from_unit='Hz/m', to_unit='mT/m')
		DM_0 = np.trapz(grad_wave, dx=dt)
		DM_1 = np.trapz(t*grad_wave, dx=dt)
		DM_2 = np.trapz(t*t*grad_wave, dx=dt)
		venc = np.pi / (self.system.gamma * DM_1)
		return DM_0, DM_1, DM_2, venc

	def get_gradients(self, velocity_vector, channels=['x', 'y', 'z']):
		
		GRT = self.system.grad_raster_time

		max_grad = self.system.max_grad * self.sl.grad_ratio
		max_slew = self.system.max_slew * self.sl.slew_ratio

		# We need to find which gradient will take the longset time, and how long that time is
		grad_lengths = [
							(0.0, 0.0) if vel is None else
							calc_T_DT(
								abs(vel), 
								convert(max_grad, from_unit='Hz/m', to_unit='mT/m'), 
								convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'), 
								self.system.gamma
							)
							for vel in velocity_vector
						]   

		max_grad_length = max([gradlen[0] + 2*gradlen[1] for gradlen in grad_lengths])
		
		L = 2 * max_grad_length
		t = np.arange(0, math.ceil(L/GRT)+1)*GRT

		grad_waves = []
		grad_properties = []

		for i in range(len(velocity_vector)):
			
			if velocity_vector[i] is None or velocity_vector[i] > 1e4:

				grad_wave = np.zeros((t.shape[0] + self.ltik.max_kernel_length,))

				if self.print_calc:
					print('Velocity is None or too high, setting to zero')
					print('Zeroth order moment: ', 0, ' First order moment: ', 0, 'Second order moment: ', 0, ' Venc: infty')

				grad_waves.append(grad_wave)
				grad_properties.append((0.0, 0.0, math.inf))
			else:

				T, DT = calc_T_DT(
							abs(velocity_vector[i]), 
							convert(max_grad, from_unit='Hz/m', to_unit='mT/m'), 
							convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'), 
							self.system.gamma, 
							L
						)
				
				#print('T: ', T, ' DT: ', DT)

				Gmax = max_slew*DT

				if velocity_vector[i] > 0:
					grad_blip = np.array([0, -Gmax, -Gmax, 0, Gmax, Gmax, 0, 0])
				else:
					grad_blip = np.array([0, Gmax, Gmax, 0, -Gmax, -Gmax, 0, 0])

				grad_wave = interp1d(
								np.array([0, DT, T+DT, T+2*DT, T+3*DT, 2*T+3*DT, 2*T+4*DT, 2*T+4*DT+GRT]),
								grad_blip,
								kind='linear'
							)(t)

				if self.print_calc:
					plt.figure()
					plt.plot(t, convert(grad_wave, from_unit='Hz/m', to_unit='mT/m'), 'r-')
					plt.title('Velocity Gradient waveforms')
					plt.xlabel('Time [s]')
					plt.ylabel('Gradient [mT/m]')
					plt.show()

				smooth_velenc = True
				if smooth_velenc:
					kernel = np.linspace(0,1,grad_wave.shape[0] // 4) - 0.5
					kernel = np.exp(-np.square(kernel)/0.05)
					kernel /= np.sum(kernel)

					#plt.figure()
					#plt.plot(kernel)
					#plt.title('Smoothing kernel')
					#plt.show()

					grad_wave = np.convolve(grad_wave, kernel, mode='full')

				grad_wave = np.concatenate([grad_wave, np.zeros((self.ltik.max_kernel_length,))])

				DM0, DM1, DM2, venc = self.calculate_moments_and_venc(*self.ltik_corrected_upsamp_grad(grad_wave, channels[i]))

				#if self.print_calc:
				print('Zeroth order moment: ', DM0, ' First order moment: ', DM1, 'Second order moment: ', DM2, ' Venc: ', venc)

				grad_waves.append(grad_wave)
				grad_properties.append((DM0, DM1, DM2, venc))

		return grad_waves, grad_properties



if __name__ == "__main__":

	#T, DT = calc_T_DT(0.001, 80, 200, 42.58e6)
	system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)

	ltik = LTIGradientKernels(system, LTIGradientKernels.kernels_from_test(system.grad_raster_time))

	vef = VelocityEncodingFactory(ltik, SafetyLimits(), print_calc=True)

	back = vef.get_gradients([0.7, None, 0.2])