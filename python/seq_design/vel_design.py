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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu


class VelocityEncodingFactory:
	def __init__(self, system, LTI_kernels, kernel_oversampling=100, print_calc = False):
		self.system = system
		self.LTI_kernels = LTI_kernels
		self.kernel_oversampling = kernel_oversampling
		self.kernel_delays = np.array([kernel.shape[0] for kernel in LTI_kernels.values()]) * self.system.grad_raster_time / self.kernel_oversampling

		self.venc_t_list = np.linspace(0.2, 10.0, 100)*1e-3
		venc_list_x = np.array([self._calculate_moments(t, 'x', False)[1] for t in self.venc_t_list])
		venc_list_y = np.array([self._calculate_moments(t, 'y', False)[1] for t in self.venc_t_list])
		venc_list_z = np.array([self._calculate_moments(t, 'z', False)[1] for t in self.venc_t_list])

		venc_x_interp = PchipInterpolator(np.flip(venc_list_x), np.flip(self.venc_t_list))
		venc_y_interp = PchipInterpolator(np.flip(venc_list_y), np.flip(self.venc_t_list))
		venc_z_interp = PchipInterpolator(np.flip(venc_list_z), np.flip(self.venc_t_list))
		self.venc_interpers = {'x': venc_x_interp, 'y': venc_y_interp, 'z': venc_z_interp}

		if print_calc:
			plt.figure()
			plt.plot(venc_list_x, self.venc_t_list, 'r-')
			vencinterp = np.linspace(venc_list_x[0], venc_list_x[-1], 1000)
			plt.plot(vencinterp, venc_x_interp(vencinterp), 'b-')
			plt.title('Grad X Venc vs input_area_time')
			plt.show()

			plt.figure()
			plt.plot(venc_list_y, self.venc_t_list, 'r-')
			vencinterp = np.linspace(venc_list_y[0], venc_list_y[-1], 1000)
			plt.plot(vencinterp, venc_y_interp(vencinterp), 'b-')
			plt.title('Grad Y Venc vs input_area_time')
			plt.show()

			plt.figure()
			plt.plot(venc_list_z, self.venc_t_list, 'r-')
			vencinterp = np.linspace(venc_list_z[0], venc_list_z[-1], 1000)
			plt.plot(vencinterp, venc_z_interp(vencinterp), 'b-')
			plt.title('Grad Z Venc vs input_area_time')
			plt.show()

	def _calculate_moments(self, input_area_time, channel, print_calc=False):
		input_area = input_area_time * (system.max_grad)

		# Backwards phasing
		trap1 = pp.make_trapezoid(channel=channel, system=self.system, area=-input_area)
		temp_wave = pp.points_to_waveform(
			np.array([0.0, trap1.amplitude, trap1.amplitude, 0.0]),
			self.system.grad_raster_time, 
			np.array([0.0, trap1.rise_time, trap1.flat_time + trap1.rise_time, trap1.flat_time + trap1.rise_time + trap1.fall_time])
		)
		trap1_wave = np.zeros((1 + temp_wave.shape[0],))
		trap1_wave[1:] = temp_wave
		# Forwards phasing
		trap2 = pp.make_trapezoid(channel=channel, system=self.system, area=input_area)
		temp_wave = pp.points_to_waveform(
			np.array([0.0, trap2.amplitude, trap2.amplitude, 0.0]), 
			self.system.grad_raster_time, 
			np.array([0.0, trap2.rise_time, trap2.flat_time + trap2.rise_time, trap2.flat_time + trap2.rise_time + trap2.fall_time])
		)
		trap2_wave = np.zeros((1 + temp_wave.shape[0],))
		trap2_wave[:-1] = temp_wave


		grad = np.concatenate([trap1_wave, trap2_wave])


		# Oversample the gradient waveform for convolution with LTI kernels
		grad_oversamp = mn.oversample(grad[None,:], grad.shape[0]*self.kernel_oversampling, kind='linear')[0,:]
		t = np.linspace(0, grad_oversamp.shape[0]*self.system.grad_raster_time / self.kernel_oversampling, grad_oversamp.shape[0])

		# Actual waveform
		grad_corr = np.convolve(grad_oversamp, self.LTI_kernels[channel], mode='full')
		t_corr = np.linspace(0, (t[1]-t[0])*grad_corr.shape[0], grad_corr.shape[0])

		if print_calc:
			plt.figure()
			plt.plot(t, grad_oversamp, 'r-')
			plt.plot(t_corr, grad_corr, 'b-')
			plt.title('Velocity Gradient waveforms')
			plt.show()

		grad_oversamp = convert(grad_oversamp, from_unit='Hz/m', to_unit='mT/m')
		grad_corr = convert(grad_corr, from_unit='Hz/m', to_unit='mT/m')

		dt = t[1] - t[0]
		DM_0_perf = np.trapz(grad_oversamp, dx=dt)
		DM_0_corr = np.trapz(grad_corr, dx=dt)

		DM_1_perf = np.trapz(t*grad_oversamp, dx=dt)
		DM_1_corr = np.trapz(t_corr*grad_corr, dx=dt)

		# Corrected Venc
		#venc_perf = 1e3*np.pi / (system.gamma * DM_1_perf)
		venc_corr = 1e3*np.pi / (system.gamma * DM_1_corr)

		venc_per_T = venc_corr / input_area_time

		if print_calc:
			print('Zeroth order moment: perf: ', DM_0_perf, ' corr: ', DM_0_corr, ' ratio: ', DM_0_corr / DM_0_perf)
			print('First order moment: perf: ', DM_1_perf, ' corr: ', DM_1_corr, ' ratio: ', DM_1_corr / DM_1_perf)
			print('Venc is: ', venc_corr, ' m/s')

		return [trap1, trap2, self.LTI_delay], float(venc_corr)

	def get_velocity_encoding_gradients(self, venc_vector, print_calc=False):
		grad_out = []
		durations = []
		venc_vector_out = []
		channels = ['x', 'y', 'z']
		for i in range(3):
			channel = channels[i]
			if venc_vector[i] is None:
				grad_out.append(None)
			else:
				wanted_venc = venc_vector[i]
				ret = self._calculate_moments(self.venc_interpers[channel](wanted_venc), 'x', print_calc)
				grad_out.append(ret[0])
				venc_vector_out.append(ret[1])

		return grad_out, venc_vector_out




def test_kernel(system):
	kernel_oversampling = 100
	k1 = 8.0
	k2 = 1.5
	Dk = 6.0 / system.grad_raster_time

	n_grad_raster_times = 3
	t = np.linspace(0, n_grad_raster_times*system.grad_raster_time, n_grad_raster_times*kernel_oversampling)
	kernel = (Dk*t)**(k1) * np.exp(-k2*Dk*t)
	kernel /= np.sum(kernel)

	return kernel, kernel_oversampling



if __name__ == '__main__':

	system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)
	kernel, kernel_oversampling = test_kernel(system)

	vef = VelocityEncodingFactory(system, {'x': kernel, 'y': kernel, 'z': kernel}, kernel_oversampling, print_calc=True)

	ret = vef.get_velocity_encoding_gradients([0.5, 0.25, 0.1])
	print(ret[1])