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

from sequtil import SafetyLimits, LTIGradientKernels, ImageProperties

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

class SeiffertSpiral:
	def __init__(self, ltik: LTIGradientKernels, safety: SafetyLimits, imgprop: ImageProperties ,print_calc=False):
		self.system = ltik.system
		self.print_calc = print_calc
		self.ltik = ltik
		self.safety = safety
		self.imgprop = imgprop

		print("Seiffert Spiral")

	def get_gradients_from_T(self, T, undersamp_traj):
		nupsamp = T / GRT

		GRT = self.system.grad_raster_time

		nshots = undersamp_traj.shape[0]

		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio
		
		fov = self.imgprop.fov
		resolution = self.imgprop.resolution
		delta_k = 1.0 / fov

		trajectory = np.zeros((nshots, nupsamp+2, 3))
		trajectory[:,2:,:] = mn.oversample(undersamp_traj, nupsamp)
	
		grad_list = []
		grad_max_list = []

		slew_list = []
		slew_max_list = []

		for i in range(nshots):
			gi, si = pp.traj_to_grad(trajectory[i,...].transpose(1,0), raster_time=GRT)

			max_grad = np.abs(gi).max()
			max_slew = np.abs(si).max()
			if max_grad > safe_max_grad:
				print('Gradient failure: ', convert(max_grad, from_unit='Hz/m', to_unit='mT/m'))
				raise RuntimeError('Gradient failure: ', max_grad)
			elif max_slew > safe_max_slew:
				print('Slew failure: ', convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'))
				raise RuntimeError('Slew failure')
			
			grad_list.append(gi)
			slew_list.append(si)
			grad_max_list.append(max_grad)
			slew_max_list.append(max_slew)

		grad_list = np.stack(grad_list, axis=0)
		slew_list = np.stack(slew_list, axis=0)
		grad_max_list = np.array(grad_max_list)
		slew_max_list = np.array(slew_max_list)

		return trajectory, grad_list, slew_list, grad_max_list, slew_max_list

	def get_gradients_calc_T(self, undersamp_traj):
		GRT = self.system.grad_raster_time

		nshots = undersamp_traj.shape[0]

		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio
		
		fov = self.imgprop.fov
		resolution = self.imgprop.resolution
		delta_k = 1.0 / fov

		grad_list = []
		grad_max_list = []

		slew_list = []
		slew_max_list = []

		max_grad = 0.0
		max_slew = 0.0

		tmax = 1e-3
		safe_gradients = False
		for _ in range(100):

			slew_failure = False
			grad_failure = False

			nsamp = int(tmax / GRT)
			trajectory = np.zeros((nshots, nsamp+2, 3))
			trajectory[:,2:,:] = mn.oversample(
				undersamp_traj, 
				nsamp #nsamp
			) * (resolution * delta_k)

			max_grad = 0.0
			max_slew = 0.0

			grad_list.clear()
			grad_max_list.clear()
			slew_list.clear()
			slew_max_list.clear()
			for i in range(nshots):
				gi, si = pp.traj_to_grad(trajectory[i,...].transpose(1,0), raster_time=GRT)

				max_grad = np.abs(gi).max()
				max_slew = np.abs(si).max()
				if max_grad > safe_max_grad:
					print('Gradient failure: ', convert(max_grad, from_unit='Hz/m', to_unit='mT/m'))
					grad_failure = True
					break
				elif max_slew > safe_max_slew:
					print('Slew failure: ', convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'))
					slew_failure = True
					break

				if i == 0 and False:
					plot31(si, 'Slew Rate')
					plot31(gi, 'Gradient')

					v = np.sqrt(np.sum(np.square(gi), axis=0))

					plt.figure()
					plt.plot(v)
					plt.title('K-space speed')
					plt.show()

				grad_list.append(gi)
				slew_list.append(si)
				grad_max_list.append(max_grad)
				slew_max_list.append(max_slew)


			if grad_failure:
				tmax *= (1.05 * max_grad / safe_max_grad)
			elif slew_failure:
				tmax *= (1.05 * max_slew / safe_max_slew)
			else:
				safe_gradients = True
				break

		if not safe_gradients:
			raise RuntimeError("Gradient or slew rate failure")
		
		grad_list = np.stack(grad_list, axis=0)
		slew_list = np.stack(slew_list, axis=0)
		grad_max_list = np.array(grad_max_list)
		slew_max_list = np.array(slew_max_list)

		return trajectory, grad_list, slew_list, grad_max_list, slew_max_list

	def get_gradients(self, nshots, T, curveiness=0.9999, oncurve_samples=50):
		undersamp_traj = mn.initialize_3D_seiffert_spiral(nshots, oncurve_samples, curve_index=curveiness)

		if T is None:
			trajectory, grad_list, slew_list, grad_max_list, slew_max_list = self.get_gradients_calc_T(undersamp_traj)
		else:
			trajectory, grad_list, slew_list, grad_max_list, slew_max_list = self.get_gradients_from_T(T, undersamp_traj)

		if self.print_calc:
			#tu.show_trajectory(trajectory, 0, figure_size = 8)

			subsample = trajectory[np.random.randint(0, nshots, 50),...]
			tu.show_trajectory(0.5 * subsample / np.abs(subsample).max(), 0, figure_size = 8)

			rad_sample = np.sqrt(np.sum(np.square(subsample), axis=-1))
			plt.figure()
			plt.plot(rad_sample.T)
			plt.title('Distance to center over time')
			plt.show()

		return trajectory, grad_list, slew_list, grad_max_list, slew_max_list
		

if __name__ == "__main__":
		#T, DT = calc_T_DT(0.001, 80, 200, 42.58e6)
	system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)

	ltik = LTIGradientKernels(system).init_from_test()

	trajfactory = SeiffertSpiral(ltik, SafetyLimits(), ImageProperties(None, 220e-3, 256+128), print_calc=True)

	scan_time = 60*10

	nshots = int(scan_time / 0.019)

	back = trajfactory.get_gradients(nshots, None, curveiness=0.9999, oncurve_samples=50)