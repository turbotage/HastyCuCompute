import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp
from pypulseq.convert import convert
from pypulseq import add_ramps as pp_ramp

import scipy as sp
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
import scipy.special as sps
from scipy.integrate import cumulative_trapezoid

from sequtil import SafetyLimits, LTIGradientKernels, ImageProperties

#from ramping import calc_ramp, add_ramps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def plot31(slew, title='', vlines=None):
	plt.figure()
	plt.plot(slew[0, :], 'r-*')
	plt.plot(slew[1, :], 'g-*')
	plt.plot(slew[2, :], 'b-*')
	if vlines is not None:
		for vline in vlines:
			plt.axvline(x=vline, color='k', linestyle='--')
		#plt.vlines(vlines, color='k', linestyle='--')
	plt.title(title)
	plt.show()

def traj_from_grad(grad, raster_time):
	return cumulative_trapezoid(grad, dx=raster_time, axis=2)

def calc_trapz_T_DT(area, slew_rate, max_grad, grad_raster_time):
	# T=Zero =>
	DT = math.sqrt(area / slew_rate)
	GRT = grad_raster_time

	if DT * slew_rate > max_grad:
		DT = max_grad / slew_rate
		T = area / (slew_rate * DT) - DT
	else:
		T = 0.0
	return T, DT

# def make_many_trapz_grad(area, slew_rate, max_grad, grad_raster_time):
# 	grads = []
# 	for i in range(area.shape[0]):
# 		inner_grads = np.empty((area.shape[1],))
# 		for j in range(area.shape[1]):
# 			inner_grads[j] = make_trapz_grad(float(area[i,j]), slew_rate, max_grad, grad_raster_time)
# 		grads.append(inner_grads)
# 	return np.stack(grads, axis=0)
			

class Spiral3D:
	def __init__(self, ltik: LTIGradientKernels, safety: SafetyLimits, imgprop: ImageProperties ,print_calc=False):
		self.system = ltik.system
		self.print_calc = print_calc
		self.ltik = ltik
		self.safety = safety
		self.imgprop = imgprop

		print("Seiffert Spiral")

	@staticmethod
	def get_default_seiffert_settings():
		return {
			'spiral_type': "seiffert",
			'oncurve_samples': 80, 
			'curveiness': 0.9999,
			'add_rand_perturb': False
		}
	
	@staticmethod
	def get_default_cones_settings():
		return {
			'spiral_type': "cones-3d",
			'oncurve_samples': 80, 
			'tilt': "golden",
			'in_out': False,
			'nb_zigzags': 5,
			'spiral': "archimedes",
			'width': 1.0,
			'add_rand_perturb': False
		}

	def one_seiffert_run(self, trajectory):
		MKL = self.ltik.max_kernel_length
		GRT = self.system.grad_raster_time
		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio
		
		gi, si = pp.traj_to_grad(trajectory, raster_time=GRT)

		# We wan't our ramp to have a max_slew same as max_slew of trajectory
		max_traj_slew = min(np.abs(si).max(), safe_max_slew)

		start_traj_gi = gi[:,:,0][:,:,None]
		max_start_gi = np.abs(start_traj_gi).max()

		upramp_NGRT = math.ceil((max_start_gi / max_traj_slew) / GRT)
		upramp_DT = upramp_NGRT * GRT
		upramp_slew = start_traj_gi / upramp_DT
		upramp_gi = np.arange(upramp_NGRT)[None,None,:] * GRT * upramp_slew
		upramp_gi = np.concatenate([upramp_gi, np.repeat(gi[:,:,0][...,None], MKL+1, axis=2)], axis=2)
		upramp_area = start_traj_gi * (upramp_NGRT/2 + MKL+1) * GRT

		# Add a prephaser
		max_area = max_start_gi * (upramp_NGRT/2+MKL+1) * GRT
		T, DT = calc_trapz_T_DT(max_area, max_traj_slew, safe_max_grad, GRT)
		T_N = math.ceil(T / GRT)
		T = T_N * GRT
		DT_N = math.ceil(DT / GRT)
		DT = DT_N * GRT
		prephase_slew_rates = -upramp_area / (DT*(T+DT))
		prephase_ramp = np.arange(DT_N)[None,None,:] * GRT * prephase_slew_rates
		prephaser_gi = np.concatenate([
						prephase_ramp, 
						np.repeat(prephase_slew_rates * DT, T_N, axis=2), 
						np.flip(prephase_ramp, axis=2)
					], axis=2)
		
		# We need to downramp the gradients to zero
		end_traj_gi = gi[:,:,-1][:,:,None]
		max_end_gi = np.abs(end_traj_gi).max()
		DT = max_end_gi / (0.95*safe_max_slew)
		DT_N = math.ceil(DT / GRT)
		DT = DT_N * GRT
		downramp_slew = end_traj_gi / DT
		downramp_gi = np.flip(np.arange(DT_N)[None,None,:] * GRT * downramp_slew, axis=2)

		# Start the trajectory with 2 GRT of zero gradients
		zero_gi = np.zeros((gi.shape[0],3,3))

		gi = np.concatenate([zero_gi, prephaser_gi, upramp_gi, gi, downramp_gi], axis=2)

		trajectory = traj_from_grad(gi, GRT)

		grad, slew = pp.traj_to_grad(trajectory, raster_time=GRT)

		return trajectory, grad, slew

	def one_cones_run(self, trajectory):
		GRT = self.system.grad_raster_time
		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio
		
		gi, si = pp.traj_to_grad(trajectory, raster_time=GRT)

		end_traj_gi = gi[:,:,-1][:,:,None]
		max_end_gi = np.abs(end_traj_gi).max()
		DT = max_end_gi / (0.95*safe_max_slew)
		DT_N = math.ceil(DT / GRT)
		DT = DT_N * GRT
		downramp_slew = end_traj_gi / DT
		downramp_gi = np.flip(np.arange(DT_N)[None,None,:] * GRT * downramp_slew, axis=2)

		# Start the trajectory with 2 GRT of zero gradients
		zero_gi = np.zeros((gi.shape[0],3,3))

		gi = np.concatenate([zero_gi, gi, downramp_gi], axis=2)

		trajectory = traj_from_grad(gi, GRT)

		grad, slew = pp.traj_to_grad(trajectory, raster_time=GRT)

		return trajectory, grad, slew

	def get_fastest_spiral_gradients(self, undersamp_traj, spiral_settings):

		MKL = self.ltik.max_kernel_length
		GRT = self.system.grad_raster_time
		nshots = undersamp_traj.shape[0]
		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio

		def one_run(NGRT):

			trajectory = CubicSpline(
							np.linspace(0,1,undersamp_traj.shape[2]), 
							undersamp_traj, 
							axis=2
						)(np.linspace(0,1,NGRT))

			if spiral_settings['spiral_type'] == "seiffert":
				trajectory, grad, slew = self.one_seiffert_run(trajectory)
			elif spiral_settings['spiral_type'] == "cones-3d":
				trajectory, grad, slew = self.one_cones_run(trajectory)
			else:
				raise ValueError('Unknown spiral type: ', spiral_settings['spiral_type'])

			#max_slew_shot = np.argmax(np.abs(gi).max(axis=(1,2)), axis=0)
			#random_shot = np.random.randint(0, nshots)
			#plot31(convert(gi[max_slew_shot,...], from_unit='Hz/m', to_unit='mT/m'), 'Gradient')
			#plot31(convert(si[max_slew_shot,...], from_unit='Hz/m/s', to_unit='T/m/s'), 'Slew Rate')

			max_grad = np.abs(grad).max()
			max_slew = np.abs(slew).max()

			return trajectory, grad, slew, max_grad, max_slew

		NGRT_i = max(math.ceil(0.5e-3 / GRT), undersamp_traj.shape[2])
		trajectory, gi, si, max_grad, max_slew = one_run(NGRT_i)

		if max_grad > safe_max_grad or max_slew > safe_max_slew:
			off_ratio= max_grad / safe_max_grad
			if off_ratio < max_slew / safe_max_slew:
				off_ratio = max(off_ratio, math.sqrt(max_slew / safe_max_slew))
				
			NGRT_i = math.ceil(NGRT_i * off_ratio)

		trajectory, gi, si, max_grad, max_slew = one_run(NGRT_i)
		if max_grad > safe_max_grad or max_slew > safe_max_slew:
			decrease_NGRT = False
		else:
			decrease_NGRT = True
		
		print('')
		for i in range(500):
			if decrease_NGRT:
				NGRT_i -= 100
			else:
				NGRT_i += 100

			print('\rAttempting a readout of: ', NGRT_i * GRT * 1e3, 'ms', end='')

			temp_trajectory, temp_gi, temp_si, temp_max_grad, temp_max_slew = one_run(NGRT_i)

			if temp_max_grad < safe_max_grad and temp_max_slew < safe_max_slew:
				# The update gave a new successful trajectory
				trajectory = 	temp_trajectory
				gi = 			temp_gi
				si = 			temp_si
				max_grad = 		temp_max_grad
				max_slew = 		temp_max_slew
				if not decrease_NGRT:
					# If we were increasing NGRT, ie previously our trajectory did not meat
					# grad or slew requirements, we have now found a successful trajectory
					# and should stop
					break
			elif decrease_NGRT:
				# If we were decreasing NGRT our previous trajectory was successful, temp_trajectory
				# is now the first non-successful trajectory, so stop here with the previous trajectory
				break
		print('')

		max_slew_shot = np.argmax(np.abs(gi).max(axis=(1,2)), axis=0)
		random_shot = np.random.randint(0, nshots)
		plot31(convert(gi[max_slew_shot,...], from_unit='Hz/m', to_unit='mT/m'), 'Gradient')
		plot31(convert(si[max_slew_shot,...], from_unit='Hz/m/s', to_unit='T/m/s'), 'Slew Rate')

		max_grad = np.abs(gi).max()
		max_slew = np.abs(si).max()

		plot31(trajectory[max_slew_shot,...], 'Trajectory')

		return trajectory, gi, si, max_grad, max_slew

	def get_gradients(self, nshots, spiral_settings=get_default_seiffert_settings(), speed_interpolation=None):

		if spiral_settings['spiral_type'] == "seiffert":
			undersamp_traj = mn.initialize_3D_seiffert_spiral(
									nshots, 
									spiral_settings['oncurve_samples'], 
									curve_index=spiral_settings['curveiness']
								)
		elif spiral_settings['spiral_type'] == "cones-3d":
			undersamp_traj = mn.initialize_3D_cones(
								nshots, 
								spiral_settings['oncurve_samples'], 
								tilt=spiral_settings['tilt'],
								in_out=spiral_settings['in_out'],
								nb_zigzags=spiral_settings['nb_zigzags'],
								spiral=spiral_settings['spiral'],
								width=spiral_settings['width']
							)
		else:
			raise ValueError('Unknown spiral type: ', spiral_settings['spiral_type'])

		fov = self.imgprop.fov
		resolution = self.imgprop.resolution
		delta_k = 1.0 / self.imgprop.fov

		undersamp_traj = np.ascontiguousarray(undersamp_traj.transpose(0,2,1))
		undersamp_traj *= (resolution * delta_k)

		if spiral_settings['add_rand_perturb']:
			perturbation = 0.2*np.random.uniform(-delta_k, delta_k, undersamp_traj.shape)
			perturb_factor = np.linspace(0,1,undersamp_traj.shape[2])[None,None,:]
			perturb_factor = np.square(perturb_factor)
			perturbation *= perturb_factor
			undersamp_traj += perturbation

		if speed_interpolation is not None:
			undersamp_traj = speed_interpolation(undersamp_traj)


		trajectory, grad, slew, max_grad, max_slew = self.get_fastest_spiral_gradients(undersamp_traj, spiral_settings)

		if self.print_calc:
			print('Max gradient: ', convert(max_grad, from_unit='Hz/m', to_unit='mT/m'))
			print('Max slew: ', convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'))

			rand_sample_mask = np.random.randint(0, nshots, 50)
			traj_subsample = trajectory[rand_sample_mask,...].transpose(0,2,1)
			tu.show_trajectory(0.5 * traj_subsample / np.abs(traj_subsample).max(), 0, figure_size = 8)

			rad_sample = np.sqrt(np.sum(np.square(traj_subsample), axis=-1))
			plt.figure()
			plt.plot(rad_sample.T)
			plt.title('Distance to center over time')
			plt.show()

			grad_subsample = grad[rand_sample_mask,...]
			velocity_sample = np.sqrt(np.sum(np.square(grad_subsample), axis=1))
			plt.figure()
			plt.plot(velocity_sample.T)
			plt.title('Velocity over time')
			plt.show()

		return trajectory, grad, slew, max_grad, max_slew
		

if __name__ == "__main__":
		#T, DT = calc_T_DT(0.001, 80, 200, 42.58e6)
	system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)

	ltik = LTIGradientKernels(system, LTIGradientKernels.kernels_from_test())

	#trajfactory = SeiffertSpiral(ltik, SafetyLimits(), ImageProperties(None, 220e-3, 256+64), print_calc=True)

	scan_time = 60*10

	nshots = int(scan_time / 0.019)

	back = trajfactory.get_gradients(nshots, None, curveiness=0.9999, oncurve_samples=50)




