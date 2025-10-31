import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pulserver as pps
import pypulseq as pp
from pypulseq.convert import convert
from pypulseq import add_ramps as pp_ramp

import scipy as sp
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d, LinearNDInterpolator
import scipy.special as sps
from scipy.integrate import cumulative_trapezoid

from sequtil import SafetyLimits, LTIGradientKernels, ImageProperties
from vel_design import VelocityEncodingFactory



import my_trajectories as mytraj

#from ramping import calc_ramp, add_ramps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def plot31(data, title='', vlines=None, norm=False):
	plt.figure()
	plt.plot(data[0, :], 'r-*')
	plt.plot(data[1, :], 'g-*')
	plt.plot(data[2, :], 'b-*')
	if vlines is not None:
		for vline in vlines:
			plt.axvline(x=vline, color='k', linestyle='--')
		#plt.vlines(vlines, color='k', linestyle='--')
	if norm:
		plt.plot(np.sqrt(np.sum(np.square(data), axis=0)), 'c-*')
	plt.title(title)
	plt.show()

def plot1(data, title='', vlines=None):
	plt.figure()
	plt.plot(data, 'b-*')
	if vlines is not None:
		for vline in vlines:
			plt.axvline(x=vline, color='k', linestyle='--')
		#plt.vlines(vlines, color='k', linestyle='--')
	plt.title(title)
	plt.show()


class YarnballSpiralSettings:
	def __init__(self):
		self.nshots = 20
		#self.oncurve_samples = 2000
		self.nb_revs = 5
		self.nb_folds = 5
		self.add_rand_perturb = False
		self.rand_perturb_factor = 0.2
		self.rho_lambda = None


class YarnballSpiral:
	def __init__(self, ltik: LTIGradientKernels, safety: SafetyLimits, imgprop: ImageProperties, yarnsettings: YarnballSpiralSettings, print_calc=False):
		self.system = ltik.system
		self.print_calc = print_calc
		self.ltik = ltik
		self.safety = safety
		self.imgprop = imgprop
		self.yarnsettings = yarnsettings

		print("Yarnball Spiral")

	def one_run(self, NGRT, nshots=None):
		t = np.linspace(0,1,NGRT)

		if nshots is None:
			nshots = self.yarnsettings.nshots

		trajectory = mytraj.initialize_my_yarn_ball(
						nshots, 
						NGRT, 
						tilt="",
						nb_revs=self.yarnsettings.nb_revs,
						nb_folds=self.yarnsettings.nb_folds,
						rho_lambda=self.yarnsettings.rho_lambda
					)

		resolution = self.imgprop.resolution
		delta_k = 1.0 / self.imgprop.fov

		trajectory = np.ascontiguousarray(trajectory.transpose(0,2,1))
		trajectory *= (0.5*resolution * delta_k)[None,:,None]

		zeroi = np.zeros((nshots, 3, 2))
		trajectory = np.concatenate([zeroi, trajectory, zeroi], axis=-1)

		trajectory[:,:,1:-1] = 0.4*trajectory[:,:,1:-1] + 0.3*trajectory[:,:,0:-2] + 0.3*trajectory[:,:,2:]

		grad, slew = pp.traj_to_grad(trajectory, raster_time=self.system.grad_raster_time)

		max_grad = np.abs(grad).max()
		max_slew = np.abs(slew).max()

		return trajectory, grad, slew, max_grad, max_slew

	def get_fastest_gradients(self):
		MKL = self.ltik.max_kernel_length
		GRT = self.system.grad_raster_time
		nshots = self.yarnsettings.nshots
		safe_max_grad = self.system.max_grad * self.safety.grad_ratio
		safe_max_slew = self.system.max_slew * self.safety.slew_ratio

		NGRT_i = max(math.ceil(0.5e-3 / GRT), self.yarnsettings.oncurve_samples)
		trajectory, gi, si, max_grad, max_slew = self.one_run(NGRT_i)

		if max_grad > safe_max_grad or max_slew > safe_max_slew:
			off_ratio= max_grad / safe_max_grad
			if off_ratio < max_slew / safe_max_slew:
				off_ratio = max(off_ratio, math.sqrt(max_slew / safe_max_slew))
				
			NGRT_i = math.ceil(NGRT_i * off_ratio)

		trajectory, gi, si, max_grad, max_slew = self.one_run(NGRT_i)
		if max_grad > safe_max_grad or max_slew > safe_max_slew:
			decrease_NGRT = False
		else:
			decrease_NGRT = True

		print('', end='')
		for i in range(2000):
			if decrease_NGRT:
				NGRT_i -= 20
			else:
				NGRT_i += 20

			print('\rAttempting a readout of: ', NGRT_i * GRT * 1e3, 'ms, iteration: ', i, end='')

			temp_trajectory, temp_gi, temp_si, temp_max_grad, temp_max_slew = self.one_run(NGRT_i)

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

		if self.print_calc:
			max_slew_shot = np.argmax(np.abs(si).max(axis=(1,2)), axis=0)
			max_grad_shot = np.argmax(np.abs(gi).max(axis=(1,2)), axis=0)
			plot31(convert(gi[max_slew_shot,...], from_unit='Hz/m', to_unit='mT/m'), 'Gradient : Max Slew Spoke', norm=True)
			plot31(convert(si[max_slew_shot,...], from_unit='Hz/m/s', to_unit='T/m/s'), 'Slew Rate : Max Slew Spoke', norm=True)
			plot31(convert(gi[max_grad_shot,...], from_unit='Hz/m', to_unit='mT/m'), 'Gradient : Max Slew Spoke', norm=True)
			plot31(convert(si[max_grad_shot,...], from_unit='Hz/m/s', to_unit='T/m/s'), 'Slew Rate : Max Slew Spoke', norm=True)
			plot31(trajectory[max_slew_shot,...], 'Trajectory : Max Slew Spoke')
			plot31(trajectory[max_grad_shot,...], 'Trajectory : Max Grad Spoke')
			plot1(np.sqrt(np.sum(np.square(trajectory[max_slew_shot,...]), axis=0)), 'Distance to center over time : Max Slew Spoke')
		
		max_grad = np.abs(gi).max()
		max_slew = np.abs(si).max()

		return trajectory, gi, si, max_grad, max_slew


def get_yarnball_optimal_yarnball(system, ltik, sl, imgprop):
	yarnsettings = YarnballSpiralSettings()
	#yarnsettings.oncurve_samples = 3000
	yarnsettings.nb_revs = 6
	yarnsettings.nb_folds = 3
	#yarnsettings.rho_lambda = mytraj.my_yarn_ball_default_rho(0.01, 0.8, 50, 9)
	#yarnsettings.rho_lambda = mytraj.my_yarn_ball_default_rho_2(0.01, 20, 0.05)
	yarnsettings.rho_lambda = mytraj.my_yarn_ball_default_rho_3(0.8, 0.8, 0.75, 25, 12)
	#yarnsettings.rho_lambda = mytraj.my_yarn_ball_default_rho_4()

	yarnball = YarnballSpiral(ltik, sl, imgprop, yarnsettings, print_calc=True)

	traj, gi, si, max_grad, max_slew = yarnball.get_fastest_gradients()

	NGRT = traj.shape[2]

	yarnball_runner = lambda ngrt, ns: yarnball.one_run(ngrt, ns)

	return yarnball_runner, NGRT

def get_velocity_encodings(system, ltik, sl, venc):
	vencs = [None, 		 None, 		 None] 		 	+ \
			[venc*(-1.0), venc*(-1.0), venc*(-1.0)] + \
			[venc*(1.0),  venc*(1.0),  venc*(-1.0)] + \
			[venc*(1.0),  venc*(-1.0), venc*(1.0)] 	+ \
			[venc*(-1.0), venc*(1.0),  venc*(1.0)]

	channels = []
	for i,_ in enumerate(vencs):
		channels += ['x', 'y', 'z']

	vef = VelocityEncodingFactory(ltik, SafetyLimits(0.7, 0.95), print_calc=False)
	VelEnc_ret = vef.get_gradients(vencs, channels)
	VelEnc_grads = []
	VelEnc_props = []
	for enci in range(len(VelEnc_ret[0])//3):
		VelEnc_grads.append([
			VelEnc_ret[0][enci*3], 
			VelEnc_ret[0][enci*3+1], 
			VelEnc_ret[0][enci*3+2]
		])
		VelEnc_props.append([
			VelEnc_ret[1][enci*3], 
			VelEnc_ret[1][enci*3+1], 
			VelEnc_ret[1][enci*3+2]
		])

	return VelEnc_grads, VelEnc_props


if __name__ == "__main__":

	system = pps.Opts(
		max_grad=80, grad_unit='mT/m', 
		max_slew=200, slew_unit='T/m/s',
		rf_raster_time=2e-6,
		rf_dead_time=100e-6,
		rf_ringdown_time=60e-6,
		adc_raster_time=2e-6,
		adc_dead_time=40e-6,
		grad_raster_time=4e-6,
		block_duration_raster=4e-6,
		B0=3.0, 
	)

	ltik = LTIGradientKernels(system, LTIGradientKernels.kernels_from_test(system.grad_raster_time))

	sl = SafetyLimits()
	imgprop = ImageProperties([320,320,320], 
				np.array([220e-3, 220e-3, 220e-3]), np.array([320,320,320]))
	

	yarnball_runner, NGRT = get_yarnball_optimal_yarnball(system, ltik, sl, imgprop)

	vel_grads, vel_props = get_velocity_encodings(system, ltik, sl, venc=0.7)

	TRAJ_N_SAMPLES = NGRT+2
	VELENC_N_SAMPLES = vel_grads[0][0].shape[0]

	TRAJ_TIME = TRAJ_N_SAMPLES*system.grad_raster_time
	VELENC_TIME = VELENC_N_SAMPLES*system.grad_raster_time

	scan_time = 5*60.0 # 5 minutes

	nshots = math.floor(scan_time / (TRAJ_TIME + VELENC_TIME))

	seq = pps.Sequence(
		system, platform="pulseq"
	)

	vel_enc_adc = pp.make_adc(
					num_samples=VELENC_N_SAMPLES, 
					duration=VELENC_TIME, 
					delay=0, 
					system=system
				)

	for encidx, velenc_grad in enumerate(vel_grads):

		vel_gx = pp.make_arbitrary_grad(channel='x', waveform=velenc_grad[0], system=system)
		vel_gy = pp.make_arbitrary_grad(channel='y', waveform=velenc_grad[1], system=system)
		vel_gz = pp.make_arbitrary_grad(channel='z', waveform=velenc_grad[2], system=system)

		seq.register_block(name=f"VelEnc_[{encidx}]", gx=vel_gx, gy=vel_gy, gz=vel_gz, adc=vel_enc_adc)



	



