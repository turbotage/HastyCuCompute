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

from vel_design import VelocityEncodingFactory
from traj_design import Spiral3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def speed_interpolation(undersamp_traj, option):
	PRE_N_SAMPLES = undersamp_traj.shape[2]

	cs = CubicSpline(np.linspace(0,1,PRE_N_SAMPLES), undersamp_traj, axis=2)
	if option==1:
		k1 = 0.2
		k2 = 10.0
		t_end = -(1.0 / k2) * math.log(1.0 - (1.0 / k1))
		tnew = np.linspace(0, t_end, 5*PRE_N_SAMPLES)
		tnew = k1*(1.0 - np.exp(-k2*tnew))
	elif option==2:
		k1 = 0.02
		k2 = 5.0
		t_end = ((1/k2) + math.sqrt(k1))**2 - k1
		tnew = np.linspace(0, t_end, 5*PRE_N_SAMPLES)
		tnew = k2*(np.sqrt(k1 + tnew) - math.sqrt(k1))

	convolve = True
	if convolve:
		kernel = np.array([0.0, 0.0, 0.1, 0.3, 0.7, 0.3, 0.1, 0.0, 0.0])
		kernel /= np.sum(kernel)
		tnew = np.convolve(tnew, kernel, mode='full')[:-kernel.shape[0]]
		tnew = tnew[tnew < 1.0]

	#plt.figure()
	#plt.plot(tnew)
	#plt.plot(np.linspace(0,1,5*PRE_N_SAMPLES))
	#plt.show()
	undersamp_traj = cs(tnew) 
	return undersamp_traj

venc1 = 0.4
venc2 = 0.9

system = pp.Opts(
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

vencs = [None, 		 None, 		 None] 		 + \
		[venc1*-1.0, venc1*-1.0, venc1*-1.0] + \
		[venc2* 1.0, venc2*-1.0, venc2*-1.0] + \
		[venc1* 1.0, venc1* 1.0, venc1*-1.0] + \
		[venc2*-1.0, venc2*-1.0, venc2* 1.0] + \
		[venc1* 1.0, venc1*-1.0, venc1* 1.0] + \
		[venc2*-1.0, venc2* 1.0, venc2*-1.0] + \
		[venc1*-1.0, venc1* 1.0, venc1* 1.0]

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

tf = Spiral3D(ltik, sl, imgprop, print_calc=True)

scan_time = 60*10

estimate_rough_TR = 0.03
nshots = int(scan_time / estimate_rough_TR) // 8
nshots = 20

spiral_type = 'my_yarn_ball'
if spiral_type == 'cones':
	spiral_settings = Spiral3D.get_default_cones_settings()
	spiral_settings['width'] = 40
	spiral_settings['nb_zigzags'] = 12
	spiral_settings['oncurve_samples'] = 160
	spiral_settings['add_rand_perturb'] = True

	speed_interpolator = lambda x: speed_interpolation(x, 2)
elif spiral_type == 'seiffert':
	spiral_settings = Spiral3D.get_default_seiffert_settings()
	spiral_settings['add_rand_perturb'] = True
	speed_interpolator = None
elif spiral_type == 'my_yarn_ball':
	spiral_settings = Spiral3D.get_default_my_yarn_ball_settings()
	spiral_settings['nb_revs'] = 17
	spiral_settings['nb_folds'] = 5
	spiral_settings['add_rand_perturb'] = True
	spiral_settings['oncurve_samples'] = 800
	spiral_settings['rand_perturb_factor'] = 1e-3

	speed_interpolator = lambda x: speed_interpolation(x, 2)



tfret = tf.get_gradients(nshots, spiral_settings, speed_interpolation=speed_interpolator)
trajectory = tfret[0]
traj_grad_list = tfret[1]
traj_slew_list = tfret[2]
traj_grad_max_list = tfret[3]
traj_slew_max_list = tfret[4]

TRAJ_N_SAMPLES = trajectory.shape[2]
VELENC_N_SAMPLES = VelEnc_grads[0][0].shape[0]

TRAJ_TIME = TRAJ_N_SAMPLES*system.grad_raster_time
VELENC_TIME = VELENC_N_SAMPLES*system.grad_raster_time

TR = 3e-3 + TRAJ_TIME + VELENC_TIME

T1 = 1.9 # We target blood with T1 approx 1.9 s
ernst_angle = np.arccos(np.exp(-TR/T1))
print('TR: ', TR, 'Ernst angle: ', np.rad2deg(ernst_angle))

vel_enc_adc = pp.make_adc(
					num_samples=VELENC_N_SAMPLES, 
					duration=VELENC_TIME, 
					delay=0, 
					system=system
				)

traj_adc = pp.make_adc(
				num_samples=TRAJ_N_SAMPLES,
				duration=TRAJ_TIME,
				delay=0,
				system=system
			)

shotperm = np.random.permutation(nshots)


rf_spoiling_inc = 117
rf_inc = 0
rf_phase = 0

rf, gzs, gzsr = pp.make_sinc_pulse(
	ernst_angle,
	system=system,
	duration=3e-3,
	slice_thickness=imgprop.fov[2],
	apodization=0.5,
	time_bw_product=4,
	return_gz=True,
	max_slew=system.max_slew / 5,
)

seq = pp.Sequence(system=system)
for shotidx, shot in enumerate(shotperm):

	for encidx, velenc_grad in enumerate(VelEnc_grads):
		rf.phase_offset = math.pi * rf_phase / 180.0
		traj_adc.phase_offset = math.pi * rf_phase / 180.0
		vel_enc_adc.phase_offset = math.pi * rf_phase / 180.0

		seq.add_block(rf, gzsr)

		#vel_gx = pp.make_arbitrary_grad(channel='x', waveform=velenc_grad[0], system=system)
		#vel_gy = pp.make_arbitrary_grad(channel='y', waveform=velenc_grad[1], system=system)
		#vel_gz = pp.make_arbitrary_grad(channel='z', waveform=velenc_grad[2], system=system)

		#seq.add_block(vel_gx, vel_gy, vel_gz, vel_enc_adc)

		traj_gx = pp.make_arbitrary_grad(channel='x', waveform=traj_grad_list[shot, 0, :], system=system)
		traj_gy = pp.make_arbitrary_grad(channel='y', waveform=traj_grad_list[shot, 1, :], system=system)
		traj_gz = pp.make_arbitrary_grad(channel='z', waveform=traj_grad_list[shot, 2, :], system=system)

		seq.add_block(traj_gx, traj_gy, traj_gz, traj_adc)

		seq.add_block(pp.make_delay(1e-4))

seq.plot(time_range=(0, TR))
seq.plot(time_range=(0, 3*TR))
seq.plot(time_range=(0, 9*TR))
#seq.plot()

from pypulseq.utils.safe_pns_prediction import safe_example_hw

pns_ok, pns_n, pns_c, tpns = seq.calculate_pns(safe_example_hw(), do_plots=True)  # Safe example HW
plt.show()

print('PNS OK: ', pns_ok)
print('PNS Norm: ', pns_n)
print('PNS Components: ', pns_c)
print('PNS Time: ', tpns)

seq.write('/home/turbotage/Documents/seiffert_4D_flow.seq')