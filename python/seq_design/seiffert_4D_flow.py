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
from traj_design import SeiffertSpiral

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

venc1 = 0.2
venc2 = 0.8

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

ltik = LTIGradientKernels(system).init_from_test()

sl = SafetyLimits()
imgprop = ImageProperties([320,320,320], 220e-3, 320)

vef = VelocityEncodingFactory(ltik, sl, print_calc=False)

E0g = vef.get_gradients([None, None, None])
E1g = vef.get_gradients([venc1*-1.0, venc1*-1.0, venc1*-1.0])
E2g = vef.get_gradients([venc1* 1.0, venc1* 1.0, venc1*-1.0])
E3g = vef.get_gradients([venc1* 1.0, venc1*-1.0, venc1* 1.0])
E4g = vef.get_gradients([venc1*-1.0, venc1* 1.0, venc1* 1.0])
E5g = vef.get_gradients([venc2*-1.0, venc2*-1.0, venc2* 1.0])
E6g = vef.get_gradients([venc2* 1.0, venc2*-1.0, venc2*-1.0])
E7g = vef.get_gradients([venc2*-1.0, venc2* 1.0, venc2*-1.0])

Eg_list = [E0g, E1g, E2g, E3g, E4g, E5g, E6g, E7g]

tf = SeiffertSpiral(ltik, sl, imgprop, print_calc=True)

scan_time = 60*10

nshots = int(scan_time / 0.019) // 8

trajectory, traj_grad_list, traj_slew_list, traj_grad_max_list, traj_slew_max_list = tf.get_gradients(nshots, None)

TRAJ_N_SAMPLES = trajectory.shape[1]
VELENC_N_SAMPLES = E0g[0][0].shape[0]

TRAJ_TIME = TRAJ_N_SAMPLES*system.grad_raster_time
VELENC_TIME = VELENC_N_SAMPLES*system.grad_raster_time

TR = 3e-3 + TRAJ_TIME + VELENC_TIME

T1 = 1.9 # We target blood with T1 approx 1.9 s
ernst_angle = np.arccos(np.exp(-TR/T1))
print('TR: ', TR, 'Ernst angle: ', np.rad2deg(ernst_angle))

rf, gzs, gzsr = pp.make_sinc_pulse(
	ernst_angle,
	system=system,
	duration=3e-3,
	slice_thickness=imgprop.fov,
	apodization=0.5,
	time_bw_product=4,
	return_gz=True,
)

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

seq = pp.Sequence(system=system)

for shotidx, shot in enumerate(shotperm):

	for encidx, enc in enumerate(Eg_list):

		seq.add_block(rf, gzsr)

		vel_gx = pp.make_arbitrary_grad(channel='x', waveform=enc[0][0], system=system)
		vel_gy = pp.make_arbitrary_grad(channel='y', waveform=enc[0][1], system=system)
		vel_gz = pp.make_arbitrary_grad(channel='z', waveform=enc[0][2], system=system)

		seq.add_block(vel_gx, vel_gy, vel_gz, vel_enc_adc)

		traj_gx = pp.make_arbitrary_grad(channel='x', waveform=traj_grad_list[shot, 0, :], system=system)
		traj_gy = pp.make_arbitrary_grad(channel='y', waveform=traj_grad_list[shot, 1, :], system=system)
		traj_gz = pp.make_arbitrary_grad(channel='z', waveform=traj_grad_list[shot, 2, :], system=system)

		seq.add_block(traj_gx, traj_gy, traj_gz, traj_adc)

		seq.add_block(pp.make_delay(1e-4))

seq.plot()

seq.write('/home/turbotage/Documents/seiffert_4D_flow.seq')