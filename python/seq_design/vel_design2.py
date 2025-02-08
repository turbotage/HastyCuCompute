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
def calc_T_DT(venc, max_grad, slew_rate, gamma):
	max_grad *= 1e-3

	DM_s = np.pi/(gamma*venc)

	DM = first_moment(0.005, max_grad/slew_rate, slew_rate)

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

class VelocityEncodingFactory:
	def __init__(self, ltik: LTIGradientKernels, sl: SafetyLimits, print_calc=False):
		self.system = ltik.system
		self.ltik = ltik
		self.sl = sl
	
	def get_gradients(self, velocity_vector):
		
		max_grad = self.system.max_grad * self.sl.grad_ratio
		max_slew = self.system.max_slew * self.sl.slew_ratio
		
		grad_lengths = [calc_T_DT(
								velocity_vector[i], 
								max_grad, 
								max_slew, 
								self.system.gamma
							) for i in enumerate(velocity_vector)
						]   

		max_grad_length = max([gradlen[0] + 2*gradlen[1] for gradlen in grad_lengths])
		
		zerovec = np.zeros((np.ceil(max_grad_length / self.system.grad_raster_time),))

		channels = ['x', 'y', 'z']
		for i in range(len(velocity_vector)):
			T, DT = grad_lengths[i]
			temp_wave = pp.points_to_waveform(
				np.array([0.0, -max_slew*DT, -max_slew*DT, 0.0]),
				self.system.grad_raster_time,
				np.array([0.0, DT, T + DT, T + 2*DT])
			)

			vel_prephase = np.zeros_like(zerovec)
			



T, DT = calc_T_DT(0.001, 80, 200, 42.58e6)