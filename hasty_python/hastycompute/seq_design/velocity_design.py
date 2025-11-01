import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hastycompute.seq_design.gradient_design as gd
import hastycompute.utils.torch_utils as torch_utils

from hastycompute.seq_design.sequtil import SafetyLimits, LTIGradientKernels, ImageProperties, convert


def first_moment(T, DT, slew_rate):
	return DT*slew_rate*(2*torch.square(DT) + 3*T*DT + torch.square(T))

# max_grad [mT/m], slew_rate [T/m/s]
def calc_T_DT(venc, max_grad, slew_rate, gamma, tot_T=None):
	max_grad_T = max_grad * 1e-3
	
	DM = torch.pi/(gamma*venc)
	#DM = first_moment(0.005, max_grad/slew_rate, slew_rate)

	if tot_T is None:
		# T=0 gradients =>
		DT = (DM / (2 * slew_rate)) ** (1.0 / 3.0)

		if DT * slew_rate > max_grad_T:
			DT = max_grad_T / slew_rate

			root = 0.25 * DT*DT + DM/(slew_rate*DT)
			T = -(3/2)*DT + math.sqrt(root)
			#DM_back = first_moment(T, DT, slew_rate)
			return T, DT
		else:
			return 0.0, DT
	else:
		# For solutions see the SymPy script at vel_maths.ipynb

		L = float(tot_T)

		T = (1/2)*math.sqrt(-32*DM/(L*slew_rate) + L**2)
		DT = (L - 2*T) / 4

		if slew_rate * DT > max_grad:
			raise RuntimeError("Can't calculate T,DT from total time given this slew \
					  max gradient is exceeded, increase total time, slew_rate or max_grad")
		
		if abs(L - (2*T + 4*DT)) / abs(L) > 1e-9:
			raise RuntimeError("Large difference between total time and calculated T,DT")

		return T, DT



class VelocityEncodingFactory:
	def __init__(self, gk: gd.GradientKernels, sl: SafetyLimits, smooth_kernel=None):
		self.gk = gk
		self.system = gk.system
		self.sl = sl
		self.smooth_kernel = smooth_kernel

	def get_gradients(self, velocity_vector, channels=['x', 'y', 'z']):
		device = velocity_vector.device

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
		NGRT = math.ceil(L/GRT)+1
		t = torch.arange(0, NGRT)*GRT

		if self.smooth_kernel is None:
			self.smooth_kernel = torch.flip(torch.linspace(0,1,NGRT//8, dtype=torch.float64, device=device), dims=[0])
			self.smooth_kernel = torch.exp(-torch.square(self.smooth_kernel)/0.05)
			self.smooth_kernel /= torch.sum(self.smooth_kernel)
		smooth_kernel_length = self.smooth_kernel.shape[0]

		grad_waves = []
		grad_properties = []

		for i in range(len(velocity_vector)):
			
			if velocity_vector[i] is None or velocity_vector[i] > 1e4:

				grad_wave = torch.zeros((t.shape[0] + smooth_kernel_length - 1 + self.gk.max_kernel_length,), dtype=torch.float64, device=device)

				# if self.print_calc:
				# 	print('Velocity is None or too high, setting to zero')
				# 	print('Zeroth order moment: ', 0, ' First order moment: ', 0, 'Second order moment: ', 0, ' Venc: infty')

				grad_waves.append(grad_wave)
				grad_properties.append((0.0, 0.0, 0.0, math.inf))
			else:

				T, DT = calc_T_DT(
							abs(velocity_vector[i]), 
							convert(max_grad, from_unit='Hz/m', to_unit='mT/m'), 
							convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s'), 
							self.system.gamma, 
							L
						)

				Gmax = 1e3*convert(max_slew, from_unit='Hz/m/s', to_unit='T/m/s')*DT

				if velocity_vector[i] > 0:
					grad_blip = torch.tensor([0, -Gmax, -Gmax, 0, Gmax, Gmax, 0, 0], dtype=torch.float64, device=device)
				else:
					grad_blip = torch.tensor([0, Gmax, Gmax, 0, -Gmax, -Gmax, 0, 0], dtype=torch.float64, device=device)


				grad_wave = torch_utils.interp(
								t, 
								torch.tensor([
									0, DT, T+DT, T+2*DT, T+3*DT, 2*T+3*DT, 2*T+4*DT, 2*T+4*DT+GRT
								], dtype=torch.float64, device=device), 
								grad_blip
							)

				grad_wave = torch_utils.lagged_convolve(grad_wave, self.smooth_kernel)

				grad_wave = torch.cat([grad_wave, torch.zeros((self.gk.max_kernel_length,), dtype=torch.float64, device=device)])

				grad_wave_up = gd.Gradient.calculate_actual_waveform(grad_wave.expand(1,3,-1), self.gk)[0,0,:]

				DM0, DM1, DM2 = gd.Gradient.calculate_specific_moments(
									grad_wave_up.expand(1,1,-1), 
									self.system.grad_raster_time / self.gk.oversamp(), 
									torch.tensor([grad_wave_up.shape[0]-1], dtype=torch.int64, device=device)
								)
				DM0, DM1, DM2 = DM0[0].squeeze(), DM1[0].squeeze(), DM2[0].squeeze()

				venc = torch.pi / (self.system.gamma * DM1)

				grad_waves.append(grad_wave)
				grad_properties.append((DM0, DM1, DM2, venc))

		return grad_waves, grad_properties

if __name__ == "__main__":

	import pypulseq as pp

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

	gk = gd.GradientKernels(
			system,
			gd.GradientKernels.kernels_from_test(system.grad_raster_time, kernel_oversampling=20),
			kernel_oversampling=20
	)

	vencs = [None, 		 None, 		 None] 		 	+ \
			[0.8*(-1.0), 0.8*(-1.0), 0.8*(-1.0)]

	channels = ['x', 'y', 'z'] * 2

	vef = VelocityEncodingFactory(gk, SafetyLimits())
	vel_grads = vef.get_gradients(vencs, channels)


	print()