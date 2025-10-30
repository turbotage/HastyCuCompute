import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pypulseq as pp
from sequtil import convert

from collections import OrderedDict

class GradientKernels:
	def __init__(self, system, kernels, kernel_oversampling=20):
		self.system = system
		self.kernels = kernels
		self.kernel_oversampling = kernel_oversampling

		self.max_kernel_length = max([math.ceil(kernel.shape[0]/kernel_oversampling) for kernel in kernels.values()])

	@staticmethod
	def kernels_from_test(grad_raster_time, kernel_oversampling=100, device=torch.device('cpu')):
		k1 = 8.0
		k2 = 1.5
		Dk = 2.0 / grad_raster_time

		n_grad_raster_times = 6
		t = torch.linspace(0, n_grad_raster_times*grad_raster_time, 
						   n_grad_raster_times*kernel_oversampling, device=device)
		kernel = (Dk*t)**(k1) * torch.exp(-k2*Dk*t)
		kernel /= torch.sum(kernel)

		kernels = {'x': kernel, 'y': kernel, 'z': kernel}

		return kernels
	
	def get(self, channel='x'):
		return self.kernels[channel]
	
	def oversamp(self):
		return self.kernel_oversampling

class GradientSegment(nn.Module):
	def __init__(self):
		super().__init__()
	def output_shape(self):
		raise NotImplementedError("Output size method not implemented.")
	def forward(self):
		raise NotImplementedError("Forward method not implemented.")

class GradientTimeDynamicSegment(GradientSegment):
	def __init__(self, NGRT: int):
		super().__init__()
		self.NGRT = NGRT
	def output_shape(self):
		raise NotImplementedError("Output size method not implemented.")
	def forward(self):
		raise NotImplementedError("Forward method not implemented.")
	def set_number_of_timepoints(self, NGRT):
		self.NGRT = NGRT

class GradientScaledTimeDynamicSegment(GradientTimeDynamicSegment):
	def __init__(self, inner_gs: GradientTimeDynamicSegment, scale: torch.Tensor, NGRT: int):
		super().__init__(NGRT)
		self.inner_gs = inner_gs
		self.scale = scale
	
	def output_shape(self):
		return self.inner_gs.output_shape()
	
	def set_scale(self, scale: torch.Tensor):
		self.scale = scale

	def get_scale(self) -> torch.Tensor:
		return self.scale

	def set_number_of_timepoints(self, NGRT):
		self.NGRT = NGRT
		self.inner_gs.set_number_of_timepoints(NGRT)

	def get_number_of_timepoints(self) -> int:
		return self.NGRT
	
	def forward(self):
		return self.scale.view(1,3,1) * self.inner_gs.forward()

class GradientStaticSegment(GradientSegment):
	def __init__(self, initial_gradwave):
		super().__init__()
		self.static_gradwave = initial_gradwave
	
	def output_shape(self):
		return self.static_gradwave.shape

	def forward(self):
		return self.static_gradwave
	
class GradientFreeSegment(GradientSegment):
	def __init__(self, initial_gradwave):
		super().__init__()
		self.free_gradwave = nn.Parameter(initial_gradwave)
	
	def output_shape(self):
		return self.free_gradwave.shape

	def forward(self):
		return self.free_gradwave

class Gradient(nn.Module):
	def __init__(self, gk: GradientKernels, device: torch.device, gradient_segments: OrderedDict[str, GradientSegment]):
		super().__init__()
		self.gk = gk
		self.device = device
		
		self.segments = nn.ModuleDict(gradient_segments)

		self.NGRT = 0
		self.end_of_segment_timepoints = []
		self.max_batch = 0
		for name, gradseg in gradient_segments.items():
			seg_shape = gradseg.output_shape()
			if len(seg_shape) != 3:
				raise ValueError("Gradient segment output shape must be of length 3 (B, 3, N).")
			if seg_shape[1] != 3:
				raise ValueError("Gradient segment output shape second dimension must be 3.")
			self.NGRT += seg_shape[-1]
			self.end_of_segment_timepoints.append(max(self.NGRT - 1, 0))
			max_batch = seg_shape[0]
			if max_batch > self.max_batch:
				self.max_batch = max_batch

	def get_gradient_kernels(self):
		return self.gk
	
	def get_system(self):
		return self.gk.system

	def end_of_segments(self):
		return self.end_of_segment_timepoints

	def forward(self):
		cat_list = []
		zcat = torch.zeros((self.max_batch, 3, 1), device=self.device)
		cat_list.append(zcat)
		for name, gradseg in self.segments.items():
			grad_wave = gradseg.forward()
			# If batch size is 1, expand to max_batch
			if grad_wave.shape[0] == 1 and self.max_batch > 1:
				grad_wave = grad_wave.expand(self.max_batch, -1, -1)
			cat_list.append(grad_wave)
		cat_list.append(zcat)
		waveform = torch.cat(cat_list, dim=-1)  # [B, 3, NGRT]
		return waveform
	
	@staticmethod
	def calculate_kspace_traj(waveform, grad_raster_time, gamma):
		# waveform: [B, 3, N]
		ktraj = (1e-3 * gamma) * torch.cumulative_trapezoid(
						waveform, dim=-1, dx=grad_raster_time)  # in m^-1
		return ktraj

	@staticmethod
	def calculate_slew_rate(waveform, grad_raster_time):
		return (waveform[:,:,1:] - waveform[:,:,:-1]) / (1e3 * grad_raster_time) # in T/m/s

	@staticmethod
	def calculate_actual_waveform(waveform, gk: GradientKernels):
		# waveform: [1, 3, N]
		N = waveform.shape[-1]
		oversamp = gk.oversamp()
		NO = N * oversamp

		# 1. Linear interpolate
		wf_up = F.interpolate(waveform, size=NO, mode="linear", align_corners=False)

		# 2. Prepare kernels
		kx = gk.get("x").to(waveform.device, waveform.dtype)
		ky = gk.get("y").to(waveform.device, waveform.dtype)
		kz = gk.get("z").to(waveform.device, waveform.dtype)
		kernels = torch.stack([kx, ky, kz])[:, None, :]  # [3,1,L]
		kernel_length = kernels.shape[-1]

		# 3. Convolve (per-axis)
		kernels_flipped = torch.flip(kernels, dims=[-1])
		wf_conv = F.pad(wf_up, (kernel_length-1, kernel_length-1))
		wf_conv = F.conv1d(wf_conv, kernels_flipped, padding=0, groups=3)

		if False:
			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(wf_up[0,0,:].detach().cpu().numpy(), label='Ups')
			plt.plot(wf_conv[0,0,:].detach().cpu().numpy(), label='Conv')
			plt.legend()
			plt.show()

		return wf_conv


	# Relationship between DMX and DMX_Lists is that DMX[:,:,points[k]-2] == DMX_List[k]
	@staticmethod
	def calculate_moments(waveform, grad_raster_time, start_time=0.0):
		"""
		Calculate zeroth, first, and second order moments of a gradient waveform.

		Parameters:
		waveform (torch.Tensor): Gradient waveform of shape (3, N). (mT/m)
		grad_raster_time (float): Time interval between gradient samples.

		Returns:
		tuple: Zeroth, first, and second order moments as torch.Tensors.
		"""
		N = waveform.shape[-1]
		t = torch.arange(N, device=waveform.device) * grad_raster_time

		DM0 = 1e-3*torch.cumulative_trapezoid(waveform, dim=-1, dx=grad_raster_time)
		DM1 = 1e-3*torch.cumulative_trapezoid(waveform * (t+start_time), dim=-1, dx=grad_raster_time)
		DM2 = 1e-3*torch.cumulative_trapezoid(waveform * torch.square(t+start_time), dim=-1, dx=grad_raster_time)

		return DM0, DM1, DM2
	
	# Relationship between DMX and DMX_Lists is that DMX[:,:,points[k]-2] == DMX_List[k]
	@staticmethod
	def calculate_specific_moments(waveform, grad_raster_time, points, start_time=0.0):
		if points.max() >= waveform.shape[-1]:
			raise ValueError("Points exceed waveform length.")
		points = torch.sort(points).values

		t0 = start_time
		DM0_list = []
		DM1_list = []
		DM2_list = []

		seg = waveform[:,:,0:points[0]]
		N = seg.shape[-1]
		t = torch.arange(N, device=waveform.device) * grad_raster_time + t0

		DM0 = 1e-3*torch.trapz(seg, dim=-1, dx=grad_raster_time)
		DM1 = 1e-3*torch.trapz(seg * t, dim=-1, dx=grad_raster_time)
		DM2 = 1e-3*torch.trapz(seg * torch.square(t), dim=-1, dx=grad_raster_time)

		t0 += N * grad_raster_time

		DM0_list.append(DM0)
		DM1_list.append(DM1)
		DM2_list.append(DM2)

		for i in range(points.shape[0]-1):
			seg = waveform[:,:,points[i]:points[i+1]]
			N = seg.shape[-1]
			t = torch.arange(N, device=waveform.device) * grad_raster_time + t0

			DM0 = 1e-3*torch.trapz(seg, dim=-1, dx=grad_raster_time)
			DM1 = 1e-3*torch.trapz(seg * t, dim=-1, dx=grad_raster_time)
			DM2 = 1e-3*torch.trapz(seg * torch.square(t), dim=-1, dx=grad_raster_time)

			DM0_list.append(DM0)
			DM1_list.append(DM1)
			DM2_list.append(DM2)

			t0 += N * grad_raster_time

		return DM0_list, DM1_list, DM2_list

if __name__ == "__main__":
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

	NGRT = 500

	grad_segments = []

	seg = torch.sin(2*torch.pi*torch.linspace(0, 1, NGRT)).view(1,1,NGRT).expand(-1,3,-1).clone()
	grad_segments.append((seg, True))
	seg = torch.sin(2*torch.pi*torch.linspace(0, 1, 2*NGRT)).view(1,1,2*NGRT).expand(10,3,-1).clone()
	grad_segments.append((seg, False))
	seg = torch.sin(2*torch.pi*torch.linspace(0, 1, NGRT)).view(1,1,NGRT).expand(-1,3,-1).clone()
	grad_segments.append((seg, True))
	
	kernel_os = 20
	grad = Gradient(
				GradientKernels(
					system,
					GradientKernels.kernels_from_test(
						system.grad_raster_time, kernel_oversampling=kernel_os
					),
					kernel_oversampling=kernel_os
				),
				device=torch.device('cpu'),
				gradient_segments=grad_segments
			)
	
	waveform = grad.forward()

	optimizer = torch.optim.AdamW(grad.parameters(), lr=1.0)

	for iter in range(10000):

		waveform = grad.forward()

		waveform_up = Gradient.calculate_actual_waveform(grad)

		DM0_list, DM1_list, DM2_list = Gradient.calculate_specific_moments(
										waveform_up,
										grad_raster_time=system.grad_raster_time / kernel_os,
										points=torch.tensor([NGRT*kernel_os-1, 4*NGRT*kernel_os-1])
									)
		# For M0 we wan't zero M0 after velocity encoding and 4pi dephasing after spoiler
		M0_loss = torch.sum(torch.square(DM0_list[0])) + torch.sum(torch.square(DM0_list[1] - 2))
		# For M1 we wan't venc M1 after velocity encoding and zero M1 after spoiler
		M1_loss = torch.sum(torch.square(DM1_list[0] - 2)) + torch.sum(torch.square(DM1_list[1]))

		loss = M0_loss + M1_loss

		print("Loss: ", loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if iter % 1000 == 0:
			waveform = grad.forward()
			waveform_up = Gradient.calculate_actual_waveform(grad)

			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(waveform_up[0,0,:].detach().cpu().numpy(), label='X')
			plt.plot(waveform_up[0,1,:].detach().cpu().numpy(), label='Y')
			plt.plot(waveform_up[0,2,:].detach().cpu().numpy(), label='Z')
			plt.legend()
			plt.show()

	# DM0, DM1, DM2 = Gradient.calculate_moments(
	# 					wf_conv, 
	# 					grad_raster_time=system.grad_raster_time
	# 				)
	
	# DM0_list, DM1_list, DM2_list = Gradient.calculate_specific_moments(
	# 									wf_conv,
	# 									grad_raster_time=system.grad_raster_time,
	# 									points=torch.tensor([500, 1500])
	# 								)


	
	print()