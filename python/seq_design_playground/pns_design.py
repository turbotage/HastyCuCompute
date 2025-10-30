import torch
import torch.nn as nn

class PNS:
	def __init__(self,
			nsamp,
			grad_raster_time=4e-6,
			chronaxie=334e-6*torch.ones(3, requires_grad=False), # s
			smin=70.27*torch.ones(3, requires_grad=False), # T/m/s
			max_pns = 1.0,
			device = torch.device('cpu')
		):
		self.device = device
		with self.device:
			self.nsamp = nsamp
			self.grad_raster_time = grad_raster_time
			self.chronaxie = chronaxie.to(device)
			self.smin = smin.to(device)

		t = torch.arange(self.nsamp, requires_grad=False, device=device, dtype=torch.float64)*grad_raster_time
		# PNS kernel: chronaxie / (chronaxie + t)^2
		# No normalization - kernel used as-is per literature
		self.pns_kernel = self.chronaxie.unsqueeze(-1) / torch.square(t + self.chronaxie.unsqueeze(-1))
		self.pns_kernel.requires_grad = False
		# Store raster time for scaling during convolution
		self.grad_raster_time_stored = grad_raster_time
		self.max_pns = max_pns

	def __call__(self, slew_waveform):
		with self.device:
			# PNS model: convolve each axis separately, then take magnitude
			# slew_waveform shape: (nshots, 3, time)
			
			# Convolve each axis separately
			convolved = []
			for i in range(3):
				# Extract slew for this axis: (nshots, 1, time)
				slew_i = slew_waveform[:,i,:].unsqueeze(1)
				
				# Take absolute value (magnitude of slew on this axis)
				slew_i_abs = torch.abs(slew_i)
				
				# For causal convolution: pad LEFT only (past values)
				pad_left = self.nsamp - 1
				slew_i_padded = torch.nn.functional.pad(slew_i_abs, (pad_left, 0), mode='constant', value=0)
				
				# NOTE: conv1d FLIPS the kernel, but we want correlation (no flip)
				# So we need to flip the kernel before conv1d to cancel out the flip
				kernel_flipped = torch.flip(self.pns_kernel[i,:], dims=[0]).view(1,1,self.nsamp)
				
				# Convolve absolute slew with PNS kernel (not normalized - per literature)
				conv_result = torch.nn.functional.conv1d(
					slew_i_padded, 
					kernel_flipped,
					padding=0
				)
				# In case the convolution produced small negative values due to numerical issues, clamp to zero
				# conv_result = torch.clamp(conv_result, min=0.0)
				# Scale by smin for this axis and grad_raster_time
				# PNS = (Δt / S_min) * conv(|slew|, kernel)
				# The Δt factor accounts for discrete time integration
				convolved.append((self.grad_raster_time_stored / self.smin[i]) * conv_result)
						
			# Stack along axis 1: (nshots, 3, time)
			R = torch.cat(convolved, dim=1)

			# Compute magnitude across all three axes
			R = torch.sqrt(torch.square(R).sum(dim=1) + 1e-9)  # (nshots, time)

			return R