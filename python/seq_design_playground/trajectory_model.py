import torch
import torch.nn as nn
import mrinufft as mn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traj_utils as tu

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

		t = torch.arange(self.nsamp, requires_grad=False, device=device)*grad_raster_time
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
				
			# Scale by smin for this axis and grad_raster_time
			# PNS = (Δt / S_min) * conv(|slew|, kernel)
			# The Δt factor accounts for discrete time integration
			convolved.append((self.grad_raster_time_stored / self.smin[i]) * conv_result)			# Stack along axis 1: (nshots, 3, time)
			R = torch.cat(convolved, dim=1)

			# Compute magnitude across all three axes
			R = torch.sqrt(torch.square(R).sum(dim=1))  # (nshots, time)

			return R

class TrajectoryModel(nn.Module):
	def __init__(self, 
			nshots, 
			nsamp, 
			kspace_guess, 
			kspace_edges=None,
			grad_edges=None,
			pns=None,
			device=torch.device('cpu'),
			resolution=torch.tensor([320,320,320]),
			fov = torch.tensor([0.2,0.2,0.2]),
			max_grad=1e-3*torch.tensor([60,60,60]), # T/m
			max_slew=180.0*torch.tensor([1,1,1]), # T/m/s
			gamma=42.576e6,
			grad_raster_time=4e-6
			):
		super().__init__()
		
		self.device = device
		self.resolution = resolution.to(device)
		self.fov = fov.to(device)
		self.max_grad = max_grad.to(device)
		self.max_slew = max_slew.to(device)
		self.gamma = gamma
		self.grad_raster_time = grad_raster_time

		self.delta_k = 1.0 / self.fov
		self.kspace_extent = self.resolution * self.delta_k
		with self.device:
			self.kspace_edges = kspace_edges
			if kspace_edges is None:
				self.kspace_edges = [torch.zeros(nshots,3), torch.zeros(nshots,3)]
			else:
				self.kspace_edges = [kspace_edges[i].to(device) for i in range(3)]
			if grad_edges is None:
				self.grad_edges = [torch.zeros(nshots,3), torch.zeros(nshots,3)]
			else:
				self.grad_edges = [grad_edges[i].to(device) for i in range(3)]
			self.nshots = nshots
			self.nsamp = nsamp
			self.kspace = torch.nn.Parameter(kspace_guess.to(device))
			self.pns = pns
			self.time = torch.arange(nsamp, requires_grad=False)*self.grad_raster_time


	def forward(self):
		return torch.concatenate([
			self.kspace_edges[0].unsqueeze(-1), 
			self.kspace, 
			self.kspace_edges[1].unsqueeze(-1)
		], axis=-1)

	def traj_to_grad(self, kspace):
		# Compute finite difference for gradients
		g = (1.0 / self.gamma) * (kspace[...,1:] - kspace[...,:-1]) / self.grad_raster_time
		# Compute the slew rate
		sr0 = (g[...,1:] - g[...,:-1]) / self.grad_raster_time

		# Gradient is now sampled between k-space points whilst the slew rate is between gradient points
		sr = torch.zeros(sr0.shape[:-1] + (sr0.shape[-1] + 1,), device=self.device)
		sr[...,0] = sr0[...,0]
		sr[...,1:-1] = 0.5 * (sr0[...,:-1] + sr0[...,1:])
		sr[...,-1] = sr0[...,-1]

		return g, sr

	def calc_grad_waveform(self, slew):
		return torch.cumulative_trapezoid(slew, axis=-1, dx=self.grad_raster_time)

	def calc_k_space(self, grad_waveform):
		return self.gamma*torch.cumulative_trapezoid(grad_waveform, axis=-1, dx=self.grad_raster_time)

	def calc_slew(self, grad_waveform):
		return torch.diff(grad_waveform, axis=-1) / self.grad_raster_time

	def kspace_spread_loss(self, k):
		loss = 0.0
		for i in range(k.shape[0]-1):
			loss += torch.sum(torch.square(self.time[-1]) / (1e-3 + torch.square(k[i] - k[(i+1):]).sum(dim=0)))
	
		return loss

	def kspace_mesh_loss(self, k, meshpoints):
		#kb = self.kspace_extent
		#ri = self.resolution // 20
		#kstep = kb // ri
		loss = 0.0
		#for xi in range(ri[0]):
		#	for yi in range(ri[1]):
		#		for zi in range(ri[2]):
		for i in range(meshpoints.shape[0]):
			kpoint = meshpoints[i,:].view(1,3,1)
			diff = torch.square(k - kpoint).sum(dim=1) / torch.square(kpoint).sum(dim=1)
			loss += 1e-3*torch.sum(diff)
		return loss

	def kspace_bound_loss(self, k, logt=1.0):
		kb = self.kspace_extent.view(1,3,1)
		perc = torch.square(k / kb)
		if logt < 1.0:
			return torch.sum(torch.pow(perc, 1.0 / logt)) / k.numel()
		else:
			p_1 = torch.pow(perc, 8)
			barrier = (-1.0/logt)*torch.log(1-p_1)
			return torch.sum(1+barrier) / k.numel()

	def grad_bound_loss(self, grad_waveform, logt=1.0):
		gb = self.max_grad.view(1,3,1)
		grad_perc = torch.square(grad_waveform / gb)
		if logt < 1.0:
			return torch.sum(torch.pow(grad_perc, 1.0 / logt)) / grad_waveform.numel()
		else:
			k1 = 1.0
			r1 = 1.05
			d = 80
			p_2 = k1*((r1*grad_perc) - torch.pow(r1*grad_perc, d))
			#print('Max Grad extent: ', grad_perc.max().item(), 'Mean Grad extent: ', grad_perc.mean().item())
			#print('p_2', p_2[0,0,250])
			barrier = (-1.0/logt)*torch.log(1+p_2)
			return torch.sum(1+barrier) / grad_waveform.numel()
	
	def slew_bound_loss(self, slew_waveform, logt=1.0):
		# Simple quadratic penalty for violations
		sb = self.max_slew.view(1,3,1)
		slew_mag = torch.abs(slew_waveform)
		violation = torch.maximum(slew_mag - sb, torch.zeros_like(slew_mag))
		penalty = torch.square(violation / sb)  # Normalized quadratic penalty
		return torch.mean(penalty)

	def pns_loss(self, slew_waveform, logt = 1.0):
		pnsval = self.pns(slew_waveform)
		
		# Use simple quadratic penalty for severe violations
		max_pns = self.pns.max_pns
		perc = pnsval / max_pns
		
		# For severe violations (>1), use simpler quadratic penalty
		# Barrier functions break down when constraints are badly violated
		violation = torch.maximum(perc - 1.0, torch.zeros_like(perc))
		penalty = torch.square(violation)  # Quadratic penalty on violations
		
		return torch.mean(penalty)

		# if logt < 1.0:
		# 	return torch.sum(torch.pow(pnsval, 1.0 / logt)) / slew_waveform.numel()
		# else:
		# 	k1 = 1.0
		# 	r1 = 1.05
		# 	d = 80
		# 	p_2 = k1*((r1*pnsval) - torch.pow(r1*pnsval, d))
		# 	barrier = (-1/logt)*torch.log(1+p_2)
		# 	return torch.sum(1+barrier) / slew_waveform.numel()
	
	def pns_variance_loss(self, slew_waveform):
		"""Encourage uniform PNS distribution - penalize variance in PNS values"""
		pnsval = self.pns(slew_waveform)
		# We want PNS to be more uniform across time
		pns_mean = pnsval.mean()
		pns_var = torch.mean(torch.square(pnsval - pns_mean))
		return pns_var
	
	def pns_peak_reduction_loss(self, slew_waveform, percentile=95):
		"""Specifically target the highest PNS values to reduce spikes"""
		pnsval = self.pns(slew_waveform)
		# Target top 5% of PNS values with strong penalty
		# This encourages reducing peaks rather than just variance
		threshold = torch.quantile(pnsval.flatten(), percentile / 100.0)
		high_pns = pnsval[pnsval > threshold]
		if high_pns.numel() > 0:
			return torch.mean(torch.square(high_pns / self.pns.max_pns))
		else:
			return torch.tensor(0.0, device=self.device)
	
	def kspace_smoothness_loss(self, kspace):
		"""Penalize sharp changes in k-space trajectory (encourages smooth curves)"""
		# Second derivative (curvature) penalty
		kspace_diff1 = kspace[..., 1:] - kspace[..., :-1]  # First derivative
		kspace_diff2 = kspace_diff1[..., 1:] - kspace_diff1[..., :-1]  # Second derivative
		return torch.mean(torch.square(kspace_diff2))
	
	def slew_smoothness_loss(self, slew_waveform):
		"""Penalize sharp changes in slew rate (encourages smooth slew)"""
		slew_diff = slew_waveform[..., 1:] - slew_waveform[..., :-1]
		return torch.mean(torch.square(slew_diff))
	
	def grad_edge_loss(self, grad_waveform):
		return torch.sum(torch.square(grad_waveform[:,:,0] - self.grad_edges[0]) + torch.square(grad_waveform[:,:,-1] - self.grad_edges[1]))


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from traj_design import Spiral3D
	from sequtil import SafetyLimits, LTIGradientKernels, ImageProperties
	import pypulseq as pp
	import numpy as np
	import scipy as sp
	from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
	import scipy.special as sps
	import math

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

	nshots = 20

	device = torch.device('cuda:0')

	tf = Spiral3D(ltik, sl, imgprop, print_calc=False)
	tfret = tf.get_gradients(nshots, spiral_settings, speed_interpolation=speed_interpolator)
	trajectory = tfret[0]
	# Add a downramp
	trajend = trajectory[:,:,-1][...,None]
	n_traj_downsample = 100
	trajend = trajend - np.arange(n_traj_downsample+1)[None,None,:] * trajend / n_traj_downsample
	trajectory = np.concatenate([trajectory, trajend], axis=-1)

	plt.figure()
	plt.plot(trajectory[19,0, :], 'r-*')
	plt.plot(trajectory[19,1, :], 'g-*')
	plt.plot(trajectory[19,2, :], 'b-*')
	plt.title('Trajectory')
	plt.show()

	trajectory = torch.tensor(trajectory, device=device)

	nsamp = trajectory.shape[-1]

	pns = PNS(nsamp=nsamp-2, grad_raster_time=system.grad_raster_time, max_pns=0.905, device=device)
	traj = TrajectoryModel(nshots, nsamp, kspace_guess=trajectory, pns=pns, device=device, grad_raster_time=system.grad_raster_time)

	optimizer = torch.optim.Adam(traj.parameters(), lr=1e-3)  # Increase learning rate to compensate for small gradients  # Use Adam with larger learning rate
	max_iter = 40000
	do_every_n = 5000
	for i in range(max_iter):
		if i % do_every_n == 0:
			meshpoints = traj.kspace_extent*torch.randn(500,3, device=device).clamp(-1,1)

		kspace = traj()

		# Don't scale for gradient/slew calculation - use original trajectory
		grad, slew = traj.traj_to_grad(kspace)

		#logt = 0.1 + 0.9*((i / max_iter - max_iter/2)/max_iter)**2 + (i / (2*max_iter))**2
		logt = 0.5
		#print('logt: ', logt)

		#kspace_spread_loss = 1e-2*traj.kspace_spread_loss(kspace)
		#mesh_loss = 1e-5*traj.kspace_mesh_loss(kspace, meshpoints)

		
		kspace_bound_loss = traj.kspace_bound_loss(kspace, logt=logt)
		
		pns_loss = 1e6 * traj.pns_loss(slew, logt=logt)  # Increased another 10x - PNS still 2x over limit
		
		# Add smoothness regularization to encourage smooth PNS redistribution
		pns_variance_loss = 1e4 * traj.pns_variance_loss(slew)  # Increased 2x
		pns_peak_loss = 5e4 * traj.pns_peak_reduction_loss(slew, percentile=90)  # Increased 5x, target top 10% instead of 5%
		kspace_smooth_loss = 1e2 * traj.kspace_smoothness_loss(kspace)  # Reduced - let PNS constraints dominate
		slew_smooth_loss = 0.0 * traj.slew_smoothness_loss(slew)  # Disabled - let slew vary naturally
		
		grad_loss = traj.grad_bound_loss(grad, logt=logt)
		slew_loss = 5e3 * traj.slew_bound_loss(slew, logt=logt)  # Increased 5x - still above 180 T/m/s limit

		#grad_edge_loss = traj.grad_edge_loss(grad)
		

		loss = pns_loss + pns_variance_loss + pns_peak_loss + kspace_smooth_loss + slew_smooth_loss + grad_loss + slew_loss + kspace_bound_loss

		if i % do_every_n == 0:
			plot_index = 19

			kspace_r = torch.square(kspace[plot_index,...]).detach().sum(dim=0).sqrt()
			kspace_r /= kspace_r.max()
			kspace_plot = torch.clone(kspace).detach().cpu().numpy()
			kspace_plot = 0.5 * kspace_plot / np.abs(kspace_plot).max()
			tu.show_trajectory(kspace_plot.transpose(0,2,1), 0, figure_size=15)

			plt.figure()
			plt.plot(kspace_plot[plot_index,0, :], 'r-*')
			plt.plot(kspace_plot[plot_index,1, :], 'g-*')
			plt.plot(kspace_plot[plot_index,2, :], 'b-*')
			plt.title('Trajectory')
			plt.show()

			

			pns_waveform = traj.pns(slew)
			pns_max = pns_waveform[plot_index,...].max().detach()
			kspace_r *= pns_max
			kspace_r = kspace_r.cpu().numpy()

			slew_r = torch.square(slew[plot_index,...]).detach().sum(dim=0).sqrt()
			slew_r *= (pns_max / slew_r.max())

			plt.figure()
			plt.plot(pns_waveform[plot_index,:].detach().cpu().numpy(), 'm-')
			plt.plot(kspace_r, 'k--')
			plt.plot(slew_r.cpu().numpy(), 'b-')
			plt.title('PNS waveform')
			plt.show()

			print(
				'Iter %: ', f"{100*(i / max_iter):.1f}",
				' loss: ', f"{loss.item():.3e}",
				' PNS loss:', f"{pns_loss.item():.3e}",
				' PNS var:', f"{pns_variance_loss.item():.3e}",
				' PNS peak:', f"{pns_peak_loss.item():.3e}",
			)
			print(
				' K-smooth:', f"{kspace_smooth_loss.item():.3e}",
				' K-bound:', f"{kspace_bound_loss.item():.3e}",
				' Grad loss:', f"{grad_loss.item():.3e}",
				' Slew loss:', f"{slew_loss.item():.3e}",
			)
			pns_vals = pns(slew)
			
			# Show PNS distribution across trajectory segments
			n_samples = pns_vals.shape[-1]
			pns_early = pns_vals[..., :n_samples//3].max()  # First third
			pns_mid = pns_vals[..., n_samples//3:2*n_samples//3].max()  # Middle third
			pns_late = pns_vals[..., 2*n_samples//3:].max()  # Last third
			
			slew_mag = torch.sqrt(torch.square(slew).sum(dim=1))  # Magnitude across axes
			slew_early = slew_mag[..., :n_samples//3].max()
			slew_mid = slew_mag[..., n_samples//3:2*n_samples//3].max()
			slew_late = slew_mag[..., 2*n_samples//3:].max()
			
			print(
				'Max PNS: ', f"{pns_vals.max().item():.3e}",
				' Mean PNS: ', f"{pns_vals.mean().item():.3e}",
				' PNS > max:', f"{(pns_vals > pns.max_pns).sum().item()}",
			)
			print(
				' PNS early/mid/late:', f"{pns_early.item():.3f}/{pns_mid.item():.3f}/{pns_late.item():.3f}",
				' Slew early/mid/late:', f"{slew_early.item():.0f}/{slew_mid.item():.0f}/{slew_late.item():.0f}"
			)
			print(
				' Max Grad: ', f"{torch.abs(grad).max().item():.3e}",
				' Mean Grad: ', f"{torch.abs(grad).mean().item():.3e}",
				' Max Slew: ', f"{torch.abs(slew).max().item():.3e}",
				' Mean Slew: ', f"{torch.abs(slew).mean().item():.3e}"
			)
			print('')

		optimizer.zero_grad()
		loss.backward()
		
		# Check gradient flow
		if i % do_every_n == 0:
			# Save old params
			old_kspace = traj.kspace.data.clone()
			
		# Check gradient flow
		if i % do_every_n == 0:
			for name, param in traj.named_parameters():
				if param.grad is not None:
					print(f"{name} grad: mean={param.grad.abs().mean().item():.6e}, max={param.grad.abs().max().item():.6e}")
				else:
					print(f"{name} grad: None")
		
		# Clip gradients to prevent instability (may be limiting updates - try without)
		#torch.nn.utils.clip_grad_norm_(traj.parameters(), max_norm=1.0)
		
		optimizer.step()
		
		# Check parameter changes
		if i % do_every_n == 0:
			param_change = (traj.kspace.data - old_kspace).abs()
			print(f"Param change: mean={param_change.mean().item():.6e}, max={param_change.max().item():.6e}")
			print('')
		#with torch.no_grad():
			#for param in traj.parameters():
				#param[:,0].clamp_(-traj.max_slew[0], traj.max_slew[0])
				#param[:,1].clamp_(-traj.max_slew[1], traj.max_slew[1])
				#param[:,2].clamp_(-traj.max_slew[2], traj.max_slew[2])

	kspace = traj()
	kspace = 0.5 * kspace / torch.abs(kspace).max()
	kspace = kspace.cpu().detach().numpy()
	tu.show_trajectory(kspace.transpose(0,2,1), 0, figure_size=8)