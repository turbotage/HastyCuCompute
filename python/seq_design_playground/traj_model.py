import torch
from torch import nn
import mrinufft as mn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traj_utils as tu
import math
import pypulseq as pp
from sequtil import SafetyLimits, LTIGradientKernels, ImageProperties
import numpy as np

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

			t = torch.arange(self.nsamp, requires_grad=False)*grad_raster_time
			self.pns_kernel = self.chronaxie.unsqueeze(-1) / torch.square(self.chronaxie.unsqueeze(-1) + t)
			self.pns_kernel = self.pns_kernel / self.pns_kernel.sum(axis=-1, keepdim=True)
			self.pns_kernel.requires_grad = False
			self.max_pns = max_pns

	def __call__(self, slew_waveform):
		with self.device:
			R = (1.0 / self.smin.view(1,3,1)) * torch.stack([torch.conv1d(
								slew_waveform[:,i,:].unsqueeze(1), 
								self.pns_kernel[i,:].view(1,1,self.nsamp),
								padding='same'
							) for i in range(3)], axis=1)

			R = torch.sqrt(torch.square(R).sum(axis=1))

			return R




def target_distribution(x, t, mean, std, tmax):
	r = torch.square(x).sum(dim=0).sqrt()
	meant = mean(t)
	stdt = std(t)

	const = 3*math.sqrt(2) / (4*(math.pi**(5/2))*(tmax**3))

	return const * torch.exp(-torch.square((r - meant) / (math.sqrt(2)*torch.square(stdt)))) / stdt

def target_score(x, mean, std, meandt, stddt):
	c = x[0:3,:]
	t = x[3,:]

	r = (torch.square(c).sum(dim=0) + 1e-5).sqrt() + 1e-5
	meant = mean(t)
	stdt = std(t)
	stdt2 = torch.square(stdt)
	meandtt = meandt(t)
	stddtt = stddt(t)

	gx = (c[0,...] / stdt2) * (1.0 - meant/r)
	#gx[gx != gx] = 0.0
	gy = (c[1,...] / stdt2) * (1.0 - meant/r)
	#gy[gy != gy] = 0.0
	gz = (c[2,...] / stdt2) * (1.0 - meant/r)
	#gz[gz != gz] = 0.0
	gt = ((r - meant)*stddtt + stdt*meandtt) * (r - meant)
	gt += stdt2 * stddtt
	gt /= stdt2 * stdt
	gt = gt.neg()

	return torch.stack([gx,gy,gz,gt], axis=0)

def hutchinson_trace_hessian(f, x, num_samples=10):
	"""Vectorized Hutchinson's trick to approximate Tr(Hessian)."""
	#dim, batch_size = x.shape

	trace_estimate = 0.0

	for _ in range(num_samples):
		v = torch.rand_like(x)
		# Compute Hessian-vector products in parallel
		Hv = torch.autograd.grad(f(x), x, grad_outputs=v, create_graph=True)[0]
		trace_estimate += ((Hv * v).sum(dim=-1))

	trace_estimate /= num_samples

	return trace_estimate

def stein_score_matching_loss_fast(xsamp, mean, std, meandt, stddt, num_hutchinson_samples=10):
	"""Efficient Stein Score Matching loss using vectorized Hutchinson’s trick."""

	xsamp.requires_grad_(True)
	score_x = target_score(xsamp, mean, std, meandt, stddt)  # Compute ∇_x log p_target(x)
	
	# Compute trace term using Hutchinson's trick
	trace_term = hutchinson_trace_hessian(lambda x: target_score(x, mean, std, meandt, stddt), 
					xsamp, num_samples=num_hutchinson_samples)

	return (trace_term + 0.5 * (score_x ** 2).sum(dim=-1)).mean()

class TrajectoryModel(nn.Module):
	def __init__(self, 
			nshots,
			nsamp,
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
		
		self.pns = pns
		self.device = device
		self.resolution = resolution.to(device)
		self.fov = fov.to(device)
		self.max_grad = max_grad.to(device)
		self.max_slew = max_slew.to(device)
		self.gamma = gamma
		self.grad_raster_time = grad_raster_time

		self.delta_k = 1.0 / self.fov
		self.kspace_extent = 0.5 * self.resolution * self.delta_k
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
			self.tmax = nsamp*grad_raster_time
			self.times = self.grad_raster_time*torch.arange(self.nsamp)

			self.kspace_freq = torch.nn.Parameter(torch.rand((3,2,nshots)))
			self.kspace_amp = torch.nn.Parameter(torch.rand((3,2,nshots)))

	def all_times(self):
		return self.times.repeat(self.nshots,1)

	def forward(self):
		with self.device:
			kspace = self.kspace_amp.unsqueeze(-1) * torch.sin(
							self.kspace_freq.unsqueeze(-1)*self.times.view(1,1,1,self.nsamp)/self.tmax)
			kspace = kspace.sum(axis=1)

			return kspace

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
	
	def pns_loss(self, slew_waveform, logt = 1.0):
		pnsval = self.pns(slew_waveform)
		if logt < 1.0:
			return torch.sum(torch.pow(pnsval, 1.0 / logt)) / slew_waveform.numel()
		else:
			k1 = 1.0
			r1 = 1.05
			d = 80
			p_2 = k1*((r1*pnsval) - torch.pow(r1*pnsval, d))
			barrier = (-1/logt)*torch.log(1+p_2)
			return torch.sum(1+barrier) / slew_waveform.numel()


if __name__ == "__main__":

	device = torch.device('cuda:0')
	with device:
		res = 128
		fov = [0.2,0.2,0.2]
		delta_k = 1.0 / torch.tensor(fov)
		# coord = torch.stack(torch.where(torch.ones((res,res,res))),dim=-1).to(torch.float32)
		# coord -= res//2
		# t = torch.linspace(0,0.08, 32)
		tmax = 0.08
		mean_slope = 0.5*res / (tmax*fov)

		mean = lambda tarr: mean_slope*tarr
		std = lambda tarr: 2*torch.ones_like(tarr)
		meandt = lambda tarr: mean_slope*torch.ones_like(tarr)
		stddt = lambda tarr: torch.zeros_like(tarr)

		# dists = torch.empty((t.shape[0], res, res, res), device=device)
		# scores = torch.empty((4, t.shape[0], res, res, res), device=device)

		# x = torch.concatenate([coord, torch.ones_like(coord[:,0]).unsqueeze(-1)*t[0]], dim=-1)

		#loss = stein_score_matching_loss_fast(x, mean, std, meandt, stddt)
		nsamp = 10000

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
		#imgprop = ImageProperties([320,320,320], 
		#			np.array([220e-3, 220e-3, 220e-3]), np.array([320,320,320]))


		pns = PNS(
			nsamp=nsamp,
			grad_raster_time=system.grad_raster_time, 
			max_pns=0.905, 
			device=device
		)

		tm = TrajectoryModel(
			nshots=32,
			nsamp=nsamp, 
			kspace_edges=None, 
			grad_edges=None, 
			pns=pns, 
			device=torch.device('cuda:0'),
			resolution=torch.tensor([res,res,res]),
			fov=torch.tensor([0.2,0.2,0.2]),
			grad_raster_time=system.grad_raster_time
		)


		optimizer = torch.optim.SGD(tm.parameters(), lr=1e-9, momentum=0.9)
		for i in range(1000):

			kspace = tm()

			times = tm.all_times()

			if i % 100 == 0:
				to_plot = kspace[:,0:1,::100].permute(1,2,0)
				to_plot = 0.5 * to_plot / to_plot.abs().max()
				tu.show_trajectory(to_plot.detach().cpu().numpy(), 0, 8)

			x = torch.cat(	[
								kspace.view(3, kspace.shape[1]*kspace.shape[2]), 
								times.view(1, times.shape[0]*times.shape[1])
							], dim=0)

			stein_loss = stein_score_matching_loss_fast(x, mean, std, meandt, stddt)

			grad, slew = tm.traj_to_grad(kspace)

			pns_loss = 1e10*tm.pns_loss(slew)

			loss = stein_loss + pns_loss

			print('Stein Loss: ', stein_loss.item(), ' PNS Loss: ', pns_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# coord = torch.stack(torch.where(torch.ones((res,res,res))),dim=0).to(torch.float32)
		# coord -= res//2
		# t = torch.linspace(0,0.08, 32)
		# dists = torch.empty((t.shape[0], res, res, res), device=device)
		# scores = torch.empty((4, t.shape[0], res, res, res), device=device)

		# for i in range(t.shape[0]):
		# 	dist = target_distribution(coord, t[i], mean, std, t[-1])
		# 	dists[i] = dist.reshape(res,res,res)

		# 	x = torch.cat([coord, t[i]*torch.ones_like(coord[0]).unsqueeze(0)], dim=0)

		# 	score = target_score(x, mean, std, meandt, stddt)
		# 	scores[0,i] = score[0,...].reshape(res,res,res)
		# 	scores[1,i] = score[1,...].reshape(res,res,res)
		# 	scores[2,i] = score[2,...].reshape(res,res,res)
		# 	scores[3,i] = score[3,...].reshape(res,res,res)

		# #Compute the loss

		# import orthoslicer as ort
		# ort.image_nd(scores[0].cpu().numpy())
		# ort.image_nd(scores[1].cpu().numpy())
		# ort.image_nd(scores[2].cpu().numpy())
		# ort.image_nd(scores[3].cpu().numpy())
		# #ort.image_nd(scores.cpu().numpy())

		# ort.image_nd(dists.cpu().numpy())
		
