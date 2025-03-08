import torch
import math
import rot

import spiral_trajectory as sptraj

def traj_to_grad(kspace, GRT, gamma=42.576e6):
	with kspace.device:
		# Compute finite difference for gradients
		g = (1.0 / gamma) * (kspace[...,1:] - kspace[...,:-1]) / GRT
		# Compute the slew rate
		sr0 = (g[...,1:] - g[...,:-1]) / GRT

		# Gradient is now sampled between k-space points whilst the slew rate is between gradient points
		sr = torch.zeros(sr0.shape[:-1] + (sr0.shape[-1] + 1,))
		sr[...,0] = sr0[...,0]
		sr[...,1:-1] = 0.5 * (sr0[...,:-1] + sr0[...,1:])
		sr[...,-1] = sr0[...,-1]

		return g, sr
	
class Spiral3DProperties:
	def __init__(self,
			  		nsamp: int = 4000,
					nb_revs: torch.Tensor = torch.scalar_tensor(5.0), 
					nb_folds: torch.Tensor = torch.scalar_tensor(5.0), 
					nb_rots: torch.Tensor = torch.scalar_tensor(1.0), 
					theta: torch.Tensor = torch.scalar_tensor(0.0), 
					phi: torch.Tensor = torch.scalar_tensor(0.0), 
					radial_downramp_start: int | None = None,
					radial_downramp_stop: int | None = None,
					rev_downramp_start: int | None = None,
					fold_downramp_start: int | None = None,
					rot_downramp_start: int | None = None,
					decay_width: torch.Tensor = torch.scalar_tensor(0.25), 
				 ):
		
		self.nsamp = nsamp

		self.nb_revs = nb_revs
		self.nb_folds = nb_folds
		self.nb_rots = nb_rots

		self.theta = theta
		self.phi = phi

		# Radial
		if radial_downramp_start is None:
			self.radial_downramp_start = int(0.9*nsamp)
		else:
			self.radial_downramp_start = radial_downramp_start

		if radial_downramp_stop is None:
			self.radial_downramp_stop = int(0.99*nsamp)
		else:
			self.radial_downramp_stop = radial_downramp_stop

		if self.radial_downramp_stop < self.radial_downramp_start:
			raise ValueError('Radial downramp stop must be greater than radial downramp start')
		if self.radial_downramp_start >= nsamp:
			raise ValueError('Radial downramp start must be less than nsamp')
		if self.radial_downramp_stop >= nsamp:
			raise ValueError('Radial downramp stop must be less than nsamp')

		# Revolutions
		if rev_downramp_start is None:
			self.rev_downramp_start = int(0.95*self.radial_downramp_start)
		else:
			self.rev_downramp_start = rev_downramp_start
		self.rev_downramp_stop = nsamp-1

		if self.rev_downramp_start >= nsamp:
			raise ValueError('Rev downramp start must be less than nsamp')

		# Folds
		if fold_downramp_start is None:
			self.fold_downramp_start = int(0.95*self.rev_downramp_start)
		else:
			self.fold_downramp_start = fold_downramp_start
		self.fold_downramp_stop = nsamp-1

		if self.fold_downramp_start >= nsamp:
			raise ValueError('Fold downramp start must be less than nsamp')

		# Rotation
		if rot_downramp_start is None:
			self.rot_downramp_start = int(0.95*self.fold_downramp_start)
		else:
			self.rot_downramp_start = rot_downramp_start
		self.rot_downramp_stop = nsamp-1

		if self.rot_downramp_start >= nsamp:
			raise ValueError('Rot downramp start must be less than nsamp')

		self.decay_width = decay_width

class System:
	def __init__(self, max_slew: float, max_grad: float, grad_raster_time: float):
		self.max_slew = max_slew
		self.max_grad = max_grad
		self.grad_raster_time = grad_raster_time

class Spiral3D(torch.nn.Module):
	def __init__(self, nsamp: int, spp: Spiral3DProperties, system: System, device=torch.device('cpu')):
		with device:
		
			super().__init__()

			self.nsamp = nsamp
			
			self.spp = spp
			self.system = system

			self.device = device

			learn_revs = False
			if learn_revs:
				self.nb_revs = torch.nn.Parameter(spp.nb_revs, requires_grad=True)
			else:
				self.nb_revs = spp.nb_revs

			learn_folds = False
			if learn_folds:
				self.nb_folds = torch.nn.Parameter(spp.nb_folds, requires_grad=True)
			else:
				self.nb_folds = spp.nb_folds

			learn_rots = False
			if learn_rots:
				self.nb_rots = torch.nn.Parameter(spp.nb_rots, requires_grad=True)
			else:
				self.nb_rots = spp.nb_rots

			self.theta = spp.theta
			self.phi = spp.phi

			self.radial_downramp_start = spp.radial_downramp_start
			self.radial_downramp_stop = spp.radial_downramp_stop
			self.rev_downramp_start = spp.rev_downramp_start
			self.fold_downramp_start = spp.fold_downramp_start
			self.rot_downramp_start = spp.rot_downramp_start

			learn_decay_width = False
			if learn_decay_width:
				self.decay_width = torch.nn.Parameter(spp.decay_width, requires_grad=True)
			else:
				self.decay_width = spp.decay_width

			#self.r0 = torch.nn.Parameter(torch.scalar_tensor(1000, requires_grad=True))
			self.r0 = torch.scalar_tensor(900.0)
			self.n_time_points = 100
			self.times = torch.nn.Parameter(torch.ones(
								self.n_time_points, 
								requires_grad=True
							)*math.sqrt(system.grad_raster_time))

			self.rad_decay = sptraj.create_decay(
								self.nsamp, 
								self.decay_width, 
								spp.radial_downramp_start, 
								spp.radial_downramp_stop, 
								torch.scalar_tensor(0.0),
								1
							)
			self.rev_decay = sptraj.create_decay(
								self.nsamp, 
								self.decay_width, 
								spp.rev_downramp_start, 
								self.nsamp, 
								torch.scalar_tensor(0.84),
								4
							)
			self.fold_decay = sptraj.create_decay(
								self.nsamp, 
								self.decay_width, 
								spp.fold_downramp_start, 
								self.nsamp, 
								torch.scalar_tensor(0.80),
								4
							)
			self.rot_decay = sptraj.create_decay(
								self.nsamp, 
								self.decay_width, 
								spp.rot_downramp_start, 
								self.nsamp, 
								torch.scalar_tensor(0.78),
								4
							)

	def forward(self):
		with self.device:

			times = torch.zeros(self.nsamp)
			times[1:] = torch.cumsum(torch.square(self.times), dim=0)

			traj = sptraj.spiral_trajectory(
						times,
						self.r0,
						self.nb_revs,
						self.nb_rots,
						self.nb_folds,
						self.phi,
						self.theta,
						self.rad_decay,
						self.rev_decay,
						self.fold_decay,
						self.rot_decay
					)
			
			return traj, times


	def pns_loss(self, slew, times, logt,
		chronaxie=334e-6*torch.ones(3, requires_grad=False),
		smin=70.27*torch.ones(3, requires_grad=False),
		):
		with self.device:
			chronaxie = chronaxie.to(self.device).unsqueeze(-1)
			smin = smin.to(self.device)

			pns_kernel = chronaxie / torch.square(chronaxie + times.unsqueeze(0))
			pns_kernel /= pns_kernel.sum(axis=-1, keepdim=True)

			n_upsample = math.ceil(1.2*slew.shape[1])

			ft_pns_kernel = torch.fft.rfft(pns_kernel, n=n_upsample, dim=-1)


			ft_slew = torch.fft.rfft(slew, n=n_upsample, dim=-1)

			R = torch.fft.irfft(ft_slew*ft_pns_kernel, dim=-1) / smin.view(3,1)

			R = torch.sqrt(torch.square(R).sum(axis=0))

			if logt < 1.0:
				return torch.sum(torch.pow(R, 1.0/logt)) / R.shape[0], R
			else:
				k1 = 1.0
				r1 = 1.05
				d = 80
				p_2 = k1*((r1*R) - torch.pow(r1*R, d))
				barrier = (-1/logt)*torch.log(1+p_2)
				return torch.sum(1+barrier) / R.shape[0], R

	def slew_bound_loss(self, slew, logt):
		with self.device:
			slew_perc = torch.square(slew / self.system.max_slew)
			if logt < 1.0:
				return torch.sum(torch.pow(slew_perc, 1.0 / logt)) / slew.numel()
			else:
				k1 = 1.0
				r1 = 1.05
				d = 80
				p_2 = k1*((r1*slew_perc) - torch.pow(r1*slew_perc, d))
				barrier = (-1.0/logt)*torch.log(1+p_2)
				return torch.sum(1+barrier) / slew.numel()
			
	def grad_bound_loss(self, grad, logt):
		grad_perc = torch.square(grad / self.system.max_grad)
		if logt < 1.0:
			return torch.sum(torch.pow(grad_perc, 1.0 / logt)) / grad.numel()
		else:
			k1 = 1.0
			r1 = 1.05
			d = 80
			p_2 = k1*((r1*grad_perc) - torch.pow(r1*grad_perc, d))
			barrier = (-1.0/logt)*torch.log(1+p_2)
			return torch.sum(1+barrier) / grad.numel()

if __name__ == "__main__":

	import os
	import sys
	sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	import traj_utils as tu
	import numpy as np
	from scipy.interpolate import interp1d
	import matplotlib.pyplot as plt

	nsamp = 4000

	spp = Spiral3DProperties(
		nsamp = nsamp,
		nb_revs = torch.scalar_tensor(5.0), 
		nb_folds = torch.scalar_tensor(5.0), 
		nb_rots = torch.scalar_tensor(1.0), 
		theta = torch.scalar_tensor(0.0), 
		phi = torch.scalar_tensor(0.0), 
		radial_downramp_start = int(0.85*nsamp),
		radial_downramp_stop = nsamp-1,
		rev_downramp_start = None,
		fold_downramp_start = None,
		rot_downramp_start = None,
		decay_width = torch.scalar_tensor(0.3)
	)

	system = System(max_slew=200, max_grad=80, grad_raster_time=4e-6)

	spiral = Spiral3D(nsamp, spp, system, torch.device('cpu'))

	optimizer = torch.optim.Adam(spiral.parameters(), lr=1e-2)

	GRT = spiral.system.grad_raster_time
	system_times = GRT*torch.arange(nsamp, device=spiral.device)

	niter = 50000
	for iter in range(niter):

		traj, times = spiral()

		g, s = traj_to_grad(traj, GRT)

		pns_loss, pns_val = spiral.pns_loss(s, system_times, 0.2)

		grad_loss = spiral.grad_bound_loss(g, 0.1)
		slew_loss = spiral.slew_bound_loss(s, 0.5)

		loss = pns_loss + grad_loss + slew_loss

		print(	f'Iter {iter+1}/{niter}: '+
				f'Loss: {loss.item()}: '+
				f'PNS Loss: {pns_loss.item()}: '+
				f'Grad Loss: {grad_loss.item()}: '+
				f'Slew Loss: {slew_loss.item()}'
			)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		for piter, param in enumerate(spiral.parameters()):
			if piter == 0:
				param.data.clamp_(spiral.system.grad_raster_time/3, spiral.system.grad_raster_time*3)

		if iter % 5000 == 0:

			for param in enumerate(spiral.parameters()):
				print('\n', param, '\n')

			plt.figure()
			plt.plot(traj[0,:].detach().cpu().numpy())
			plt.plot(traj[1,:].detach().cpu().numpy())
			plt.plot(traj[2,:].detach().cpu().numpy())
			plt.title('Trajectory')
			plt.show()

			plt.figure()
			plt.plot(1e3*g[0,:].detach().cpu().numpy())
			plt.plot(1e3*g[1,:].detach().cpu().numpy())
			plt.plot(1e3*g[2,:].detach().cpu().numpy())
			plt.title('Gradient: mT/m')
			plt.show()

			plt.figure()
			plt.plot(s[0,:].detach().cpu().numpy())
			plt.plot(s[1,:].detach().cpu().numpy())
			plt.plot(s[2,:].detach().cpu().numpy())
			plt.title('Slew rate: T/m/s')
			plt.show()

			plt.figure()
			plt.plot(pns_val.detach().cpu().numpy())
			plt.title('PNS Value')
			plt.show()
			


	# to_plot1 = traj[:,:nsamp].detach().cpu().numpy()

	# to_plot2 = traj[:,nsamp:].detach().cpu().numpy()
	# to_plot2 = interp1d(np.linspace(0,1,to_plot2.shape[1]), to_plot2, axis=1)(np.linspace(0,1,nsamp))

	# to_plot = np.stack([to_plot1, to_plot2], axis=0)
	# to_plot = 0.5*to_plot / np.max(np.abs(to_plot))

	# tu.show_trajectory(to_plot.transpose(0,2,1), 0, 10)



	

