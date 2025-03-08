import torch
from rot import Rz, Ry, Ra
import math

def create_decay(
		NT: int, 
		decay_width: torch.Tensor, 
		downramp_start: int, 
		downramp_stop: int, 
		stop_val: torch.Tensor = torch.scalar_tensor(0.0),
		conv_type: int = 1
	):

	if decay_width.device != stop_val.device:
		raise ValueError('decay_width and stop_val must be on the same device')
	
	if decay_width.item() > 0.99999:
		raise ValueError('Decay width must be less than 1.0')

	with decay_width.device:
		NTR = downramp_stop-downramp_start
		ramp = torch.linspace(1, stop_val, NTR)
		decay = torch.ones(NT+2*NTR)
		decay[NTR+downramp_start:NTR+downramp_stop] = ramp
		decay[NTR+downramp_stop:] = stop_val

		DECAY_N = int(decay_width.item()*NTR)

		if conv_type == 1:
			if DECAY_N < 15:
				raise ValueError('Decay profile is too short for Gaussian profile')
			ckernel = torch.linspace(-0.5,0.5,DECAY_N)
			ckernel = torch.exp(-torch.square(ckernel)/((0.3)**2))
			ckernel /= torch.sum(ckernel)
			ckernel[:5] = 0.0
			ckernel[-5:] = 0.0
		if conv_type == 2:
			if DECAY_N < 8:
				raise ValueError('Decay profile is too short for Gaussian profile')
			ckernel = torch.linspace(0,0.5,DECAY_N)
			ckernel = torch.exp(-torch.square(ckernel)/((0.3)**2))
			ckernel /= torch.sum(ckernel)
			ckernel[-5:] = 0.0
		if conv_type == 3:
			if DECAY_N < 8:
				raise ValueError('Decay profile is too short for Gaussian profile')
			ckernel = torch.linspace(-0.5,0,DECAY_N)
			ckernel = torch.exp(-torch.square(ckernel)/((0.3)**2))
			ckernel /= torch.sum(ckernel)
			ckernel[:5] = 0.0
		if conv_type == 4:
			ckernel = torch.ones(DECAY_N)
			ckernel /= torch.sum(ckernel)

		ft_decay = torch.fft.rfft(decay.to(torch.float64))

		ckernel = torch.fft.rfft(ckernel.to(torch.float64), n=decay.shape[0])

		decay = torch.fft.irfft(ft_decay*ckernel).to(torch.float32)

		if conv_type > 3:
			decay = decay[NTR:NTR+NT]
		else:
			decay = decay[NTR+DECAY_N:NTR+DECAY_N+NT]

		if decay.shape[0] != NT:
			raise ValueError('Decay profile has incorrect length')

		decay[decay < 0.5*1e-6] = 0.0

		return decay

def spiral_trajectory(
				times: torch.Tensor,
				r0: torch.Tensor, 
				nb_revs: torch.Tensor,
				nb_rot: torch.Tensor,
				nb_folds: torch.Tensor,
				phi: torch.Tensor,
				theta: torch.Tensor,
				rad_decay: torch.Tensor,
				rev_decay: torch.Tensor,
				fold_decay: torch.Tensor,
				rot_decay: torch.Tensor,
				smooth_width: int = 80,
				plot_profiles: bool = False
			   ):
	if plot_profiles:
		import matplotlib.pyplot as plt

	tdev = times.device
	with tdev:
		
		rad = r0*times/times[-1]
		rad = rad*rad_decay

		if plot_profiles:
			plt.figure()
			plt.plot(times.detach().cpu().numpy(), rad.detach().cpu().numpy())
			plt.title('Radius Profile')
			plt.show()

		rev = (2*torch.pi*nb_revs/times[-1])*times
		rev = rev*rev_decay

		if plot_profiles:
			plt.figure()
			plt.plot(times.detach().cpu().numpy(), rev.detach().cpu().numpy())
			plt.title('Revolution Profile')
			plt.show()

		pos = torch.zeros((times.shape[0], 3))
		pos[:,0] = rad*torch.cos(rev)
		pos[:,1] = rad*torch.sin(rev)

		rot = (torch.pi*nb_rot/times[-1])*times
		rot = rot*rot_decay

		if plot_profiles:
			plt.figure()
			plt.plot(times.detach().cpu().numpy(), rot.detach().cpu().numpy())
			plt.title('Rotation Profile')
			plt.show()
		
		fold = (torch.pi*nb_folds/times[-1])*times
		fold = fold*fold_decay

		if plot_profiles:
			plt.figure()
			plt.plot(times.detach().cpu().numpy(), fold.detach().cpu().numpy())
			plt.title('Folding Profile')
			plt.show()

		rot_vectors = torch.stack([
							torch.cos(rot), 
							torch.sin(rot), 
							torch.zeros_like(rot)
						], dim=0)
		
		rotmats =   torch.bmm(
						Rz(torch.tensor([phi])).expand(times.shape[0], -1, -1),
						torch.bmm(
							Ry(torch.tensor([theta])).expand(times.shape[0], -1, -1),
							Ra(rot_vectors, fold)
						)
					)   

		pos = torch.bmm(rotmats, pos.unsqueeze(-1)).squeeze(-1)
		pos = pos.permute(1,0).contiguous()

		#tgauss = torch.linspace(0,0.5,smooth_width)
		#gauss = torch.exp(-torch.square(tgauss)/(0.25)**2)
		#gauss /= torch.sum(gauss)
		#tgauss = torch.fft.rfft(gauss.to(torch.float64), n=pos.shape[1])
		#pos = torch.fft.rfft(pos.to(torch.float64), dim=-1)
		#pos = torch.fft.irfft(pos*tgauss).to(torch.float32)

		return pos