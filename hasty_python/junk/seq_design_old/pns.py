import torch



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