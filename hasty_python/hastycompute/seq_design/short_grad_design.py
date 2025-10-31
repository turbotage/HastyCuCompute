import torch
import math
import gradient_design as gd
from sequtil import SafetyLimits, ImageProperties
import copy
from sequtil import convert

class ShortGradDesign:
	def __init__(self, 
				 max_ngrt: int, 
				 gk: gd.GradientKernels, 
				 gs: gd.GradientTimeDynamicSegment, 
				 sl: SafetyLimits, 
				 ip: ImageProperties,
				 device=torch.device('cpu')
			):
		self.gk = gk
		self.sl = sl
		self.ip = ip
		self.max_ngrt = max_ngrt
		self.device = device

		resolution = self.ip.resolution
		fov = self.ip.fov
		delta_k = 1.0 / fov
		self.kmax = 0.5 * delta_k * resolution

		gs.set_number_of_timepoints(max_ngrt)

		self.gs = gd.GradientScaledTimeDynamicSegment(
			gs,
			scale=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=device),
			NGRT=max_ngrt
		)

		self.found_NGRT = max_ngrt

	def get_gradient_segment(self):
		#copygs = copy.deepcopy(self.gs)
		#copygs.set_number_of_timepoints(self.found_NGRT)
		#return copygs
		return self.gs

	def optimize_NGRT(self, upper_bound_time=20e-3):
		GRT = self.gk.system.grad_raster_time

		safe_max_grad = convert(self.gk.system.max_grad, from_unit="Hz/m", to_unit="mT/m") * self.sl.grad_ratio
		safe_max_slew = convert(self.gk.system.max_slew, from_unit="Hz/m/s", to_unit="T/m/s") * self.sl.slew_ratio

		def is_safe(NGRT_now):
			self.gs.set_number_of_timepoints(NGRT_now)
			grad_waveform = self.gs.forward()  # [B, 3, N]
			kspace_waveform = gd.Gradient.calculate_kspace_traj(grad_waveform, GRT, self.gk.system.gamma)
			kspace_max = torch.abs(kspace_waveform).max(dim=-1).values.mean(dim=0)
			scale = self.gs.get_scale() * self.kmax / kspace_max
			self.gs.set_scale(scale)
			grad_waveform = self.gs.forward()
			slew_waveform = gd.Gradient.calculate_slew_rate(grad_waveform, GRT)

			#kswav = gd.Gradient.calculate_kspace_traj(grad_waveform, GRT, self.gk.system.gamma)
			#ksmax = torch.abs(kswav).max(dim=-1).values.mean(dim=0)
			#print(f"NGRT: after scaling kspace max: {ksmax.detach().cpu().numpy()}  Scale: {scale.detach().cpu().numpy()}")

			max_grad = grad_waveform.abs().max()
			max_slew = slew_waveform.abs().max()
			return (max_grad <= safe_max_grad) and (max_slew <= safe_max_slew)

		# Super long initial guess of 40 ms
		high = math.ceil(upper_bound_time / GRT)
		low  = math.ceil(0.5e-3 / GRT)

		if not is_safe(high):
			raise RuntimeError("Gradient design is not safe even at upper bound time.")

		while low < high:
			mid = (low + high) // 2
			if is_safe(mid):
				found = mid
				high = mid - 1
			else:
				low = mid + 1

		if found is None:
			raise RuntimeError("Could not find a safe gradient design lower than upper bound time.")
		
		self.found_NGRT = found
		self.gs.set_number_of_timepoints(found)


if __name__ == "__main__":
	import yarnball_design as ybd
	import pulserver as pps
	import radius_design as rd

	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

	system = pps.Opts(
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

	kernel_os = 20
	gk = gd.GradientKernels(
			system,
			gd.GradientKernels.kernels_from_test(
				system.grad_raster_time, kernel_oversampling=kernel_os, device=device
			),
			kernel_oversampling=kernel_os
		)

	yb_settings = ybd.YarnballSettings(device=device)
	yb_settings.set_nshot(100)
	yb_settings.nb_revs = 7
	yb_settings.nb_folds = 4
	yb_settings.rho_lambda = rd.learnable_rho(100, 1.5, 0.5)

	sl = SafetyLimits(0.95, 0.95)
	imgprop = ImageProperties([320,320,320], 
				torch.tensor([220e-3, 220e-3, 220e-3], device=device),
				torch.tensor([320,320,320], device=device)
			)

	max_ngrt = 20000
	sgd = ShortGradDesign(
				max_ngrt,
				gk,
				ybd.YarnballTimeDynamicSegment(yb_settings, max_ngrt, device=device),
				sl,
				imgprop,
				device=device
			)

	sgd.optimize_NGRT()

	print(f"Found NGRT: {sgd.found_NGRT} time: {sgd.found_NGRT * system.grad_raster_time * 1e3:.2f} ms")
	print(f"Scale factors: {sgd.gs.get_scale().detach().cpu().numpy()}")

	grad_seg = sgd.get_gradient_segment()
	waveform = grad_seg.forward()
	slewform = gd.Gradient.calculate_slew_rate(waveform, system.grad_raster_time)
	kspaceform = gd.Gradient.calculate_kspace_traj(waveform, system.grad_raster_time, system.gamma)

	print(f"Max gradient: {waveform.abs().max().item():.2f} mT/m")
	print(f"Max slew rate: {slewform.abs().max().item():.2f} T/m/s")

	import matplotlib.pyplot as plt
	plt.figure()
	plt.subplot(3,1,1)
	plt.plot(waveform[0,0,:].detach().cpu().numpy(), label='Gx')
	plt.plot(waveform[0,1,:].detach().cpu().numpy(), label='Gy')
	plt.plot(waveform[0,2,:].detach().cpu().numpy(), label='Gz')
	plt.title("Gradient Waveform")
	#plt.xlabel("Timepoints")
	plt.ylabel("Gradient (mT/m)")
	plt.legend()
	plt.subplot(3,1,2)
	plt.plot(slewform[0,0,:].detach().cpu().numpy(), label='Gx')
	plt.plot(slewform[0,1,:].detach().cpu().numpy(), label='Gy')
	plt.plot(slewform[0,2,:].detach().cpu().numpy(), label='Gz')
	plt.title("Slew Rate Waveform")
	#plt.xlabel("Timepoints")
	plt.ylabel("Slew Rate (T/m/s)")
	plt.legend()
	plt.subplot(3,1,3)
	plt.plot(kspaceform[0,0,:].detach().cpu().numpy(), label='Kx')
	plt.plot(kspaceform[0,1,:].detach().cpu().numpy(), label='Ky')
	plt.plot(kspaceform[0,2,:].detach().cpu().numpy(), label='Kz')
	plt.title("K-space Trajectory")
	#plt.xlabel("Timepoints")
	plt.ylabel("K-space (1/m)")
	plt.legend()
	plt.show()
