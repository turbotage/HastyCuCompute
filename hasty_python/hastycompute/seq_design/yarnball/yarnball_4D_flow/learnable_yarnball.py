import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import pulserver as pps
import pypulseq as pp
from pypulseq.convert import convert
from pypulseq import add_ramps as pp_ramp
from collections import OrderedDict


import os
import sys

import hastycompute.seq_design.gradient_design as gd
import hastycompute.seq_design.short_grad_design as sgd
import hastycompute.seq_design.pns_design as pnsd
import hastycompute.seq_design.radius_design as rd
import hastycompute.seq_design.yarnball.yarnball_design as ybd
import hastycompute.seq_design.velocity_design as ved

from hastycompute.seq_design.sequtil import SafetyLimits, ImageProperties
import hastycompute.utils.torch_utils as torch_utils





device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def iteration_plot(grad, grad_up, slew, slew_up, pns_waveform, DM0L, DM1L, DM2L):
	print(	
		f"VelEnc: M0: {torch_utils.formated_list_print(DM0L[0][0,...].detach().cpu())} T·m,"
	)
	print(
		f"VelEnc: M1: {torch_utils.formated_list_print(DM1L[0][0,...].detach().cpu())} T·m·s"
	)
	print(
		f"VelEnc: M2: {torch_utils.formated_list_print(DM2L[0][0,...].detach().cpu())} T·m·s²"
	)

	print(
		f"Spoiler: M0: {torch_utils.formated_list_print(DM0L[1][0,...].detach().cpu())} T·m, "
	)
	print(
		f"Spoiler: M1: {torch_utils.formated_list_print(DM1L[1][0,...].detach().cpu())} T·m·s"
	)
	print(
		f"Spoiler: M2: {torch_utils.formated_list_print(DM2L[1][0,...].detach().cpu())} T·m·s²"
	)
	
	print(
		f"MaxGrad: {grad.abs().max().item():.4f} mT/m, MaxSlew: {slew.abs().max().item():.4f} T/m/s"
		)
	print(
		f"MaxPns: {100*pns_waveform.abs().max().item():.3f} %"
	)

	max_grad_spoke_idx = torch.argmax(grad.abs().max(-1).values.max(-1).values)
	max_slew_spoke_idx = torch.argmax(slew.abs().max(-1).values.max(-1).values)
	max_pns_spoke_idx = torch.argmax(pns_waveform.abs().max(-1).values.max(-1).values)


	import matplotlib.pyplot as plt
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(grad_up[max_grad_spoke_idx,0,:].detach().cpu().numpy(), label='X')
	plt.plot(grad_up[max_grad_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
	plt.plot(grad_up[max_grad_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
	plt.ylabel('Gradient (mT/m)')
	plt.title('Gradient waveforms: Max Grad spoke')
	plt.legend()
	plt.subplot(2,1,2)
	plt.plot(grad_up[max_slew_spoke_idx,0,:].detach().cpu().numpy(), label='X')
	plt.plot(grad_up[max_slew_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
	plt.plot(grad_up[max_slew_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
	plt.ylabel('Gradient (mT/m)')
	plt.title('Gradient waveforms: Max Slew spoke')
	plt.legend()
	#plt.show()

	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(slew[max_slew_spoke_idx,0,:].detach().cpu().numpy(), label='X')
	plt.plot(slew[max_slew_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
	plt.plot(slew[max_slew_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
	plt.title('Slew rates: Max Slew spoke')
	plt.ylabel('Slew Rate (T/m/s)')
	plt.legend()
	plt.subplot(2,1,2)
	plt.plot(100*pns_waveform[max_pns_spoke_idx,:].detach().cpu().numpy(), label='PNS')
	plt.ylabel('PNS (%)')
	plt.title('PNS Waveform: Max PNS spoke')
	plt.legend()
	plt.show()

def run_learning():
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

	pns = pnsd.PNS(20*kernel_os, system.grad_raster_time / kernel_os, device=device)

	sl = SafetyLimits(0.95, 0.95)
	imgprop = ImageProperties([320,320,320], 
				torch.tensor([220e-3, 220e-3, 220e-3]), torch.tensor([320,320,320]))


	# YARNBALL
	yb_settings = ybd.YarnballSettings(device=device)
	yb_settings.nb_revs = 7
	yb_settings.nb_folds = 4
	yb_settings.rho_lambda = rd.learnable_rho(100, 1.5, 0.5)

	NGRT = 2653
	yarnball_grad = gd.GradientScaledTimeDynamicSegment(
		ybd.YarnballTimeDynamicSegment(yb_settings, NGRT, device=device),
		torch.tensor([52.80763768, 52.80763768, 52.80763768]),
		NGRT
	).forward()
	yarnball_gds = gd.GradientStaticSegment(yarnball_grad)

	print(f"Yarnball max grad: {yarnball_grad.abs().max().item():.2f} mT/m")

	# VELOCITY ENCODING
	smooth_kernel = torch.flip(torch.linspace(0,1,21, dtype=torch.float64, device=device), dims=[0])
	smooth_kernel = torch.exp(-torch.square(smooth_kernel)/0.05)
	smooth_kernel /= torch.sum(smooth_kernel)

	vef = ved.VelocityEncodingFactory(gk, sl, smooth_kernel=smooth_kernel)
	vels = torch.tensor([-0.8, 0.8, 0.8], dtype=torch.float64, device=device)
	vel_grads, vel_props = vef.get_gradients(vels, ['x', 'y', 'z'])
	ve_grad = torch.stack(vel_grads, axis=0).unsqueeze(0)
	
	velocity_encoding_gds = gd.GradientFreeInterpolatedSegment(ve_grad[...,::14], ve_grad.shape[-1], interptype="cubic")
	#velocity_encoding_gds = gd.GradientFreeSegment(ve_grad)

	# SPOILER
	spoiler_ngrt = 301
	spoiler_grad = yarnball_grad[:,:,-1].unsqueeze(-1) * (spoiler_ngrt - torch.arange(spoiler_ngrt)).view(1,1,-1) / spoiler_ngrt
	spoiler_gds = gd.GradientFreeInterpolatedSegment(spoiler_grad[...,::10], 600)

	gradient_segments = OrderedDict([
		('velocity_encoding', velocity_encoding_gds),
		('yarnball', yarnball_gds),
		('spoiler', spoiler_gds)
	])

	grad = gd.Gradient(
				gk,
				gradient_segments=gradient_segments,
				device=device
			)

	venc = 0.8
	M1 = torch.pi / (system.gamma * venc)
	first_moment_vector = torch.tensor([-M1, M1, M1], dtype=torch.float64, device=device).unsqueeze(0)

	M0_spoil = 4*torch.pi / (system.gamma * 1e-3)
	zeroth_moment_vector = torch.tensor([M0_spoil, M0_spoil, M0_spoil], dtype=torch.float64, device=device).unsqueeze(0)

	optimizer = torch.optim.AdamW(grad.parameters(), lr=1e-2)
	#learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)
	#optimizer = torch.optim.LBFGS(grad.parameters(), lr=1.0, max_iter=20, history_size=10)

	for iter in range(100000):

		waveform = grad.forward()
		waveform_up = gd.Gradient.calculate_actual_waveform(waveform, gk)
		slew = gd.Gradient.calculate_slew_rate(waveform, system.grad_raster_time)
		slew_up = gd.Gradient.calculate_slew_rate(waveform_up, system.grad_raster_time / kernel_os)
		pns_waveform = pns(slew_up)

		end_of_segments = grad.end_of_segments()
		DM0_list, DM1_list, DM2_list = gd.Gradient.calculate_specific_moments(
										waveform_up,
										grad_raster_time=system.grad_raster_time / kernel_os,
										points=torch.tensor([
											end_of_segments[0]*kernel_os-1, 
											end_of_segments[2]*kernel_os-1
										])
									)

		if iter % 10000 == 0:
			iteration_plot(waveform, waveform_up, slew, slew_up, pns_waveform, DM0_list, DM1_list, DM2_list)


		vel_enc_M1_relerr = torch.abs((DM1_list[0] - first_moment_vector) / first_moment_vector)

		# For M0 we wan't zero M0 after velocity encoding and 4pi dephasing after spoiler
		m01 = 1e6
		m02 = 1e-3
		velenc_M0_loss  = torch.sum(torch.square(torch.square(m01*DM0_list[0])))
		spoiler_M0_loss = m02*torch.sum(torch.exp(-torch.square(DM0_list[1])/torch.square(zeroth_moment_vector)))
		# For M1 we wan't venc M1 after velocity encoding and zero M1 after spoiler
		
		m11 = 1e9
		m12 = 1e6
		velenc_M1_loss  = torch.sum(torch.pow(torch.square(m11*(DM1_list[0] - first_moment_vector)), 2))
		#spoiler_M1_loss = torch.sum(torch.square(m12*DM1_list[1]))

		slew_loss = 1e-4*torch.sum(torch.pow(slew/180, 12))

		pns_loss = 1e-4*torch.sum(torch.pow(pns_waveform, 8))

		loss = velenc_M0_loss + velenc_M1_loss + spoiler_M0_loss + slew_loss + pns_loss # + spoiler_M1_loss 

		# f"Spoiler M1_loss: {spoiler_M1_loss.item():.8f}, " +
		if iter % 50 == 0:
			print(f"Iter {iter}: ")
			print(	f"	Loss: {loss.item():.8f}, " + 
					f"VelEnc M0_loss: {velenc_M0_loss.item():.8f}, " + 
					f"VelEnc M1_loss: {velenc_M1_loss.item():.8f}, " + 
					f"Spoiler M0_loss: {spoiler_M0_loss.item():.8f}, " +
					f"Slew_loss: {slew_loss.item():.8f}, " + 
					f"PNS_loss: {pns_loss.item():.8f}")
			print(	f"	Max Grad: {waveform.abs().max().item():.4f} mT/m, " +
		 			f"Max Slew: {slew.abs().max().item():.4f} T/m/s, " + 
					f"Max PNS: {100*pns_waveform.abs().max().item():.3f} %")
			print(	f"	VelEnc M1 rel err: " +
					f"{torch_utils.formated_list_print(vel_enc_M1_relerr[0,...].detach().cpu())} ")

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			clamp_max_grad = sl.grad_ratio*convert(system.max_grad, from_unit='Hz/m', to_unit='mT/m')
			for name, p in grad.named_parameters():
				if ("velocity_encoding" in name) and ("free_gradwave" in name):
					p.clamp_(min=-clamp_max_grad, max=clamp_max_grad)
				if ("spoiler" in name) and ("free_gradwave" in name):
					p.clamp_(min=-clamp_max_grad, max=clamp_max_grad)



if __name__ == "__main__":
	with device:
		run_learning()