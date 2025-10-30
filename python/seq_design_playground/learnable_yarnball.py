import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import pulserver as pps
import pypulseq as pp
from pypulseq.convert import convert
from pypulseq import add_ramps as pp_ramp

from sequtil import SafetyLimits, ImageProperties
from vel_design import VelocityEncodingFactory

from collections import OrderedDict

import gradient_design as gd
import short_grad_design as sgd

import yarnball_design as ybd
import pns_design as pnsd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch_utils

if __name__ == "__main__":
	import velocity_design as ved

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
				system.grad_raster_time, kernel_oversampling=kernel_os
			),
			kernel_oversampling=kernel_os
		)

	pns = pnsd.PNS(20*kernel_os, system.grad_raster_time / kernel_os)

	sl = SafetyLimits(0.95, 0.95)
	imgprop = ImageProperties([320,320,320], 
				np.array([220e-3, 220e-3, 220e-3]), np.array([320,320,320]))
	
	# YARNBALL
	yb_settings = ybd.YarnballSettings()
	yb_settings.nb_revs = 7
	yb_settings.nb_folds = 4

	yarnball_grad = gd.GradientScaledTimeDynamicSegment(
		ybd.YarballTimeDynamicSegment(yb_settings, 2719),
		torch.tensor([51.14, 51.14, 51.14]),
		2719
	).forward()
	yarnball_gds = gd.GradientStaticSegment(yarnball_grad)
	
	# VELOCITY ENCODING
	smooth_kernel = torch.flip(torch.linspace(0,1,20), dims=[0])
	smooth_kernel = torch.exp(-torch.square(smooth_kernel)/0.05)
	smooth_kernel /= torch.sum(smooth_kernel)

	vef = ved.VelocityEncodingFactory(gk, sl, smooth_kernel=smooth_kernel)
	vel_grads, vel_props = vef.get_gradients([0.8*(-1.0), 0.8*(1.0 ), 0.8*(1.0)], ['x', 'y', 'z'])
	ve_grad = torch.stack(vel_grads, axis=0).unsqueeze(0)
	velocity_encoding_gds = gd.GradientFreeSegment(ve_grad)

	# SPOILER
	spoiler_ngrt = 200
	spoiler_grad = yarnball_grad[:,:,-1].unsqueeze(-1) * (spoiler_ngrt - torch.arange(spoiler_ngrt)).view(1,1,-1) / spoiler_ngrt
	spoiler_gds = gd.GradientFreeSegment(spoiler_grad)

	gradient_segments = OrderedDict([
		('velocity_encoding', velocity_encoding_gds),
		('yarnball', yarnball_gds),
		('spoiler', spoiler_gds)
	])

	grad = gd.Gradient(
				gk,
				device=torch.device('cpu'),
				gradient_segments=gradient_segments
			)
	
	venc = 0.8
	M1 = torch.pi / (system.gamma * venc)
	first_moment_vector = torch.tensor([-M1, M1, M1]).unsqueeze(0)

	M0_spoil = 4*torch.pi / (system.gamma * 1e-3)
	zeroth_moment_vector = torch.tensor([M0_spoil, M0_spoil, M0_spoil]).unsqueeze(0)

	optimizer = torch.optim.AdamW(grad.parameters(), lr=1e-1)
	for iter in range(10000):

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

		if iter % 50 == 0:
			print(	
				f"VelEnc: M0: {torch_utils.formated_list_print(DM0_list[0][0,...].detach().cpu())} T·m,"
			)
			print(
		 		f"VelEnc: M1: {torch_utils.formated_list_print(DM1_list[0][0,...].detach().cpu())} T·m·s"
			)
			print(
				f"VelEnc: M2: {torch_utils.formated_list_print(DM2_list[0][0,...].detach().cpu())} T·m·s²"
			)

			print(
				f"Spoiler: M0: {torch_utils.formated_list_print(DM0_list[1][0,...].detach().cpu())} T·m, "
			)
			print(
				f"Spoiler: M1: {torch_utils.formated_list_print(DM1_list[1][0,...].detach().cpu())} T·m·s"
			)
			print(
				f"Spoiler: M2: {torch_utils.formated_list_print(DM2_list[1][0,...].detach().cpu())} T·m·s²"
			)
			
			print(
				f"MaxGrad: {waveform.abs().max().item():.4f} mT/m, MaxSlew: {slew.abs().max().item():.4f} T/m/s"
				)
			print(
				f"MaxPns: {100*pns_waveform.abs().max().item():.3f} %"
			)

			max_grad_spoke_idx = torch.argmax(waveform.abs().max(-1).values.max(-1).values)
			max_slew_spoke_idx = torch.argmax(slew.abs().max(-1).values.max(-1).values)
			max_pns_spoke_idx = torch.argmax(pns_waveform.abs().max(-1).values.max(-1).values)


			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(waveform_up[max_grad_spoke_idx,0,:].detach().cpu().numpy(), label='X')
			plt.plot(waveform_up[max_grad_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
			plt.plot(waveform_up[max_grad_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
			plt.title('Gradient waveforms: Max Grad spoke')
			plt.legend()
			plt.show()

			plt.figure()
			plt.plot(slew[max_slew_spoke_idx,0,:].detach().cpu().numpy(), label='X')
			plt.plot(slew[max_slew_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
			plt.plot(slew[max_slew_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
			plt.title('Slew rates: Max Slew spoke')
			plt.legend()
			plt.show()

			plt.figure()
			plt.plot(waveform_up[max_slew_spoke_idx,0,:].detach().cpu().numpy(), label='X')
			plt.plot(waveform_up[max_slew_spoke_idx,1,:].detach().cpu().numpy(), label='Y')
			plt.plot(waveform_up[max_slew_spoke_idx,2,:].detach().cpu().numpy(), label='Z')
			plt.title('Gradient waveforms: Max Slew spoke')
			plt.legend()
			plt.show()

			plt.figure()
			plt.plot(pns_waveform[max_pns_spoke_idx,:].detach().cpu().numpy())
			plt.title('PNS Waveform: Max PNS spoke')
			plt.show()

		# For M0 we wan't zero M0 after velocity encoding and 4pi dephasing after spoiler
		m01 = 1e1
		m02 = 1e2
		M0_loss = torch.sum(torch.square(m01*DM0_list[0])) + torch.sum(torch.square(m02*(DM0_list[1].abs() - zeroth_moment_vector)))
		# For M1 we wan't venc M1 after velocity encoding and zero M1 after spoiler
		m11 = 1e6
		m12 = 1e6
		M1_loss = torch.sum(torch.square(m11*(DM1_list[0] - first_moment_vector))) + torch.sum(torch.square(m12*DM1_list[1]))

		slew_loss = 1e-3*torch.sum(torch.pow(slew/180, 12))

		pns_loss = 1e-3*torch.sum(torch.pow(pns_waveform, 8))

		loss = M0_loss + M1_loss + slew_loss + pns_loss #M0_loss + M1_loss + slew_loss + pns_loss

		# # test each term in isolation
		# for name, term in [('M0', M0_loss), ('M1', M1_loss), ('slew', slew_loss), ('pns', pns_loss)]:
		# 	optimizer.zero_grad()
		# 	try:
		# 		with torch.autograd.detect_anomaly():
		# 			term.backward(retain_graph=True)
		# 	except RuntimeError as e:
		# 		print(f"BACKWARD FAILED for {name}: {e}")
		# 		print('')
		# 	else:
		# 		print(f"BACKWARD OK for {name}")

		print(f"Loss: {loss.item():.8f}, M0_loss: {M0_loss.item():.8f}, M1_loss: {M1_loss.item():.8f}, Slew_loss: {slew_loss.item():.8f}, PNS_loss: {pns_loss.item():.8f}")

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			clamp_max_grad = sl.grad_ratio*convert(system.max_grad, from_unit='Hz/m', to_unit='mT/m')
			for name, p in grad.named_parameters():
				if ("velocity_encoding" in name) and ("free_gradwave" in name):
					p.clamp_(max=clamp_max_grad)
				if ("spoiler" in name) and ("free_gradwave" in name):
					p.clamp_(max=clamp_max_grad)