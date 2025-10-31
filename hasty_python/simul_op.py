import cupy as cp
import numpy as np
import h5py
import time

import cufinufft

import orthoslicer as ort
import post_processing as pp

def tseg_interpolators(rate_map, t, bins, lseg):

	t = cp.array(t)
	b0_cat = 2j * np.pi * cp.concatenate(cp.array(rate_map))
	hist_wt, bin_edges = cp.histogram(
		cp.imag(2j * np.pi * cp.concatenate(cp.array(rate_map))), bins
	)

	bin_centers = 0.5 * (bin_edges[1:] - bin_edges[1])
	zk = 0 + 1j*bin_centers

	t = t - t[0]
	T = t[-1]

	tl = cp.linspace(0, T, lseg)

	ch = cp.exp(tl[:,None] * zk[None,...])
	w = cp.diag(cp.sqrt(hist_wt))
	p = cp.linalg.pinv(w @ cp.transpose(ch)) @ w
	b = p @ cp.exp(-zk[...,None] * t[None,:])

	ct = cp.exp(-tl[:,None,None,None] * (2j * cp.pi * cp.array(rate_map)[None,:]))
	
	return b, ct




def simul_op(coord, t, img, smp, z):

	shape = img.shape[2:]
	grid = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
	grid = np.meshgrid(*grid, indexing='ij')
	grid = np.stack(grid)

	#ort.image_nd(np.abs(img))
	
	if z == None:
		z = 100*(-1*np.sum(np.square(grid / (0.5*max(shape))), axis=0)+0.75) # between -25 and 75 Hz
		#ort.image_nd(z[None,...])
		z = 2*np.pi*z
		z = cp.array((1.0/1500.0) + 1j*z, dtype=cp.complex64)

	b, ct = tseg_interpolators(z, t, int(z.size // 100), 10)

	img = cp.array(img)
	smp = cp.array(smp)
	coord = cp.array(coord)

	start = time.time()
	out = np.zeros((img.shape[0], img.shape[1], smp.shape[0], t.shape[0]), dtype=cp.complex64)
	for frame in range(img.shape[0]):
		for enc in range(img.shape[1]):
			out_temp = cp.zeros((smp.shape[0], t.shape[0]), dtype=cp.complex64)
			for i in range(ct.shape[0]):
				out_temp += b[i,...] * cufinufft.nufft3d2(coord[frame,0], coord[frame,1], coord[frame,2], 
											ct[i,...][None,...] * smp * img[frame, enc,...][None,...])
			out[frame,enc,...] = cp.sum(out_temp, axis=0).get()
	end = time.time()
	print('Elapsed time for forward: ', end-start)

	start = time.time()
	recon_img = np.zeros((img.shape[0], img.shape[1], smp.shape[0], shape[0], shape[1], shape[2]), dtype=cp.complex64)
	for frame in range(img.shape[0]):
		for enc in range(img.shape[1]):
			recon_temp = cp.zeros((smp.shape[0], shape[0], shape[1], shape[2]), dtype=cp.complex64)
			out_temp = cp.array(out[frame,enc,...])
			for i in range(ct.shape[0]):
				recon_temp += smp.conj() * ct[i,...].conj()[None,...] * cufinufft.nufft3d1(
					coord[frame,0], coord[frame,1], coord[frame,2], b[i,...].conj()[None,...] * out_temp, n_modes=shape)
			recon_img[frame,enc,...] = recon_temp.get()

	end = time.time()
	print('Elapsed time for adjoint: ', end-start)

	smp = smp.get()

	sos_image = np.sum(smp.conj()[None,None,...] * recon_img, axis=2) / np.sum(smp.conj()*smp, axis=0)[None,None,...]

	ort.image_nd(np.abs(sos_image))

	print('Coo')

def simul_op_mean(coord, t, img, smp, z):

	shape = img.shape[2:]
	grid = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
	grid = np.meshgrid(*grid, indexing='ij')
	grid = np.stack(grid)

	#ort.image_nd(np.abs(img))
	
	if z == None:
		z = 100*(-1*np.sum(np.square(grid / (0.5*max(shape))), axis=0)+0.75) # between -25 and 75 Hz
		#ort.image_nd(z[None,...])
		z = 2*np.pi*z
		z = cp.array((1.0/1500.0) + 1j*z, dtype=cp.complex64)

	b, ct = tseg_interpolators(z, t, int(z.size // 100), 10)

	img = cp.array(img)
	smp = cp.array(smp)
	coord = cp.array(coord)

	img = cp.mean(img, axis=0)

	start = time.time()
	out = np.zeros((img.shape[1], smp.shape[0], t.shape[0]), dtype=cp.complex64)
	for enc in range(img.shape[1]):
		out_temp = cp.zeros((smp.shape[0], t.shape[0]), dtype=cp.complex64)
		for i in range(ct.shape[0]):
			out_temp += b[i,...] * cufinufft.nufft3d2(coord[0], coord[1], coord[2], 
										ct[i,...][None,...] * smp * img[enc,...][None,...])
		out[enc,...] = cp.sum(out_temp, axis=0).get()
	end = time.time()
	print('Elapsed time for forward: ', end-start)

	#Create Right Hand Side
	start = time.time()
	recon_img = np.zeros((img.shape[1], smp.shape[0], shape[0], shape[1], shape[2]), dtype=cp.complex64)
	for enc in range(img.shape[1]):
		recon_temp = cp.zeros((smp.shape[0], shape[0], shape[1], shape[2]), dtype=cp.complex64)
		out_temp = cp.array(out[frame,enc,...])
		for i in range(ct.shape[0]):
			recon_temp += smp.conj() * ct[i,...].conj()[None,...] * cufinufft.nufft3d1(
				coord[0], coord[1], coord[2], b[i,...].conj()[None,...] * out_temp, n_modes=shape)
		recon_img[frame,enc,...] = recon_temp.get()
	end = time.time()
	print('Elapsed time for adjoint: ', end-start)

	sum_rhs = cp.sum(recon_img, axis=1)

	


NK = 100000
NS = 16

with h5py.File('/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/background_corrected_cropped.h5', 'r') as f:
	img = f['img'][:]
	smp = f['smaps'][:]


run_framed = True
if run_framed:

	coord = np.empty((img.shape[0], 3, NK))
	for frame in range(img.shape[0]):
		coord[frame,...] = 2*np.pi*np.random.rand(3,NK) - np.pi

	t = np.linspace(0.005, 0.01, NK)

	simul_op(coord, t, img, smp[::2], z=None)

else:
	coord = 2*np.pi*np.random.rand(3,NK) - np.pi
	t = np.linspace(0.005, 0.01, NK)
	simul_op_mean(coord, t, img, smp[::2], z=None)