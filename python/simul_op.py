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

	ct = np.exp(-tl[:,None] * (2j * np.pi * rate_map[None,:]))
	
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

	cufinufft.nufft3d1()

	#p = np.linalg.pinv()


	print('Coo')

NK = 10000
NS = 16

with h5py.File('/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/background_corrected_cropped.h5', 'r') as f:
	img = f['img'][:]
	smp = f['smaps'][:]

coord = np.random.rand(3,NK)
t = np.linspace(0.005, 0.01, NK)

simul_op(coord, t, img, smp, z=None)