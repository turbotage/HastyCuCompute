import cupy as cp
import numpy as np
import h5py
import time

import orthoslicer as ort
import post_processing as pp

def simul_op(coord, t, img, smp, z):

	shape = img.shape[2:]
	grid = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
	grid = np.meshgrid(*grid, indexing='ij')
	grid = np.stack(grid)

	ort.image_nd(np.abs(img))
	
	if z == None:
		z = 100*(-1*np.sum(np.square(grid / (0.5*max(shape))), axis=0)+0.75) # between -25 and 75 Hz
		ort.image_nd(z[None,...])
		z = 2*np.pi*z
		z = cp.array((1.0/1500.0) + 1j*z, dtype=cp.complex64)


	print('Coo')

NK = 10000
NS = 16

with h5py.File('/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/background_corrected_cropped.h5', 'r') as f:
	img = f['img'][:]
	smp = f['smaps'][:]

coord = np.random.rand(3,NK)
t = np.linspace(0.005, 0.01, NK)

simul_op(coord, t, img, smp, z=None)