import cupy as cp
import numpy as np
import h5py
import time

def simul_op(coord, t, img, smp, z):

	shape = img.shape
	grid = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
	grid = np.meshgrid(*grid, indexing='ij')
	grid = np.stack(grid)
	grid = cp.array(grid, dtype=cp.float32)

	with h5py.File('/home/turbotage/Documents/4DRecon/other_data/reconed_full_simulated.h5', 'r') as f:
		img = f['img'][:]
	with h5py.File('/home/turbotage/Documents/4DRecon/other_data/reconed_full_simulated.h5', 'r') as f:
		smp = f['smp'][:]

	coord = cp.array(coord, dtype=cp.float32)
	t = cp.array(t, dtype=cp.float32)
	img = cp.array(img, dtype=cp.complex64)
	smp = cp.array(smp, dtype=cp.complex64)
	z = cp.array(z, dtype=cp.complex64)

	ret = cp.empty((smp.shape[0], coord.shape[1]), dtype=cp.complex64)

	@cp.fuse
	def one_coord(c, t, img, smp, z, grid):
		integrand = cp.exp(1j * cp.sum(c[:,None,None,None] * grid, axis=0))
		integrand *= img
		integrand *= cp.exp(1j * t * z)
		integrand = smp * integrand[None,...]
		return cp.sum(integrand, axis=(1,2,3))

	start = time.time()
	for i in range(coord.shape[1]):
		ret[:,i] = one_coord(coord[:,i], t[i], img, smp, z, grid)

	ret = cp.asnumpy(ret)
	end = time.time()

	print('Elapsed time: ', end-start)

	print('Coo')

NK = 10000
NX, NY, NZ = 96, 96, 96
NS = 16



coord = np.random.rand(3,NK)
t = np.linspace(0.005, 0.01, NK)
smp = np.random.rand(NS,NX,NY,NZ) + 1j*np.random.rand(NS,NX,NY,NZ)
img = np.random.rand(NX,NY,NZ) + 1j*np.random.rand(NX,NY,NZ)
z = np.random.rand(NX,NY,NZ) + 1j*np.random.rand(NX,NY,NZ)

simul_op(coord, t, img, smp, z)