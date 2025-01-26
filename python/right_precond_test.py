import numpy as np
import cupy as cp
import cufinufft
import time
from scipy import signal

import seq_design2 as sd2
import pypulseq as pp
import h5py

#import plot_utility as pu
import orthoslicer as ort
from traj_utils import show_trajectory

DN = 96*96*16
NX = 64

def maxAHA(coord, x, s, lamda=0.0):
    y = cufinufft.nufft3d2(coord[0,:], coord[1,:], coord[2,:], s*x[None,:])
    back = s.conj() * cufinufft.nufft3d1(coord[0,:], coord[1,:], coord[2,:], y, n_modes=x.shape)
    return cp.sum(back, axis=0) + lamda*x


def power_iter(A, x0):
    for i in range(25):
        y = A(x0)
        maxeig = cp.linalg.norm(y)
        x0 = y / maxeig
        #print('\r', i, maxeig, end='')
        #time.sleep(0.5)

    return maxeig, x0

def circular_fft(x, kernel):
    xp = cp.zeros(tuple([2*s for s in x.shape]), dtype=cp.complex64)
    xp[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    y = cp.fft.ifftshift(xp)
    y = cp.fft.fftn(y)
    y = cp.fft.fftshift(y)
    y = y * kernel
    y = cp.fft.ifftshift(y)
    y = cp.fft.ifftn(y)
    y = cp.fft.fftshift(y)
    y = cp.ascontiguousarray(y[:x.shape[0], :x.shape[1], :x.shape[2]])
    return y


with h5py.File('/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/background_corrected_cropped.h5', 'r') as f:
    img = f['img'][:]
    smp = f['smaps'][:]

smp = cp.array(smp[::2,...])

x0 = np.random.rand(NX,NX,NX).astype(np.float32) + 1j*np.random.rand(NX,NX,NX).astype(np.float32)

#coord = 0.1*np.pi*cp.random.normal(size=(3,DN)).astype(np.float32)
#coord = 2*np.pi*cp.random.rand(3,DN).astype(np.float32) - np.pi

traj, grad, slew = sd2.create_traj_grad_slew(160, 96, 220e-3, pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0))

traj = traj[:,:,:]
traj = np.pi*traj / np.abs(traj).max()

figure_size = 15  # Figure size for trajectory plots
one_shot = -5  # Highlight one shot in particular
show_trajectory(0.5*traj / np.abs(traj).max(), figure_size=figure_size, one_shot=one_shot)
coord = np.ascontiguousarray(traj.reshape(traj.shape[0] * traj.shape[1], 3).transpose(1,0).astype(np.float32))
coord = cp.array(coord)
DN = coord.shape[1]

#coord[coord > 3.1415] = 3.14
#coord[coord < -3.1415] = -3.14



#grid_scale = 1 + cp.array(np.sum(np.square(grid), axis=0))**0.5

#coord_pts = (NX*coord/np.pi).astype(np.int32) + NX//2
coord_pts = (2*NX*(coord+np.pi)/(2*np.pi)).astype(np.int32)

grid_scale = cp.ones(tuple([2*s for s in x0.shape])).astype(np.float32)
for i in range(DN):
    grid_scale[coord_pts[0,i], coord_pts[1,i], coord_pts[2,i]] += 5

sigma = 2.0     # width of kernel
x = cp.arange(-4,5,1)   # coordinate arrays -- make sure they contain 0!
y = cp.arange(-4,5,1)
z = cp.arange(-4,5,1)
xx, yy, zz = cp.meshgrid(x,y,z)
kernel = cp.exp(-(xx.astype(np.float32)**2 + yy.astype(np.float32)**2 + zz.astype(np.float32)**2)/(2*sigma**2))

#ort.image_nd(grid_scale.get())
grid_scale = (1.0 / (grid_scale/np.max(grid_scale)))**2
#ort.image_nd(grid_scale.get())
grid_scale = cp.array(signal.convolve(grid_scale.get(), kernel.get(), mode='same'))
grid_scale = (grid_scale / cp.min(grid_scale))

ort.image_nd(grid_scale.get())
#grid_scale = grid_scale*0.0 + 1.0


x0 = cp.array(x0)


lamda = 0.0

maxAHA_func = lambda x: maxAHA(coord, x, smp, lamda)
maxeig, _ = power_iter(maxAHA_func, cp.copy(x0))
minAHA_func = lambda x: maxAHA(coord, x, smp, lamda) - maxeig*x
mineig, _ = power_iter(minAHA_func, cp.copy(x0))
mineig = -mineig + maxeig

print('Maxeig: ', maxeig, 'Mineig: ', mineig, 'Ratio: ', maxeig/mineig)

max_AHA_precond_func = lambda x: circular_fft(maxAHA(coord, circular_fft(x, grid_scale), smp, lamda), grid_scale)
maxeig, _ = power_iter(max_AHA_precond_func, cp.copy(x0))
min_AHA_precond_func = lambda x: circular_fft(maxAHA(coord, circular_fft(x, grid_scale), smp, lamda), grid_scale) - maxeig*x
mineig, _ = power_iter(min_AHA_precond_func, cp.copy(x0))
mineig = -mineig + maxeig

print('Maxeig: ', maxeig, 'Mineig: ', mineig, 'Ratio: ', maxeig/mineig)








# coord_int = np.stack([np.random.randint(0,NX,DN), np.random.randint(0,NX,DN), np.random.randint(0,NX,DN)])
# coord_int = np.array([NX//2,NX//2,NX//2])[:,None]

# coord = 2*np.pi * (coord_int / NX) - np.pi

# coord = cp.array(coord.astype(np.float32))

# data = 10*cp.array(np.random.rand(DN).astype(np.float32) + 0*1j*np.random.rand(DN).astype(np.float32))

# coord_int_filled = np.zeros((NX,NX,NX), dtype=np.complex64)
# for i in range(DN):
#     coord_int_filled[coord_int[0,i], coord_int[1,i], coord_int[2,i]] = data[i].get()
# ort.image_nd(coord_int_filled)

# img_back = cufinufft.nufft3d1(coord[0,:], coord[1,:], coord[2,:], data, n_modes=[NX,NX,NX])

# img_back = cp.fft.fftshift(img_back)
# img_fk = cp.fft.fftn(img_back)
# img_fk = cp.fft.fftshift(img_fk)

# ort.image_nd(img_fk.get())


# ort.image_nd(img_back.get())




