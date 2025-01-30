import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy import signal as cusignal
import cufinufft
import time
from scipy import signal
from scipy.ndimage import zoom
from scipy import ndimage

import torch
import pywt
import ptwt

import seq_design2 as sd2
import pypulseq as pp
import h5py
import math

#import plot_utility as pu
import orthoslicer as ort
from traj_utils import show_trajectory

def soft_thresholding(x, thresh):
    return torch.maximum(torch.abs(x) - thresh, torch.tensor(0.0)) * torch.sign(x)

def wavelet_thresholding(x, thresh):
    xt = torch.as_tensor(x[None,:], device='cuda:0')
    wavelet = pywt.Wavelet('haar')

    def one_comp(xten):
        wav_data = ptwt.wavedec3(xten, wavelet, level=2)
        wav_data = list(wav_data)
        wav_data[0] = soft_thresholding(wav_data[0], thresh)
        for i in range(1,3):
            for key in wav_data[i].keys():
                wav_data[i][key] = soft_thresholding(wav_data[i][key], thresh)
        return ptwt.waverec3(tuple(wav_data),wavelet)

    return cp.asarray(
        torch.complex(
            one_comp(torch.real(xt)), 
            one_comp(torch.imag(xt))
        )[0,...]
    )


def maxAHA(coord, x, s, lamda=0.0):
    # fac = 1.0 / math.sqrt(NX*NX*NX)
    # y = cufinufft.nufft3d2(
    #     coord[0,:].astype(cp.float64), 
    #     coord[1,:].astype(cp.float64), 
    #     coord[2,:].astype(cp.float64), 
    #     (s*x[None,:]).astype(cp.complex128) * fac) * fac
    # back = s.conj() * cufinufft.nufft3d1(
    #     coord[0,:].astype(cp.float64), 
    #     coord[1,:].astype(cp.float64), 
    #     coord[2,:].astype(cp.float64), 
    #     y, n_modes=x.shape)
    # return (cp.sum(back, axis=0) + lamda*x).astype(cp.complex64)

    fac = 1.0 / math.sqrt(NX*NX*NX)
    y = cufinufft.nufft3d2(
        coord[0,:], 
        coord[1,:], 
        coord[2,:], 
        (s*x[None,:]) * fac)
    back = s.conj() * cufinufft.nufft3d1(
        coord[0,:], 
        coord[1,:], 
        coord[2,:], 
        y, n_modes=x.shape)
    return (cp.sum(back * fac, axis=0) + lamda*x)


def power_iter(A, x0, iter=50):
    for i in range(iter):
        y = A(x0)
        maxeig = cp.linalg.norm(y)
        x0 = y / maxeig
        print('\r', i, maxeig, end='')
        #time.sleep(0.25)
    print('')
    return maxeig, x0

# def circular_fft(x, kernel):
#     xp = cp.zeros(tuple([2*s for s in x.shape]), dtype=cp.complex64)
#     xp[:x.shape[0], :x.shape[1], :x.shape[2]] = x
#     y = cp.fft.ifftshift(xp)
#     y = cp.fft.fftn(y)
#     y = cp.fft.fftshift(y)
#     y = y * kernel
#     y = cp.fft.ifftshift(y)
#     y = cp.fft.ifftn(y)
#     y = cp.fft.fftshift(y)
#     y = cp.ascontiguousarray(y[:x.shape[0], :x.shape[1], :x.shape[2]])
#     return y

def circular_fft(x, kernel):
    y = cp.fft.fftn(x)
    y = y * kernel
    return cp.fft.ifftn(y)

with h5py.File('/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/background_corrected_cropped.h5', 'r') as f:
    img = f['img'][:]
    smp = f['smaps'][:]
    vessel_mask = f['vessel_mask'][:]


mean_img = np.mean(img[:,0,...], axis=0)
img = img[0,0,...]

wavelet_thresholding(img, 0.1)

def upscale_volume(ximg):
    return zoom(ximg, (2,2,2), order=2)


    # #ort.image_nd(ximg)
    # NX = ximg.shape[0]
    # ximg = cp.fft.fftn(cp.array(ximg))
    # ximg2 = cp.zeros((2*NX, 2*NX, 2*NX), dtype=cp.complex64)
    # ximg = cp.fft.fftshift(ximg)
    # ximg2[:NX, :NX, :NX] = ximg
    # ximg2 = cp.fft.ifftshift(ximg2)

    # sigma = 0.005
    # x = np.arange(-5,6,1)   # coordinate arrays -- make sure they contain 0!
    # y = np.arange(-5,6,1)
    # z = np.arange(-5,6,1)
    # xx, yy, zz = np.meshgrid(x,y,z)
    # kernel = np.sinc((xx.astype(np.float32)**2 + yy.astype(np.float32)**2 + zz.astype(np.float32)**2)/sigma)
    # ximg2 = cp.array(signal.convolve(ximg2.get(), kernel, mode='same'))
    # ximg = cp.fft.ifftn(ximg2)
    # #img = cp.fft.fftshift(img)
    # del ximg2
    # #ort.image_nd(ximg.get())
    # return ximg

smp = smp[::2,...]

img = upscale_volume(img)
smp_vec = []
for i in range(smp.shape[0]):
    smp_vec.append(upscale_volume(smp[i,...]))
smp = np.stack(smp_vec)

mean_img = upscale_volume(mean_img)
mean_val_img = np.abs(mean_img).mean()
mean_img /= mean_val_img
img /= mean_val_img
del mean_val_img


vessel_mask = upscale_volume(vessel_mask.astype(np.float32))
vessel_mask = vessel_mask > 0.9

dilate_mask = ndimage.generate_binary_structure(3,1)
vessel_mask = ndimage.binary_dilation(vessel_mask, dilate_mask, iterations=5)

not_vessel_mask = np.logical_not(vessel_mask)

#ort.image_nd(vessel_mask.astype(np.float32), max_clim=True)
#ort.image_nd(not_vessel_mask.astype(np.float32), max_clim=True)


#ort.image_nd(img)

#img_w = wavelet_thresholding(img, 0.0001)
#ort.image_nd(img_w.get())
#img_w = wavelet_thresholding(img, 0.001)
#ort.image_nd(img_w.get())
#img_w = wavelet_thresholding(img, 0.01)
#ort.image_nd(img_w.get())
#img_w = wavelet_thresholding(img, 0.1)
#ort.image_nd(img_w.get())


#ort.image_nd(smp)

img = cp.array(img)
smp = cp.array(smp)




NX = img.shape[0]

def rand_vec():
    return cp.array(np.random.rand(NX,NX,NX).astype(np.float32) + 1j*np.random.rand(NX,NX,NX).astype(np.float32))


traj, grad, slew = sd2.create_traj_grad_slew(300, NX, 220e-3, pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0))

traj = traj[:,:,:]
traj = np.pi*traj / np.abs(traj).max()

figure_size = 15  # Figure size for trajectory plots
one_shot = -5  # Highlight one shot in particular
show_trajectory(0.5*traj / np.abs(traj).max(), figure_size=figure_size, one_shot=one_shot)
coord = np.ascontiguousarray(traj.reshape(traj.shape[0] * traj.shape[1], 3).transpose(1,0).astype(np.float32))
coord = 0.9999*cp.array(coord)
DN = coord.shape[1]

print('Sample ratio DN / (NX^3): ', DN / (NX**3))


special_precond = False
if special_precond:
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
else:
    grid = cp.stack(cp.meshgrid(
        cp.arange(-NX//2,NX//2), cp.arange(-NX//2,NX//2), cp.arange(-NX//2,NX//2)
    ))
    
    grid_scale = cp.sqrt(cp.sum(cp.square(grid), axis=0).astype(cp.float32)) + 1.0
    grid_scale = cp.fft.fftshift(grid_scale)**(0.5)
    grid_mean = cp.mean(grid_scale)
    grid_scale /= grid_mean
    #ort.image_nd(grid_scale.get())
    

lamda = 0.0

maxAHA_func = lambda x: maxAHA(coord, x, smp, lamda)
max_AHA_precond_func = lambda x: maxAHA(coord, circular_fft(x, grid_scale), smp, lamda)


def conjugate_gradient(A, b, x0, iter=10, inner_iter=10):

    ort.image_nd(x0.get())

    def inner_cg(x,inner_iter):
        r = b - A(x)
        p = r
        rsold = cp.sum(r.conj() * r)
        for i in range(inner_iter):
            Ap = A(p)
            alpha = rsold / cp.sum(p.conj() * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = cp.sum(r.conj() * r)
            beta = (rsnew / rsold)
            p = r + beta * p
            rsold = rsnew
            print('\r', i, cp.sqrt(cp.abs(rsnew)), end='')
        print('')
        return x
    
    x = cp.copy(x0)

    for i in range(iter):
        x = inner_cg(x, inner_iter)
        #x = wavelet_thresholding(x, 0.001)
        ort.image_nd(x.get())

    return x

    



b = cufinufft.nufft3d2(coord[0,:], coord[1,:], coord[2,:], smp * img[None,:])
b /= (np.abs(b).max() / 2)


mean_lamda = 0.001

rhs = cp.sum(smp.conj() * cufinufft.nufft3d1(coord[0,:], coord[1,:], coord[2,:], b, n_modes=img.shape), axis=0)

def mask_filling(x, mask, lamda):
    ret = cp.zeros_like(x)
    ret[mask] = lamda*x[mask]
    return ret

rhs += mask_filling(mean_img, not_vessel_mask, mean_lamda)


prec = cp.ones(rhs.shape, dtype=cp.float32)
prec[vessel_mask] += mean_lamda
#if mean_lamda < 1.0:
#    prec[vessel_mask] += np.sqrt(1.0 / mean_lamda)
#else:
#    prec[vessel_mask] += np.sqrt(mean_lamda)

rhs *= prec

max_AHA_mask_func = lambda x: maxAHA(coord, x, smp, 0.0) + mask_filling(x, not_vessel_mask, mean_lamda)

max_AHA_mask_func_precond = lambda x: prec * max_AHA_mask_func(x)


if False:

    maxeig1, _ = power_iter(maxAHA_func, rand_vec(), 15)
    min_eig_func = lambda x: maxAHA_func(x) - maxeig1*x
    mineig1, _ = power_iter(min_eig_func, rand_vec(), 50)
    mineig1 = -mineig1 + maxeig1

    print('Maxeig: ', maxeig1, 'Mineig: ', mineig1, 'Ratio: ', maxeig1/mineig1)

    maxeig2, _ = power_iter(max_AHA_mask_func, rand_vec(), 15)
    min_eig_func = lambda x: max_AHA_mask_func(x) - maxeig2*x
    mineig2, _ = power_iter(min_eig_func, rand_vec(), 50)
    mineig2 = -mineig2 + maxeig2

    print('Maxeig: ', maxeig2, 'Mineig: ', mineig2, 'Ratio: ', maxeig2/mineig2)

    maxeig3, _ = power_iter(max_AHA_mask_func_precond, rand_vec(), 15)
    min_eig_func = lambda x: max_AHA_mask_func_precond(x) - maxeig3*x
    mineig3, _ = power_iter(min_eig_func, rand_vec(), 50)
    mineig3 = -mineig3 + maxeig3

    print('Maxeig: ', maxeig3, 'Mineig: ', mineig3, 'Ratio: ', maxeig3/mineig3)



#xback = conjugate_gradient(maxAHA_func, rhs, rhs, iter=50)
#ort.image_nd(xback.get())

back = conjugate_gradient(max_AHA_mask_func_precond, rhs, cp.array(mean_img), iter=25, inner_iter=30)
#ort.image_nd(back.get())
#xback_precond = circular_fft(back, grid_scale)
#rt.image_nd(xback_precond.get())
ort.image_nd(back.get())





