import mrinufft as mn

import numpy as np


class SafetyLimits:
    def __init__(self, slew_ratio=0.9, grad_ratio=0.9):
        self.slew_ratio = slew_ratio
        self.grad_ratio = grad_ratio

class LTIGradientKernels:
    def __init__(self, system, kernels, kernel_oversampling=100):
        self.system = system
        self.kernel_oversampling = kernel_oversampling
        self.kernels = kernels

    def get(self, channel='x'):
        return self.kernels[channel]
    
    def oversamp(self):
        return self.kernel_oversampling
    
    def transfer_gradient(self, grad, channel):
        grad_oversamp = mn.oversample(grad[None,:], grad.shape[0]*self.kernel_oversampling, kind='linear')[0,:]
        dt = self.system.grad_raster_time / self.kernel_oversampling

        grad_corr = np.convolve(grad_oversamp, self.kernels[channel], mode='full')
        t_corr = np.linspace(0, dt*grad_corr.shape[0], grad_corr.shape[0])

        return grad_corr, t_corr