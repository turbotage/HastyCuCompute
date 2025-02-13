import mrinufft as mn
import math
import numpy as np

class ImageProperties:
    def __init__(self, shape, fov, resolution):
        self.shape = shape
        self.fov = fov
        self.resolution = resolution

class SafetyLimits:
    def __init__(self, slew_ratio=0.9, grad_ratio=0.9):
        self.slew_ratio = slew_ratio
        self.grad_ratio = grad_ratio

class LTIGradientKernels:
    def __init__(self, system, kernels, kernel_oversampling=100):
        if system is None:
            raise ValueError("System must be provided")
        
        self.system = system
        self.kernel_oversampling = kernel_oversampling
        self.kernels = kernels

        self.max_kernel_length = max([math.ceil(kernel.shape[0]/kernel_oversampling) for kernel in kernels.values()])

    @staticmethod
    def kernels_from_test(grad_raster_time, kernel_oversampling=100):
        k1 = 8.0
        k2 = 1.5
        Dk = 6.0 / grad_raster_time

        n_grad_raster_times = 3
        t = np.linspace(0, n_grad_raster_times*grad_raster_time, n_grad_raster_times*kernel_oversampling)
        kernel = (Dk*t)**(k1) * np.exp(-k2*Dk*t)
        kernel /= np.sum(kernel)

        kernels = {'x': kernel, 'y': kernel, 'z': kernel}

        return kernels

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
    
    def transfer_gradient_downsamp(self, grad, channel):
        grad_corr = self.transfer_gradient(grad, channel)
        return grad_corr[0][::self.kernel_oversampling], grad_corr[1][::self.kernel_oversampling]
    
def test_kernel(system):
	kernel_oversampling = 100
	k1 = 8.0
	k2 = 1.5
	Dk = 6.0 / system.grad_raster_time

	n_grad_raster_times = 3
	t = np.linspace(0, n_grad_raster_times*system.grad_raster_time, n_grad_raster_times*kernel_oversampling)
	kernel = (Dk*t)**(k1) * np.exp(-k2*Dk*t)
	kernel /= np.sum(kernel)

	return kernel, kernel_oversampling


