import math
import torch

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
        Dk = 1.5 / grad_raster_time

        n_grad_raster_times = 6
        t = torch.linspace(0, n_grad_raster_times*grad_raster_time, n_grad_raster_times*kernel_oversampling)
        kernel = (Dk*t)**(k1) * torch.exp(-k2*Dk*t)
        kernel /= torch.sum(kernel)

        kernels = {'x': kernel, 'y': kernel, 'z': kernel}

        return kernels

    def get(self, channel='x'):
        return self.kernels[channel]
    
    def oversamp(self):
        return self.kernel_oversampling
    
from typing import Iterable, Union

# From pypulseq
def convert(
    from_value: Union[float, Iterable],
    from_unit: str,
    gamma: float = 42.576e6,
    to_unit: str = str(),
) -> Union[float, Iterable]:
    """
    Converts gradient amplitude or slew rate from unit `from_unit` to unit `to_unit` with gyromagnetic ratio `gamma`.

    Parameters
    ----------
    from_value : float
        Gradient amplitude or slew rate to convert from.
    from_unit : str
        Unit of gradient amplitude or slew rate to convert from.
    to_unit : str, default=''
        Unit of gradient amplitude or slew rate to convert to.
    gamma : float, default=42.576e6
        Gyromagnetic ratio. Default is 42.576e6, for Hydrogen.

    Returns
    -------
    out : float
        Converted gradient amplitude or slew rate.

    Raises
    ------
    ValueError
        If an invalid `from_unit` is passed. Must be one of 'Hz/m', 'mT/m', or 'rad/ms/mm'.
        If an invalid `to_unit` is passed. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms'.
    """
    valid_grad_units = ["Hz/m", "mT/m", "rad/ms/mm"]
    valid_slew_units = ["Hz/m/s", "mT/m/ms", "T/m/s", "rad/ms/mm/ms"]
    valid_units = valid_grad_units + valid_slew_units

    if from_unit not in valid_units:
        raise ValueError(
            "Invalid from_unit. Must be one of 'Hz/m', 'mT/m', or 'rad/ms/mm' for gradients;"
            "or must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms' for slew rate."
        )

    if to_unit != "" and to_unit not in valid_units:
        raise ValueError(
            "Invalid to_unit. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms' for gradients;"
            "or must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms' for slew rate.."
        )

    if to_unit == "":
        if from_unit in valid_grad_units:
            to_unit = valid_grad_units[0]
        elif from_unit in valid_slew_units:
            to_unit = valid_slew_units[0]

    # Convert to standard units
    # Grad units
    if from_unit == "Hz/m":
        standard = from_value
    elif from_unit == "mT/m":
        standard = from_value * 1e-3 * gamma
    elif from_unit == "rad/ms/mm":
        standard = from_value * 1e6 / (2 * torch.pi)
    # Slew units
    elif from_unit == "Hz/m/s":
        standard = from_value
    elif from_unit == "mT/m/ms" or from_unit == "T/m/s":
        standard = from_value * gamma
    elif from_unit == "rad/ms/mm/ms":
        standard = from_value * 1e9 / (2 * torch.pi)

    # Convert from standard units
    # Grad units
    if to_unit == "Hz/m":
        out = standard
    elif to_unit == "mT/m":
        out = 1e3 * standard / gamma
    elif to_unit == "rad/ms/mm":
        out = standard * 2 * torch.pi * 1e-6
    # Slew units
    elif to_unit == "Hz/m/s":
        out = standard
    elif to_unit == "mT/m/ms" or to_unit == "T/m/s":
        out = standard / gamma
    elif to_unit == "rad/ms/mm/ms":
        out = standard * 2 * torch.pi * 1e-9

    return out

