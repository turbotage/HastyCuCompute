import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp
from pypulseq.convert import convert

import scipy as sp
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
import scipy.special as sps

from sequtil import SafetyLimits, LTIGradientKernels

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

class SpoilerFactory:
    def __init__(self, area: float, system: pp.Opts, sl: SafetyLimits):
        self.area = area
        self.system = system
        self.sl = sl

    def create_spoiler(self, convolve_odd_n, convolve_sigma=None):
        safe_system = self.system
        safe_system.max_grad = self.system.max_grad * self.sl.grad_ratio
        safe_system.max_slew = self.system.max_slew * self.sl.slew_ratio

        trapz = pp.make_trapezoid(
            channel='z',
            area=self.area,
            system=safe_system
        )
        RISE_NGRT = int(trapz.rise_time / system.grad_raster_time)
        FLAT_NGRT = int(trapz.flat_time / system.grad_raster_time)
        FALL_NGRT = int(trapz.fall_time / system.grad_raster_time)
        assert(RISE_NGRT == FALL_NGRT)

        damp_rise = trapz.amplitude / RISE_NGRT
        t = [0.0]
        for i in range(RISE_NGRT):
            t.append(t[-1] + damp_rise)
        for i in range(FLAT_NGRT):
            t.append(t[-1])
        for i in range(FALL_NGRT):
            t.append(t[-1] - damp_rise)
        t.append(0.0)

        if convolve_sigma is None:
            convolve_sigma = convolve_odd_n / 2.0
        gauss = np.arange(2*convolve_odd_n+1)-convolve_odd_n
        gauss = np.exp(-(gauss/convolve_sigma)**2)
        gauss /= np.sum(gauss)
        gauss = np.concatenate((np.zeros(2), gauss, np.zeros(2)))
        np.convolve(t, gauss)

        print(f"Spoiler gradient created with area {self.area} mT/m*ms")


if __name__ == "__main__":
    system = pp.Opts(
        max_grad=40, grad_unit='mT/m',
        max_slew=150, slew_unit='T/m/s',
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
        adc_raster_time=10e-6
    )

    sl = SafetyLimits()

    spoiler_area = 2.0 / (1e-3) # This is (gamma / 2pi) * 4pi/(gamma * delta_x)

    spoiler_factory = SpoilerFactory(spoiler_area, system, sl)
    spoiler = spoiler_factory.create_spoiler(2)