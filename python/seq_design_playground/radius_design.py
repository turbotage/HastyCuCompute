import math
from functools import partial
from typing import Literal
import scipy as sp

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.special import ellipj, ellipk
from scipy.interpolate import PchipInterpolator

import torch

def rho1(a, c, n, m):
	def g(t):
		if isinstance(t, torch.Tensor):
			return torch.square(t) * (1 - torch.pow(t, n)) / (t+a)
		elif isinstance(t, float):
			return t*t*(1 - t**n)/(t+a)
		else:
			raise RuntimeError("t must be float or torch.Tensor")

	def f(t):
		if isinstance(t, torch.Tensor):
			mask = t < 1.0 - 1e-6
			t[mask.logical_not()] = 0.0
			t[mask] = torch.e * torch.exp(-1/torch.pow(1-torch.pow(t[mask],m), c))
			return t
		elif isinstance(t, float):
			return math.e * math.exp(-1/math.pow(1-math.pow(t,m), c))
		else:
			raise RuntimeError("t must be float or torch.Tensor")

	def rho(t):
		return g(t) * f(t)

	def rhodt(t):
		h = 1e-6
		return (rho(t+h) - rho(t)) / h

	maximum = sp.optimize.brentq(rhodt, 0.5, 0.96)
	maximum = rho(maximum)

	return lambda t: torch.tensor(rho(t) / maximum)

def rho2(a, n, b):
	def g(t):
		return torch.square(t) * (1 - torch.pow(t, n)) / (t+a)

	def f(t):
		s = (t - (1-b)) / b
		m = torch.clamp(s, 0.0, 1.0)
		return torch.square(m) * (3 - 2*m)

	def rho(t):
		return g(t) * f(t)

	def rhodt(t):
		h = 1e-6
		return (rho(t+h) - rho(t)) / h

	maximum = sp.optimize.brentq(rhodt, 0.5, 0.98)
	maximum = rho(maximum)

	return lambda t: torch.tensor(rho(t) / maximum)

def rho3():
	pchip = PchipInterpolator(
		x=[0.0, 0.001, 0.10, 0.3, 0.85, 0.9, 0.95, 0.999,  1.0],
		y=[0.0, 0.0,   0.6,  0.8, 1.0,  0.8, 0.5,  0.0,    0.0]
	)

	return lambda t: torch.tensor(pchip(t.detach().numpy()))

def non_returning_rho3():
	pchip = PchipInterpolator(
		x=[0.0, 0.001, 0.10, 0.30, 1.0],
		y=[0.0, 0.000, 0.50, 0.80, 1.0]
	)

	return lambda t: torch.tensor(pchip(t.detach().numpy()))

def learnable_rho(a, b):
	if isinstance(a, torch.Tensor) == False:
		a = torch.tensor(a, dtype=torch.float64)
	if isinstance(b, torch.Tensor) == False:
		b = torch.tensor(b, dtype=torch.float64)

	def rho(t):
		return torch.exp(-1/(a*t))*(1 - torch.exp(-b*t)) / (torch.exp(-1/a)*(1 - torch.exp(-b)))

	return lambda t: rho(t)

if __name__ == "__main__":
	
	rho_func = learnable_rho(10, 5)
	t = torch.linspace(0,1,1000, dtype=torch.float64)
	import matplotlib.pyplot as plt
	rho = rho_func(t)
	plt.figure()
	plt.plot(t.detach().numpy(), rho.detach().numpy())
	plt.show()